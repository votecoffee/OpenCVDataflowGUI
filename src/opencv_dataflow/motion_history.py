"""Timestamped motion age, direction and region masks using OpenCV motion templates."""

from heapq import nlargest

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .result_graphics import ACCEPTED
from .sequence import SEQUENCE_DEFAULTS, SEQUENCE_HELP, SequenceBlock, SequenceStep

MAX_PIXELS = 1_048_576


def frame_time(inputs, payload):
    video = payload.get("video_frame", {})
    video = video if isinstance(video, dict) else {}
    value = inputs.get("timestamp_ms", video.get("timestamp_ms"))
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1e15:
        raise ValidationError("Provide a finite, nonnegative capture timestamp in milliseconds")
    source = "input" if "timestamp_ms" in inputs else "video"
    return float(value), source


def motion_regions(motempl, history, timestamp, tolerance, limit):
    native_labels, rectangles = motempl.segmentMotion(history, timestamp, tolerance)
    labels = native_labels.astype(np.int32)
    if not len(rectangles):
        return labels, []
    counts = np.bincount(labels.ravel())
    indices = nlargest(limit, range(len(rectangles)), key=lambda i: (counts[i + 1], -i))
    segments = [
        {
            "label": index + 1,
            "rectangle": [int(v) for v in rectangles[index]],
            "pixels": int(counts[index + 1]),
        }
        for index in indices
    ]
    return labels, segments


class MotionHistory(SequenceBlock):
    type_id, title, subtype = "sequence.motion_history", "Motion history and direction", "Motion"
    description = "Track when pixels moved and estimate the direction of recent motion."
    inputs = (
        *SequenceBlock.inputs,
        Port("timestamp_ms", "number", required=False, required_when_connected=True),
    )
    data_outputs = (
        Port("recency"),
        Port("age_ms"),
        Port("active", visual_type="region"),
        Port("orientation_deg"),
        Port("orientation_mask", visual_type="region"),
        Port("direction_deg", "number", required=False),
        Port("labels", required=False),
        Port("segments", "data", required=False),
    )
    process_inputs_readonly = True
    payload_reference = "image"
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 1,
        "duration_ms": 250.0,
        "minimum_delta_ms": 1.0,
        "maximum_delta_ms": 0.0,
        "aperture": "3",
        "segment_regions": True,
        "segment_tolerance_ms": 60.0,
        "maximum_regions": 64,
    }
    parameter_choices = {**SequenceBlock.parameter_choices, "aperture": ("3", "5", "7")}
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0),
        "duration_ms": NumericRange(1, 600000),
        "minimum_delta_ms": NumericRange(0.01, 600000),
        "maximum_delta_ms": NumericRange(0, 600000),
        "segment_tolerance_ms": NumericRange(0, 600000),
        "maximum_regions": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "duration_ms": "Keep pixels up to this age in milliseconds.",
        "minimum_delta_ms": "Smallest valid neighborhood time change for the motion gradient.",
        "maximum_delta_ms": "Largest valid local time change. Zero uses the history duration.",
        "aperture": "Sobel aperture for local motion direction, in pixels.",
        "segment_regions": "Group adjacent motion history around the current silhouette.",
        "segment_tolerance_ms": "Maximum time difference between neighboring region pixels.",
        "maximum_regions": "Maximum number of region records. The label raster keeps all regions.",
    }
    input_help = {
        "image": "Uint8 silhouette. Nonzero input pixels indicate motion at this frame time.",
        "timestamp_ms": "Capture time in milliseconds, or the time from video metadata.",
        "mask": "Restrict motion updates and reported history to the selected pixels.",
        "region": "Limit updates and outputs, intersected with the optional mask.",
        "reset": "Clear this stream's history before updating with the current silhouette.",
    }
    output_help = {
        "recency": "Float32 map that fades from 1 for current motion to 0 at the time limit.",
        "age_ms": "Elapsed milliseconds since motion. Inactive pixels have a value of −1.",
        "active": "Uint8 mask of pixels with history within the configured time window.",
        "orientation_deg": "Local motion direction. Use orientation_mask to select valid angles.",
        "orientation_mask": "Uint8 mask of valid motion gradients.",
        "direction_deg": "Dominant motion direction in degrees clockwise from image-right.",
        "labels": "Int32 labels for regions that touch the newest silhouette.",
        "segments": "Largest reported region boxes, pixel counts and IDs from the label raster.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["recency", "active", "direction_deg", "segments", "status"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["segment_regions"]) is not bool:
            raise ValidationError("Segment regions must be boolean", "segment_regions")
        duration = float(p["duration_ms"])
        maximum = float(p["maximum_delta_ms"]) or duration
        if not float(p["minimum_delta_ms"]) <= maximum <= duration:
            raise ValidationError(
                "Motion gradient minimum must be within the maximum and history duration",
                "minimum_delta_ms",
                "maximum_delta_ms",
                "duration_ms",
            )

    def validate_sequence_image(self, image):
        if (
            image.ndim != 2
            or image.dtype != np.uint8
            or min(image.shape) < 7
            or image.size > MAX_PIXELS
        ):
            raise ValidationError(
                "Use a uint8 silhouette, at least 7 px per axis and at most 1,048,576 pixels"
            )

    def estimate_state_bytes(self, image):
        return image.size * np.dtype(np.float64).itemsize

    def step(self, inputs, state, mask):
        motempl = getattr(cv2, "motempl", None)
        if motempl is None:
            raise ValidationError(
                "Motion history requires the OpenCV contrib motion templates module"
            )
        p = self.parameters
        timestamp, source = frame_time(inputs, self.input_payloads["image"])
        duration = float(p["duration_ms"])
        silhouette = np.asarray(inputs["image"])
        if mask is not None:
            silhouette = cv2.bitwise_and(silhouette, silhouette, mask=mask)
        moving = silhouette != 0
        last_seen = state.get("last_seen")
        restart = last_seen is not None and (
            source != state["clock_source"] or timestamp < state["timestamp_ms"]
        )
        if last_seen is None or restart:
            last_seen = state["last_seen"] = np.full(silhouette.shape, -1, np.float64)
            state["learning"] = 0
        state["clock_source"], state["timestamp_ms"] = source, timestamp

        # OpenCV stores float32 times and reserves zero for inactive pixels.
        # Map each window onto a small positive clock while retaining capture
        # timestamps in float64 for expiration and published pixel ages.
        age = timestamp - last_seen
        recent = (last_seen >= 0) & (age <= duration)
        native_now = float(np.float32(3 * duration))
        history = np.zeros(silhouette.shape, np.float32)
        history[recent] = native_now - age[recent]
        last_seen[~recent] = -1
        # Exact expiration happened above. Leave one float32 clock step of
        # leeway for native implementations that round duration before subtracting.
        native_duration = duration + float(np.spacing(np.float32(native_now)))
        history = motempl.updateMotionHistory(silhouette, history, native_now, native_duration)
        last_seen[moving] = timestamp
        age[moving] = 0
        if mask is not None:
            history[mask == 0] = 0
        active = history > 0

        orientation_mask, angles = motempl.calcMotionGradient(
            history,
            float(p["minimum_delta_ms"]),
            float(p["maximum_delta_ms"]) or duration,
            apertureSize=int(p["aperture"]),
        )
        orientation_mask = np.where((orientation_mask != 0) & active, np.uint8(255), np.uint8(0))
        angles[orientation_mask == 0] = 0
        gradient_count = cv2.countNonZero(orientation_mask)
        direction = (
            float(
                motempl.calcGlobalOrientation(
                    angles, orientation_mask, history, native_now, duration
                )
            )
            % 360
            if gradient_count
            else NO_VALUE
        )

        labels, segments = NO_VALUE, NO_VALUE
        if p["segment_regions"]:
            if moving.any():
                # Older timestamps can round up to the newest native clock value.
                # Seed regions only on pixels of the incoming silhouette.
                history[(history == native_now) & ~moving] = np.nextafter(
                    np.float32(native_now), np.float32(0)
                )
                labels, segments = motion_regions(
                    motempl,
                    history,
                    native_now,
                    float(p["segment_tolerance_ms"]),
                    int(p["maximum_regions"]),
                )
            else:
                labels, segments = np.zeros(silhouette.shape, np.int32), []
            labels = ImageArray(labels, "gray")

        recency = np.zeros(silhouette.shape, np.float32)
        recency[active] = 1 - age[active] / duration
        return SequenceStep(
            {
                "recency": ImageArray(recency, "gray"),
                "age_ms": ImageArray(np.where(active, age, -1).astype(np.float32), "gray"),
                "active": ImageArray(np.where(active, np.uint8(255), np.uint8(0)), "gray"),
                "orientation_deg": ImageArray(angles, "gray"),
                "orientation_mask": ImageArray(orientation_mask, "gray"),
                "direction_deg": direction,
                "labels": labels,
                "segments": segments,
            },
            details={
                "timestamp_ms": timestamp,
                "timestamp_source": source,
                "clock_restarted": restart,
                "active_pixels": int(np.count_nonzero(active)),
                "orientation_pixels": gradient_count,
            },
        )

    def process(self, inputs):
        result = super().process(inputs)
        direction = result.outputs["direction_deg"]
        segments = result.outputs["segments"]
        if self.generate_graphics:
            objects = []
            if segments is not NO_VALUE:
                objects.extend(graphic_box(*segment["rectangle"], ACCEPTED) for segment in segments)
            if direction is not NO_VALUE:
                x, y, width, height = cv2.boundingRect(
                    np.asarray(result.outputs["orientation_mask"])
                )
                center = np.array([x + (width - 1) / 2, y + (height - 1) / 2])
                radians = np.deg2rad(direction)
                end = center + 20 * np.array([np.cos(radians), np.sin(radians)])
                objects.append(graphic_object("arrow", (center, end), (60, 220, 255)))
            if objects:
                result.drawings["motion history"] = GraphicDrawing(tuple(objects), "recency", True)
        return result
