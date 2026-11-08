"""Locate ArUco fiducials and emit unambiguous image-space correspondences."""

from collections import Counter

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .fiducial_detection import DICTIONARIES, MARKER_CHOICES, MARKER_OPTIONS, detector_parameters
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, contains, selection_mask

_POINT_CORNERS = {"corner_0": 0, "corner_1": 1, "corner_2": 2, "corner_3": 3}


def _projective_center(corners):
    a, b, c, d = np.column_stack((corners, np.ones(4)))
    center = np.cross(np.cross(a, c), np.cross(b, d))
    if abs(center[2]) < 1e-9:
        raise ValidationError("Detected marker corners do not define a finite center")
    return center[:2] / center[2]


class DetectArucoMarkers(Block):
    type_id, title = "inspect.aruco_markers", "Detect ArUco markers"
    category, block_type, subtype = "Inspection", "Inspection", "Fiducials"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("markers", "data"),
        Port("points", "data", required=False),
        Port("count", "number"),
        Port("overlay", required=False),
    )
    defaults = {
        "dictionary": "4x4_50",
        **MARKER_OPTIONS,
        "required_ids": [],
        "minimum_markers": 1,
        "point": "center",
        "draw_overlay": True,
    }
    parameter_choices = {**MARKER_CHOICES, "point": ("center", *_POINT_CORNERS)}
    parameter_ranges = {
        "adaptive_min": NumericRange(3, 255, whole=True),
        "adaptive_max": NumericRange(3, 1000, whole=True),
        "adaptive_step": NumericRange(1, 100, whole=True),
        "adaptive_constant": NumericRange(),
        "min_perimeter_rate": NumericRange(0.0001),
        "max_perimeter_rate": NumericRange(0.0001),
        "min_border_distance": NumericRange(0, 1000, whole=True),
        "border_bits": NumericRange(1, 10, whole=True),
        "error_correction_rate": NumericRange(0, 1),
        "minimum_markers": NumericRange(1, whole=True),
    }
    parameter_help = {
        "required_ids": "Emit points in this ID order. Empty emits all IDs in ascending order.",
        "minimum_markers": "Minimum number of unique selected IDs needed to pass.",
        "point": "Use the projected center or a corner in the marker's decoded orientation.",
    }
    input_help = {
        "mask": "Keep markers whose projected center lies inside this mask.",
        "region": "Keep markers whose projected center lies inside this region.",
    }
    process_inputs_readonly = True
    composes_result_image = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for low, high in (
            ("adaptive_min", "adaptive_max"),
            ("min_perimeter_rate", "max_perimeter_rate"),
        ):
            if float(p[low]) > float(p[high]):
                raise ValidationError(
                    f"{high.replace('_', ' ').capitalize()} must be at least the minimum", high
                )
        capacity = len(cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]]).bytesList)
        ids = p["required_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) > capacity
            or any(type(number) is not int or not 0 <= number < capacity for number in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValidationError(
                f"Use unique marker IDs from 0 through {capacity - 1}", "required_ids"
            )
        if int(p["minimum_markers"]) > capacity:
            raise ValidationError("Minimum markers exceeds dictionary capacity", "minimum_markers")

    def process(self, inputs):
        source = inputs["image"]
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        mask = selection_mask(inputs, gray)
        p = self.parameters
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]])
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, detector_parameters(p)).detectMarkers(
            gray
        )
        markers = []
        if ids is not None:
            for number, corner in zip(ids.ravel(), corners, strict=True):
                xy = corner.reshape(4, 2).astype(np.float64)
                center = _projective_center(xy)
                if contains(mask, center):
                    markers.append(
                        {"id": int(number), "corners": xy.tolist(), "center": center.tolist()}
                    )
        markers.sort(key=lambda row: row["id"])
        counts = Counter(row["id"] for row in markers)
        order = p["required_ids"] or list(counts)
        missing = [number for number in order if counts[number] == 0]
        duplicates = [number for number in order if counts[number] > 1]
        reasons = []
        if missing:
            reasons.append(
                {
                    "code": "aruco_missing_ids",
                    "message": f"Missing {len(missing)} marker IDs. First IDs {missing[:12]}",
                }
            )
        if duplicates:
            reasons.append(
                {
                    "code": "aruco_duplicate_ids",
                    "message": f"Ambiguous IDs ({len(duplicates)}). First IDs {duplicates[:12]}",
                }
            )
        if len(counts) < int(p["minimum_markers"]):
            reasons.append("aruco_marker_count_below_minimum")

        points = NO_VALUE
        if not reasons:
            by_id = {row["id"]: row for row in markers}
            index = _POINT_CORNERS.get(p["point"])
            points = [
                by_id[number]["center"] if index is None else by_id[number]["corners"][index]
                for number in order
            ]

        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        drawings = {}
        if markers and (self.generate_graphics or overlay is not NO_VALUE):
            color = REJECTED if reasons else ACCEPTED
            objects = tuple(
                graphic
                for marker in markers
                for graphic in (
                    graphic_object("polyline", marker["corners"], color, closed=True),
                    graphic_object("label", [marker["center"]], color, text=str(marker["id"])),
                )
            )
            if self.generate_graphics:
                drawings["ArUco markers"] = GraphicDrawing(objects)
            if overlay is not NO_VALUE:
                identity = np.eye(3)
                for graphic in objects:
                    draw_object(overlay, graphic, identity)

        return Result(
            {"markers": markers, "points": points, "count": len(markers), "overlay": overlay},
            drawings=drawings,
            passed=not reasons,
            fail_reasons=reasons,
            message=f"Detected {len(markers)} markers with {len(counts)} unique IDs",
        )
