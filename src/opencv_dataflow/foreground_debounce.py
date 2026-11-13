"""Consecutive, unambiguous component confirmation on current foreground masks."""

import copy
import hashlib

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import ImageArray
from .objects import blob_item, collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import selection_mask
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)


def compatible_components(current, previous, displacement, area_ratio, minimum_iou):
    """Bounded vectorized gates; mutual uniqueness is applied by the caller."""
    centers = np.array([item["center"] for item in current], float).reshape(-1, 2)
    old_centers = np.array([item["center"] for item in previous], float).reshape(-1, 2)
    distance2 = np.sum((centers[:, None] - old_centers[None]) ** 2, axis=2)
    areas = np.array([item["area"] for item in current], float)[:, None]
    old_areas = np.array([item["area"] for item in previous], float)[None]
    permitted = (distance2 <= displacement**2) & (
        np.minimum(areas, old_areas) >= area_ratio * np.maximum(areas, old_areas)
    )
    if minimum_iou and current and previous:
        boxes = np.array([item["bounds"] for item in current], float)[:, None]
        old_boxes = np.array([item["bounds"] for item in previous], float)[None]
        extents = np.maximum(
            0,
            np.minimum(boxes[..., :2] + boxes[..., 2:], old_boxes[..., :2] + old_boxes[..., 2:])
            - np.maximum(boxes[..., :2], old_boxes[..., :2]),
        )
        intersection = extents.prod(axis=2)
        union = boxes[..., 2:].prod(axis=2) + old_boxes[..., 2:].prod(axis=2) - intersection
        permitted &= intersection >= minimum_iou * union
    return permitted


class ForegroundDebounce(SequenceBlock):
    type_id, title, subtype = (
        "sequence.foreground_debounce",
        "Debounce foreground components",
        "Foreground masks",
    )
    inputs = (Port("image", visual_type="mask"), *SequenceBlock.inputs[1:])
    data_outputs = (
        Port("mask", visual_type="mask"),
        Port("rejected_mask", visual_type="mask"),
        Port("components", "data"),
        Port("count", "number"),
        Port("observations", "data"),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 3,
        "consecutive_hits": 3,
        "foreground_threshold": 0.0,
        "connectivity": "8",
        "minimum_area": 4,
        "maximum_area": 0,
        "maximum_displacement": 10.0,
        "minimum_area_ratio": 0.5,
        "minimum_iou": 0.0,
        "maximum_components": 256,
    }
    parameter_choices = {**SEQUENCE_CHOICES, "connectivity": ("4", "8")}
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "consecutive_hits": NumericRange(1, 10000, whole=True),
        "foreground_threshold": NumericRange(),
        "minimum_area": NumericRange(1, 64_000_000, whole=True),
        "maximum_area": NumericRange(0, 64_000_000, whole=True),
        "maximum_displacement": NumericRange(0, 32768),
        "minimum_area_ratio": NumericRange(0, 1),
        "minimum_iou": NumericRange(0, 1),
        "maximum_components": NumericRange(1, 512, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.001, 4096),
        "max_gap_ms": NumericRange(0),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "consecutive_hits": (
            "Required consecutive eligible unambiguous observations. "
            "Missing or ambiguous matches restart confirmation."
        ),
        "foreground_threshold": (
            "Samples strictly above this raw-value threshold are foreground. "
            "Use 127 to exclude 127-valued shadow labels."
        ),
        "maximum_area": "Zero disables the upper area limit. Area is measured after selection.",
        "maximum_displacement": (
            "Maximum centroid motion per activation in current-grid pixels. "
            "No prediction or gap bridging."
        ),
        "minimum_area_ratio": "Smaller/larger area must meet this ratio; zero disables the gate.",
        "minimum_iou": (
            "Minimum bounding-box intersection over union. "
            "Zero disables overlap gating and allows disjoint moving masks."
        ),
        "maximum_components": (
            "Eligible components per frame. Exceeding the bound fails and clears sequence memory."
        ),
    }
    input_help = {
        "image": "Single-channel foreground image; numeric samples are thresholded explicitly."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["maximum_area"] and p["maximum_area"] < p["minimum_area"]:
            raise ValidationError(
                "Maximum area must be zero or at least minimum area", "maximum_area"
            )

    def extra_context(self, inputs):
        mask = selection_mask(inputs, inputs["image"])
        return hashlib.sha256(mask.tobytes()).hexdigest() if mask is not None else None

    def estimate_state_bytes(self, image):
        # No masks, images or historical tickets are retained per component.
        return int(self.parameters["maximum_components"]) * 2048 + 4096

    def step(self, inputs, state, mask):
        source = image_value(inputs["image"])
        if source.ndim != 2:
            raise ValidationError("Foreground debounce requires a single-channel image")
        p = self.parameters
        binary = (source > float(p["foreground_threshold"])).astype(np.uint8)
        if mask is not None:
            binary[mask == 0] = 0
        number, labels, stats, centers = cv2.connectedComponentsWithStats(
            binary, connectivity=int(p["connectivity"]), ltype=cv2.CV_32S
        )
        areas = stats[1:, cv2.CC_STAT_AREA]
        eligible = areas >= int(p["minimum_area"])
        if p["maximum_area"]:
            eligible &= areas <= int(p["maximum_area"])
        indices = np.flatnonzero(eligible) + 1
        if len(indices) > int(p["maximum_components"]):
            raise ValidationError(
                "Foreground component limit exceeded; reduce noise or raise the limit"
            )
        current = [
            {
                "label": int(index),
                "bounds": stats[index, :4].tolist(),
                "area": int(stats[index, cv2.CC_STAT_AREA]),
                "center": centers[index].tolist(),
            }
            for index in indices
        ]
        previous = state.get("observations", [])
        permitted = compatible_components(
            current,
            previous,
            float(p["maximum_displacement"]),
            float(p["minimum_area_ratio"]),
            float(p["minimum_iou"]),
        )
        row_counts, column_counts = permitted.sum(axis=1), permitted.sum(axis=0)
        required = int(p["consecutive_hits"])
        lookup = np.zeros(number, np.uint8)
        items = []
        matched = ambiguous = 0
        for index, observation in enumerate(current):
            association = "new"
            old = None
            if row_counts[index] == 1:
                old_index = int(np.argmax(permitted[index]))
                if column_counts[old_index] == 1:
                    old = previous[old_index]
                    association = "matched"
                else:
                    association = "ambiguous"
            elif row_counts[index] > 1:
                association = "ambiguous"
            if old is None:
                state["next_id"] = state.get("next_id", 0) + 1
                identifier, hits = state["next_id"], 1
            else:
                identifier, hits = old["identifier"], min(required, old["hits"] + 1)
            observation.update(
                identifier=identifier,
                hits=hits,
                confirmed=hits >= required,
                association=association,
                track_id=f"{state['generation']}:{identifier}",
            )
            matched += association == "matched"
            ambiguous += association == "ambiguous"
            if observation["confirmed"]:
                label = observation["label"]
                lookup[label] = 255
                x, y, width, height = observation["bounds"]
                roi = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
                item = blob_item(roi, x, y, len(items))
                item["temporal"] = {
                    "track_id": observation["track_id"],
                    "consecutive_hits": hits,
                    "required_hits": required,
                    "association": association,
                }
                items.append(item)
        accepted = lookup[labels]
        rejected = (binary * 255) & cv2.bitwise_not(accepted)
        # Result snapshots and downstream plugins cannot mutate retained history.
        state["observations"] = copy.deepcopy(current)
        output = collection("blobs", items, source, self.input_payloads["image"])
        return SequenceStep(
            {
                "mask": ImageArray(accepted, "gray"),
                "rejected_mask": ImageArray(rejected, "gray"),
                "components": output,
                "count": len(items),
                "observations": current,
            },
            details={
                "raw_components": number - 1,
                "eligible_components": len(current),
                "confirmed_components": len(items),
                "area_rejected_components": number - 1 - len(current),
                "matched_components": matched,
                "ambiguous_components": ambiguous,
                "required_consecutive_hits": required,
                "association_policy": "mutually_unique; no prediction or absent-component carry",
                "coordinates": "current input grid; only currently observed pixels are emitted",
            },
        )

    def process(self, inputs):
        result = super().process(inputs)
        if self.generate_graphics:
            objects = []
            observations = result.outputs["observations"]
            for observation in () if observations is NO_VALUE else observations:
                color = (40, 210, 40) if observation["confirmed"] else (0, 190, 255)
                x, y, width, height = observation["bounds"]
                objects.append(graphic_box(x, y, width, height, color))
                objects.append(
                    graphic_object(
                        "text",
                        ((x, max(0, y - 3)),),
                        color,
                        text=f"{observation['hits']}/{self.parameters['consecutive_hits']}",
                    )
                )
            if objects:
                result.drawings["foreground confirmation"] = GraphicDrawing(tuple(objects))
        return result
