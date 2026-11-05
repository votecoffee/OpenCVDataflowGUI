"""Seeded segmentation in the input image's coordinate frame."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .payloads import require_same_frame
from .processing import image_value, integer
from .selection import SELECTION_INPUTS, selection_mask


def bgr8(source):
    image = image_value(source, (np.uint8,), canonical=True)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class FloodFill(Block):
    type_id, title = "inspect.flood_fill", "Seeded flood fill"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("mask"), Port("area", "number"), Port("bounds", "data"))
    defaults = {
        "seed_x": 0,
        "seed_y": 0,
        "lower_difference": 0.0,
        "upper_difference": 0.0,
        "connectivity": "4",
        "range_mode": "fixed",
    }
    parameter_choices = {"connectivity": ("4", "8"), "range_mode": ("fixed", "floating")}
    parameter_help = {
        "lower_difference": "Nonnegative tolerance below the reference value. Number or BGR list.",
        "upper_difference": "Nonnegative tolerance above the reference value. Number or BGR list.",
        "range_mode": "Compare with the seed (fixed) or an accepted adjacent pixel (floating).",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.float32), canonical=True)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        x = integer(self.parameters["seed_x"], "Seed x", 0, width - 1)
        y = integer(self.parameters["seed_y"], "Seed y", 0, height - 1)
        channels = 1 if image.ndim == 2 else 3
        tolerances = []
        for key in ("lower_difference", "upper_difference"):
            values = np.asarray(self.parameters[key], dtype=np.float64).reshape(-1)
            if values.size == 1:
                values = np.repeat(values, channels)
            if values.size != channels or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{key} needs one or {channels} finite nonnegative values")
            tolerances.append(tuple(values))
        selection = selection_mask(inputs, image)
        mask = np.zeros((height + 2, width + 2), np.uint8)
        interior = mask[1:-1, 1:-1]
        if selection is not None:
            interior[selection == 0] = 1
        flags = int(self.parameters["connectivity"]) | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        if self.parameters["range_mode"] == "fixed":
            flags |= cv2.FLOODFILL_FIXED_RANGE
        area, _, _, bounds = cv2.floodFill(
            np.ascontiguousarray(image), mask, (x, y), 0, *tolerances, flags
        )
        filled = cv2.compare(interior, 255, cv2.CMP_EQ)
        return Result({"mask": filled, "area": area, "bounds": list(bounds)})


class Watershed(Block):
    type_id, title = "inspect.watershed", "Marker-controlled watershed"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (Port("image"), Port("markers"))
    outputs = (Port("labels"), Port("boundaries"), Port("mask"), Port("count", "number"))
    defaults = {"marker_mode": "binary_components", "connectivity": "8", "output_label": 0}
    parameter_choices = {"marker_mode": ("binary_components", "labels"), "connectivity": ("4", "8")}
    parameter_help = {
        "marker_mode": "Label connected uint8 seeds or use an int32 map with 0 for unknown.",
        "connectivity": "Connectivity used to label binary seeds.",
        "output_label": "Label to output as a mask. Zero includes every positive label.",
    }

    def process(self, inputs):
        image = bgr8(inputs["image"])
        markers = inputs["markers"]
        require_same_frame((inputs["image"], markers))
        if markers.shape != image.shape[:2]:
            raise ValueError("Watershed markers must be monochrome and match the image size")
        if self.parameters["marker_mode"] == "binary_components":
            if markers.dtype != np.uint8:
                raise ValueError("Binary watershed markers must be uint8")
            _, labels = cv2.connectedComponentsWithAlgorithm(
                cv2.compare(markers, 0, cv2.CMP_NE),
                int(self.parameters["connectivity"]),
                cv2.CV_32S,
                cv2.CCL_SAUF,
            )
        else:
            if markers.dtype != np.int32 or (markers < 0).any():
                raise ValueError("Watershed label markers must be nonnegative int32")
            labels = np.array(markers, copy=True, order="C")
        if not np.any(labels > 0):
            raise ValueError("Watershed requires at least one positive seed")
        labels = cv2.watershed(image, labels)
        boundaries = cv2.compare(labels, -1, cv2.CMP_EQ)
        count = int(np.count_nonzero(np.unique(labels) > 0))
        output_label = integer(self.parameters["output_label"], "Output label", 0, 2**31 - 1)
        mask = cv2.compare(labels, output_label, cv2.CMP_EQ if output_label else cv2.CMP_GT)
        return Result({"labels": labels, "boundaries": boundaries, "mask": mask, "count": count})


SEGMENTATION_BLOCKS = (FloodFill, Watershed)
