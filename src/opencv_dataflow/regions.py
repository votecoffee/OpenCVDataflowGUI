"""Reusable region creation and point-result filtering."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import preview_rgb
from .selection import contains, rasterize, region_matrix, region_size, selection_mask


def region_result(image, region):
    mask = rasterize(region, image.shape)
    overlay = cv2.cvtColor(preview_rgb(image), cv2.COLOR_RGB2BGR)
    tint = np.full_like(overlay, (50, 220, 100))
    overlay[mask != 0] = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)[mask != 0]
    return Result(
        {"mask": mask, "region": region, "overlay": overlay},
        graphics={"region": overlay},
        message=f"{np.count_nonzero(mask)} selected pixels",
    )


class ShapeRegion(Block):
    supports_region_editor = True
    type_id, title, category = "region.shape", "Shape region", "Image processing"
    block_type, subtype = "Image processing", "Regions"
    inputs = (Port("image"),)
    outputs = (Port("mask"), Port("region", "data"), Port("overlay"))
    defaults = {
        "shape": "rectangle",
        "rectangle": [0, 0, 100, 100],
        "center": [100, 100],
        "radius": 50,
        "vertices": [[0, 0], [100, 0], [50, 100]],
        "invert": False,
    }
    parameter_choices = {"shape": ("rectangle", "circle", "polygon")}
    parameter_help = {
        "rectangle": "[x,y,width,height], whole input-image pixels.",
        "vertices": "Polygon [[x,y], ...] in input-image pixels; boundary is included.",
        "invert": "Select everything outside the shape.",
    }

    def process(self, inputs):
        image = inputs["image"]
        region = {
            "model": "image_region_v1",
            "image_size": [image.shape[1], image.shape[0]],
            **self.parameters,
        }
        return region_result(image, region)


class CombineRegions(Block):
    type_id, title, category = "region.combine", "Combine regions", "Image processing"
    block_type, subtype = "Image processing", "Regions"
    inputs = (Port("image"), Port("region_a", "data"), Port("region_b", "data", required=False))
    outputs = ShapeRegion.outputs
    defaults = {"operation": "union"}
    parameter_choices = {"operation": ("union", "intersection", "difference", "xor", "invert")}
    parameter_help = {
        "operation": "Difference selects A outside B. Invert uses A only; otherwise B is required."
    }

    def process(self, inputs):
        operation = self.parameters["operation"]
        children = [copy.deepcopy(inputs["region_a"])]
        if operation != "invert":
            if "region_b" not in inputs:
                raise ValueError("This operation requires region_b")
            children.append(copy.deepcopy(inputs["region_b"]))
        image = inputs["image"]
        region = {
            "model": "image_region_v1",
            "image_size": [image.shape[1], image.shape[0]],
            "shape": "composite",
            "operation": operation,
            "children": children,
        }
        return region_result(image, region)


class TransformRegion(Block):
    type_id, title, category = "region.transform", "Transform region", "Image processing"
    block_type, subtype = "Image processing", "Regions"
    inputs = (Port("image"), Port("region", "data"), Port("transform", "data", required=False))
    outputs = (*ShapeRegion.outputs, Port("transform", "data"))
    defaults = {
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "direction": "forward",
        "interpolation": "nearest",
        "threshold": 0.5,
    }
    parameter_choices = {
        "direction": ("forward", "inverse"),
        "interpolation": ("nearest", "linear"),
    }
    parameter_help = {
        "matrix": "Source-to-target 2x3/3x3 matrix, used when the transform port is unconnected.",
        "direction": "Forward follows image geometry or source_to_target; inverse goes back.",
        "interpolation": "Nearest preserves discrete masks; linear thresholds resampled coverage.",
        "threshold": "Linear sampling cutoff in (0,1]; nearest is binary at any valid cutoff.",
    }

    def process(self, inputs):
        image, source = inputs["image"], inputs["region"]
        source_size = list(region_size(source))
        target_size = [image.shape[1], image.shape[0]]
        inverse = self.parameters["direction"] == "inverse"
        payload = inputs.get("transform", self.parameters["matrix"])
        expected_source = expected_target = None
        if isinstance(payload, dict):
            if "output_to_input" in payload and "source_to_target" in payload:
                raise ValueError("Transform has ambiguous mapping keys")
            if "output_to_input" in payload:
                matrix = region_matrix(payload["output_to_input"])
                if not inverse:
                    matrix = np.linalg.inv(matrix)
                expected_source = payload.get("output_size" if inverse else "input_size")
                expected_target = payload.get("input_size" if inverse else "output_size")
                if expected_source is None or expected_target is None:
                    raise ValueError("Image geometry must include input_size and output_size")
            elif "source_to_target" in payload:
                matrix = region_matrix(payload["source_to_target"])
                if inverse:
                    matrix = np.linalg.inv(matrix)
                expected_source = payload.get("target_size" if inverse else "source_size")
                expected_target = payload.get("source_size" if inverse else "target_size")
            else:
                raise ValueError(
                    "Transform needs source_to_target or image geometry output_to_input"
                )
        else:
            matrix = region_matrix(payload)
            if inverse:
                matrix = np.linalg.inv(matrix)
        if (expected_source is not None and expected_source != source_size) or (
            expected_target is not None and expected_target != target_size
        ):
            raise ValueError("Transform dimensions do not match the source region and target image")
        region = {
            "model": "image_region_v1",
            "image_size": target_size,
            "shape": "transformed",
            "source": copy.deepcopy(source),
            "source_to_target": matrix.tolist(),
            "interpolation": self.parameters["interpolation"],
            "threshold": self.parameters["threshold"],
        }
        result = region_result(image, region)
        result.outputs["transform"] = {
            "model": "image_transform_v1",
            "source_to_target": matrix.tolist(),
            "source_size": source_size,
            "target_size": target_size,
        }
        return result


class MeasureRegion(Block):
    type_id, title, category = "region.measure", "Measure selected region", "Inspection"
    block_type, subtype = "Inspection", "Regions"
    inputs = (Port("image"), Port("mask", required=False), Port("region", "data", required=False))
    outputs = (
        Port("area", "number"),
        Port("coverage", "number"),
        Port("centroid", "data"),
        Port("bounds", "data"),
        Port("data", "data"),
    )
    defaults = {"min_area": 0, "max_area": 0, "min_coverage": 0.0, "max_coverage": 1.0}
    parameter_help = {
        "max_area": "Maximum selected pixel count; zero means unlimited.",
        "min_coverage": "Selected pixels / image pixels, from zero to one.",
    }

    def process(self, inputs):
        from .processing import finite, integer

        image = inputs["image"]
        mask = selection_mask(inputs, image)
        if mask is None:
            mask = np.full(image.shape[:2], 255, np.uint8)
        area = int(np.count_nonzero(mask))
        coverage = area / mask.size
        bounds, centroid = [], []
        if area:
            x, y, width, height = cv2.boundingRect(mask)
            bounds = [x, y, width, height]
            moments = cv2.moments(mask, binaryImage=True)
            centroid = [[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]
        p = self.parameters
        minimum = integer(p["min_area"], "Minimum area", 0, 64_000_000)
        maximum = integer(p["max_area"], "Maximum area", 0, 64_000_000)
        low, high = (finite(p[k], k, 0) for k in ("min_coverage", "max_coverage"))
        if (maximum and maximum < minimum) or not low <= high <= 1:
            raise ValueError(
                "Region area and coverage limits must be ordered; coverage is in [0,1]"
            )
        data = {
            "area_pixels": area,
            "coverage": coverage,
            "bounds": bounds,
            "centroid": centroid,
            "units": "pixels",
            "image_size": [image.shape[1], image.shape[0]],
        }
        return Result(
            {
                "area": area,
                "coverage": coverage,
                "centroid": centroid,
                "bounds": bounds,
                "data": data,
            },
            passed=area >= minimum and (not maximum or area <= maximum) and low <= coverage <= high,
            message=f"{area} pixels selected ({coverage:.1%} coverage)",
        )


class FilterPoints(Block):
    type_id, title, category = "region.filter_points", "Filter points by region", "Inspection"
    block_type, subtype = "Inspection", "Regions"
    inputs = (
        Port("image"),
        Port("points", "data"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("points", "data"), Port("count", "number"))

    def process(self, inputs):
        mask = selection_mask(inputs, inputs["image"])
        points = np.asarray(inputs["points"], float)
        if not points.size:
            return Result({"points": [], "count": 0})
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Points must be a finite Nx2 list")
        selected = [p.tolist() for p in points if contains(mask, p)]
        return Result({"points": selected, "count": len(selected)})


REGION_BLOCKS = (ShapeRegion, CombineRegions, TransformRegion, MeasureRegion, FilterPoints)
