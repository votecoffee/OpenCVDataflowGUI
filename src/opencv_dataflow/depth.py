"""Typed physical depth and paired color/depth images; no implied registration."""

from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import DepthArray, preview_rgb
from .parameter_validation import NumericRange
from .selection import selection_mask


class DepthFile(Block):
    type_id, title, category = "input.depth_file", "Depth image file", "Input"
    block_type, subtype = "Input", "Depth and RGB-D"
    outputs = (Port("depth", "depth"), Port("preview"), Port("info", "data"))
    defaults = {
        "path": "",
        "units_per_sample": 0.001,
        "units": "m",
        "display_min": 0.0,
        "display_max": 5.0,
    }
    parameter_choices = {"units": ("m", "mm")}

    def process(self, inputs):
        p = self.parameters
        scale, low, high = (float(p[k]) for k in ("units_per_sample", "display_min", "display_max"))
        if not np.isfinite([scale, low, high]).all() or scale <= 0 or high <= low:
            raise ValueError("Depth scale must be positive and display max greater than min")
        if p["units"] not in ("m", "mm"):
            raise ValueError("Depth units must be m or mm")
        image = cv2.imdecode(
            np.frombuffer(Path(p["path"]).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        if (
            image is None
            or image.ndim != 2
            or image.dtype not in (np.uint16, np.float32, np.float64)
        ):
            raise ValueError(
                "Depth file must decode as single-channel uint16 or floating-point pixels"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            values = image.astype(np.float32) * scale
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        depth = DepthArray(values, p["units"], low, high)
        valid = np.isfinite(depth)
        graphic = cv2.cvtColor(preview_rgb(depth), cv2.COLOR_RGB2BGR)
        return Result(
            {
                "depth": depth,
                "preview": graphic,
                "info": {
                    "units": p["units"],
                    "units_per_sample": scale,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "valid_samples": int(valid.sum()),
                    "invalid_samples": int((~valid).sum()),
                },
            },
            graphics={"depth": graphic},
            passed=bool(valid.any()),
            score=float(valid.mean()),
            message=f"Physical depth in {p['units']}",
        )


class RGBD(Block):
    type_id, title, category = "image.rgbd", "Pair color and depth", "Image processing"
    block_type, subtype = "Image processing", "Depth and RGB-D"
    inputs = (Port("image"), Port("depth", "depth"))
    outputs = (Port("image"), Port("depth", "depth"), Port("info", "data"))

    def process(self, inputs):
        image, depth = inputs["image"], inputs["depth"]
        if image.shape[:2] != depth.shape:
            raise ValueError("Paired color and depth must have identical dimensions")
        return Result(
            {
                "image": image,
                "depth": depth,
                "info": {
                    "units": getattr(depth, "units", "unknown"),
                    "registration": "caller-supplied; dimensions checked only",
                },
            },
            message="Paired planes; no geometric registration performed",
        )


class DepthSample(Block):
    type_id, title, category = "inspect.depth_sample", "Sample depth", "Inspection"
    block_type, subtype = "Inspection", "Depth"
    inputs = (Port("depth", "depth"), Port("points", "data", required=False))
    outputs = (Port("samples", "data"),)
    defaults = {"pixel_points": [[0, 0]]}

    def process(self, inputs):
        depth = inputs["depth"]
        points = np.asarray(inputs.get("points", self.parameters["pixel_points"]), float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Depth sample points must be a finite list of [x, y] coordinates")
        if np.any(points != np.rint(points)):
            raise ValueError("Depth sample coordinates must be whole pixels")
        if (
            np.any(points < 0)
            or np.any(points[:, 0] >= depth.shape[1])
            or np.any(points[:, 1] >= depth.shape[0])
        ):
            raise ValueError("Depth sample point is outside the image")
        samples = []
        for x, y in points.astype(int):
            value = float(depth[y, x])
            samples.append(
                {
                    "pixel": [int(x), int(y)],
                    "depth": value if np.isfinite(value) and value > 0 else None,
                }
            )
        return Result(
            {"samples": {"units": getattr(depth, "units", "unknown"), "samples": samples}}
        )


class FillDepthHoles(Block):
    type_id, title = "3d.fill_depth_holes", "Fill small depth holes"
    category, block_type, subtype = "3D vision", "3D vision", "Depth"
    inputs = (
        Port("depth", "depth"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("depth", "depth"),
        Port("filled_mask", visual_type="region"),
        Port("remaining_mask", visual_type="region"),
        Port("filled_count", "number"),
    )
    defaults = {
        "maximum_hole_area": 64,
        "maximum_distance": 4,
        "hole_connectivity": "8",
        "distance_metric": "manhattan",
        "include_border": False,
    }
    parameter_choices = {
        "hole_connectivity": ("4", "8"),
        "distance_metric": ("manhattan", "chessboard"),
    }
    parameter_ranges = {
        "maximum_hole_area": NumericRange(1, 1_000_000, whole=True),
        "maximum_distance": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "maximum_hole_area": "Largest connected group of invalid samples eligible for filling.",
        "maximum_distance": "Maximum pixel distance to an original valid sample.",
        "include_border": "Allow filling gaps that touch an image edge.",
    }

    def process(self, inputs):
        source = inputs["depth"]
        values = np.asarray(source)
        valid = np.isfinite(values) & (values > 0)
        invalid = ~valid
        selected = selection_mask(inputs, source)
        output = source.copy()
        filled = np.zeros(valid.shape, bool)
        p = self.parameters
        targets = invalid if selected is None else invalid & (selected != 0)
        has_valid = bool(valid.any())
        if targets.any() and has_valid:
            holes = invalid.astype(np.uint8)
            _, components, stats, _ = cv2.connectedComponentsWithStats(
                holes, connectivity=int(p["hole_connectivity"])
            )
            eligible = stats[:, cv2.CC_STAT_AREA] <= int(p["maximum_hole_area"])
            eligible[0] = False
            if not p["include_border"]:
                border_ids = np.concatenate(
                    (components[0], components[-1], components[:, 0], components[:, -1])
                )
                eligible[border_ids] = False
            targets = targets & eligible[components]
            if targets.any():
                metric = cv2.DIST_L1 if p["distance_metric"] == "manhattan" else cv2.DIST_C
                distance, labels = cv2.distanceTransformWithLabels(
                    holes, metric, cv2.DIST_MASK_3, labelType=cv2.DIST_LABEL_PIXEL
                )
                filled = targets & (distance <= int(p["maximum_distance"]))
                donors = np.empty(int(labels.max()) + 1, values.dtype)
                donors[labels[valid]] = values[valid]
                output[filled] = donors[labels[filled]]
        count = int(np.count_nonzero(filled))
        remaining = invalid & ~filled
        missing = int(np.count_nonzero(remaining))
        return Result(
            {
                "depth": output,
                "filled_mask": filled.astype(np.uint8) * 255,
                "remaining_mask": remaining.astype(np.uint8) * 255,
                "filled_count": count,
            },
            passed=has_valid,
            fail_reasons=[] if has_valid else ["depth_has_no_valid_samples"],
            message=f"Filled {count} depth pixels · {missing} remain invalid",
        )
