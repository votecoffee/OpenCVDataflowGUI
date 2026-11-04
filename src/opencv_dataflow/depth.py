"""Typed physical depth and paired color/depth images; no implied registration."""

from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import DepthArray, preview_rgb


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
