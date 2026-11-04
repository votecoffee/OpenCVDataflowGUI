"""Reusable region creation and point-result filtering."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import preview_rgb
from .selection import contains, rasterize, selection_mask


class ShapeRegion(Block):
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
        mask = rasterize(region, image.shape)
        overlay = cv2.cvtColor(preview_rgb(image), cv2.COLOR_RGB2BGR)
        tint = np.full_like(overlay, (50, 220, 100))
        overlay[mask != 0] = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)[mask != 0]
        return Result(
            {"mask": mask, "region": region, "overlay": overlay},
            graphics={"region": overlay},
            message=f"{np.count_nonzero(mask)} selected pixels",
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


REGION_BLOCKS = (ShapeRegion, FilterPoints)
