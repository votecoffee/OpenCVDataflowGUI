"""Exact JSON binary regions, encoded as bounded sorted scanline runs."""

import numpy as np

from .parameter_validation import ValidationError

MAX_REGION_RUNS = 1_000_000


def region_from_mask(mask, maximum_runs=100000):
    runs = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(row != 0, (1, 1))))
        if len(runs) + len(edges) // 2 > maximum_runs:
            raise ValidationError(
                "Remapped region exceeds Maximum region runs", "maximum_region_runs"
            )
        runs.extend([y, int(a), int(b - a)] for a, b in zip(edges[::2], edges[1::2], strict=True))
    return {
        "model": "image_region_v1",
        "shape": "raster",
        "encoding": "scanline_runs_v1",
        "image_size": [mask.shape[1], mask.shape[0]],
        "runs": runs,
        "invert": False,
    }


def raster_mask(region, shape, run_budget):
    runs = region.get("runs")
    if (
        region.get("encoding") != "scanline_runs_v1"
        or not isinstance(runs, list)
        or len(runs) > run_budget[0]
        or type(region.get("invert", False)) is not bool
    ):
        raise ValidationError("Invalid raster region or scanline-run budget exceeded")
    run_budget[0] -= len(runs)
    height, width = shape[:2]
    result = np.zeros((height, width), np.uint8)
    previous_y, previous_end = -1, 0
    for run in runs:
        if not isinstance(run, list) or len(run) != 3 or any(type(n) is not int for n in run):
            raise ValidationError("Raster runs must contain integer [row,start,length]")
        y, x, length = run
        if (
            not 0 <= y < height
            or not 0 <= x < width
            or length <= 0
            or x + length > width
            or y < previous_y
            or (y == previous_y and x < previous_end)
        ):
            raise ValidationError("Raster runs must be sorted, disjoint and inside the image")
        result[y, x : x + length] = 255
        previous_y, previous_end = y, x + length
    return 255 - result if region.get("invert", False) else result
