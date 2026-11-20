"""Deterministic random and exact greedy farthest-point cloud sampling."""

import math

import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics
from .parameter_validation import NumericRange

_BATCH = 32_768


def farthest_sample(points, limit, rng, minimum_spacing, work_limit, cancelled):
    """Maintain nearest-selected distance with bounded float64 working storage."""
    size = len(points)
    chosen = np.empty(limit, np.int32)
    nearest = np.full(size, np.inf)
    scratch_rows = min(_BATCH, size)
    delta = np.empty((scratch_rows, 3), np.float64)
    squares = np.empty(scratch_rows, np.float64)
    threshold = minimum_spacing**2
    pivot = int(rng.integers(size))
    completed = 0
    spacing = None
    for index in range(limit):
        if cancelled():
            stop = "cancelled"
            break
        if index and nearest[pivot] < threshold:
            stop = "minimum_spacing"
            break
        if (index + 1) * size > work_limit:
            stop = "work_limit"
            break
        if index:
            spacing = math.sqrt(float(nearest[pivot]))
        chosen[index] = pivot
        center = points[pivot]
        for start in range(0, size, _BATCH):
            batch = slice(start, min(start + _BATCH, size))
            n = batch.stop - batch.start
            work = delta[:n]
            np.subtract(points[batch], center, dtype=np.float64, out=work)
            np.einsum("ij,ij->i", work, work, out=squares[:n])
            np.minimum(nearest[batch], squares[:n], out=nearest[batch])
        nearest[pivot] = -1
        completed += 1
        pivot = int(np.argmax(nearest))
    else:
        stop = "point_budget"
    coverage = math.sqrt(max(0.0, float(nearest[pivot]))) if completed else None
    return chosen[:completed], spacing, coverage, completed * size, stop


class SampleCloud(Tool3D):
    type_id = "3d.cloud_sample"
    title, subtype = "Sample point cloud", "Point clouds"
    description = "Select original cloud rows by seeded random or greedy farthest-point sampling."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("cloud", "cloud"),
        Port("count", "integer"),
        Port("indices", "data"),
        Port("report", "data"),
    )
    defaults = {
        "method": "random",
        "budget": "count",
        "max_points": 1000,
        "sample_fraction": 0.1,
        "seed": 0,
        "minimum_spacing": 0.0,
        "max_distance_evaluations": 50_000_000,
    }
    parameter_choices = {"method": ("random", "farthest"), "budget": ("count", "fraction")}
    parameter_ranges = {
        "max_points": NumericRange(1, 100_000, whole=True),
        "sample_fraction": NumericRange(0.000001, 1),
        "seed": NumericRange(0, 2**32 - 1, whole=True),
        "minimum_spacing": NumericRange(0, 1e39),
        "max_distance_evaluations": NumericRange(1, 250_000_000, whole=True),
    }
    parameter_help = {
        "method": "Random draws distinct rows. Farthest selects the most distant remaining row.",
        "budget": "Retain up to Max points, or use a capped fraction of the input.",
        "max_points": "Hard limit on output rows in either budget mode.",
        "sample_fraction": "Target fraction rounded up to a whole point, capped by Max points.",
        "seed": "Independent random seed for sampling or for the first farthest point.",
        "minimum_spacing": "Farthest mode stops below this distance in the cloud's units.",
        "max_distance_evaluations": "Farthest mode reports failure if it hits this work limit.",
    }
    input_help = {
        "cloud": "Original XYZ rows, optionally with measured colors, normals and source pixels."
    }

    def process(self, inputs):
        source, p = inputs["cloud"], self.parameters
        size = len(source.points)
        target = min(int(p["max_points"]), size)
        if p["budget"] == "fraction":
            target = min(target, math.ceil(size * float(p["sample_fraction"])))
        seed = int(p["seed"])
        spacing, coverage, evaluations = None, None, 0
        rng = np.random.Generator(np.random.PCG64(seed))
        if self._cancelled():
            indices, stop = np.empty(0, np.int32), "cancelled"
        elif not size:
            indices, stop = np.empty(0, np.int32), "empty_cloud"
        elif p["method"] == "random":
            indices = rng.choice(size, target, replace=False).astype(np.int32, copy=False)
            stop = "point_budget"
        else:
            indices, spacing, coverage, evaluations, stop = farthest_sample(
                source.points,
                target,
                rng,
                float(p["minimum_spacing"]),
                int(p["max_distance_evaluations"]),
                self._cancelled,
            )

        count = len(indices)
        cloud = source.subset(indices)
        passed = bool(count) and stop in ("point_budget", "minimum_spacing")
        report = {
            "method": p["method"],
            "seed": seed,
            "input_points": size,
            "target_points": target,
            "selected_points": count,
            "stop_reason": stop,
            "spacing_units": source.units,
            "minimum_selected_spacing": spacing,
            "covering_radius": coverage,
            "distance_evaluations": evaluations,
        }
        return Result(
            {"cloud": cloud, "count": count, "indices": indices.tolist(), "report": report},
            graphics=cloud_graphics(self, cloud),
            payloads={"cloud": self.input_payloads["cloud"]},
            passed=passed,
            fail_reasons=[] if passed else [f"cloud_sampling_{stop}"],
            message=f"Selected {count} of {target} target points · {stop.replace('_', ' ')}",
        )
