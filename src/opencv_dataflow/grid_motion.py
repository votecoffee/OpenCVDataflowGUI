"""Filter paired image matches by OpenCV's grid-based motion statistics."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .match_review import paired_points
from .parameter_validation import NumericRange, ValidationError

MAX_MATCHES = 10_000
_PAIRED_FIELDS = (
    "query_points",
    "train_points",
    "matches",
    "query_segments",
    "train_segments",
    "scores",
    "inliers",
)


class GridMotionFilter(Block):
    type_id, title = "inspect.filter_matches_gms", "Grid motion match filter"
    category = block_type = "Inspection"
    subtype = "Matching"
    description = "Retain feature pairs with consistent local image motion before geometry fitting."
    inputs = (Port("matches", "data"),)
    outputs = (
        Port("matches", "data"),
        Port("rejected", "data"),
        Port("kept_mask", "data"),
        Port("count", "number"),
        Port("rejected_count", "number"),
    )
    payload_reference = "matches"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "with_rotation": False,
        "with_scale": False,
        "threshold_factor": 6.0,
        "min_matches": 0,
    }
    parameter_ranges = {
        "threshold_factor": NumericRange(0.1, 100),
        "min_matches": NumericRange(0, MAX_MATCHES, whole=True),
    }
    parameter_help = {
        "with_rotation": "Test additional grid neighborhood orientations.",
        "with_scale": "Test additional grid size ratios.",
        "threshold_factor": "Require more neighboring support as this factor increases.",
        "min_matches": "Fewest retained pairs required to pass. Zero allows an empty result.",
    }
    input_help = {
        "matches": "Paired feature matches with both image sizes in pixels and one candidate "
        "per query feature. Dense ORB nearest matches are a good starting point.",
    }
    output_help = {
        "matches": "Accepted pairs in incoming order, with their original distances and frames.",
        "rejected": "Rejected pairs in the same schema as the input.",
        "kept_mask": "One boolean per input pair, in incoming order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("with_rotation", "with_scale"):
            if type(p[name]) is not bool:
                raise ValidationError(
                    f"{name.replace('_', ' ').capitalize()} must be on or off", name
                )

    def process(self, inputs):
        data = inputs["matches"]
        if not isinstance(data, dict):
            raise ValidationError("Connect a paired feature match record")
        sizes = [data.get(f"{name}_size") for name in ("query", "train")]
        if any(
            not isinstance(size, list)
            or len(size) != 2
            or any(type(axis) is not int or not 1 <= axis <= 16384 for axis in size)
            for size in sizes
        ):
            raise ValidationError("GMS needs both image sizes, up to 16,384 pixels per axis")
        fields = {name: data[name] for name in _PAIRED_FIELDS if name in data}
        base = data.get("query_points")
        if not isinstance(base, (list, tuple)) or len(base) > MAX_MATCHES:
            raise ValidationError("GMS accepts up to 10,000 paired points")
        count = len(base)
        if any(
            not isinstance(values, (list, tuple)) or len(values) != count
            for values in fields.values()
        ):
            raise ValidationError("Paired match fields must have one entry per match")
        if "matches" in fields:
            rows = fields["matches"]
            if any(not isinstance(row, dict) for row in rows):
                raise ValidationError("Match metadata rows must be objects")
            ids = [row["query_index"] for row in rows if "query_index" in row]
            if any(type(index) is not int or index < 0 for index in ids) or len(set(ids)) != len(
                ids
            ):
                raise ValidationError("GMS requires one candidate match per query feature")
        points = paired_points(data, *sizes)
        points = [np.asarray(array, np.float32) for array in points]
        if any(np.any(array >= size) for array, size in zip(points, sizes, strict=True)):
            raise ValidationError(
                "Match coordinates round outside their image at float32 precision"
            )

        p = self.parameters
        native = getattr(getattr(cv2, "xfeatures2d", None), "matchGMS", None)
        if native is None:
            raise ValidationError("Grid motion filtering requires OpenCV contrib xfeatures2d")
        mask = [False] * count
        if count:
            query, train = [[cv2.KeyPoint(float(x), float(y), 1.0) for x, y in a] for a in points]
            # GMS uses only positions and indices. Pair-local IDs preserve the caller's rows.
            pairs = [cv2.DMatch(index, index, 0.0) for index in range(count)]
            kept = native(
                *sizes,
                query,
                train,
                pairs,
                withRotation=p["with_rotation"],
                withScale=p["with_scale"],
                thresholdFactor=float(p["threshold_factor"]),
            )
            for match in kept:
                mask[match.queryIdx] = True

        accepted = sum(mask)
        passed = accepted >= int(p["min_matches"])
        return Result(
            {
                "matches": {
                    **data,
                    **{
                        name: [row for row, keep in zip(values, mask, strict=True) if keep]
                        for name, values in fields.items()
                    },
                },
                "rejected": {
                    **data,
                    **{
                        name: [row for row, keep in zip(values, mask, strict=True) if not keep]
                        for name, values in fields.items()
                    },
                },
                "kept_mask": mask,
                "count": accepted,
                "rejected_count": count - accepted,
            },
            passed=passed,
            fail_reasons=[] if passed else ["too_few_grid_motion_matches"],
            message=f"{accepted}/{count} pairs passed grid motion filtering",
        )
