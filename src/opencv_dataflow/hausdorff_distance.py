"""Compare sampled contours with an order-independent ranked Hausdorff distance."""

import numpy as np
from scipy.spatial import cKDTree

from .contour_analysis import read_ordered_points
from .core import Block, Port, Result
from .parameter_validation import NumericRange

METRICS = {"L2": 2, "L1": 1}


def ranked_nearest_distance(source, target, order, rank):
    distances, _ = cKDTree(target).query(source, p=order)
    index = int(rank * (len(source) - 1))
    return float(np.partition(distances, index)[index])


class HausdorffContourDistance(Block):
    type_id, title = "inspect.hausdorff_distance", "Compare contours (Hausdorff)"
    category = block_type = "Inspection"
    subtype = "Contour matching"
    description = "Measure symmetric ranked nearest-point distances between contours."
    inputs = (Port("reference", "data"), Port("sample", "data"))
    outputs = (
        Port("distance", "number"),
        Port("reference_to_sample", "number"),
        Port("sample_to_reference", "number"),
    )
    payload_reference = "sample"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "reference_index": 0,
        "sample_index": 0,
        "metric": "L2",
        "rank_proportion": 1.0,
        "check_distance": False,
        "maximum_distance": 1.0,
    }
    parameter_choices = {"metric": tuple(METRICS)}
    parameter_ranges = {
        "reference_index": NumericRange(0, whole=True),
        "sample_index": NumericRange(0, whole=True),
        "rank_proportion": NumericRange(0, 1),
        "maximum_distance": NumericRange(0),
    }
    parameter_help = {
        "reference_index": "Zero-based contour collection item. Ignored for a point list.",
        "sample_index": "Zero-based contour collection item. Ignored for a point list.",
        "metric": "L2 is Euclidean distance. L1 sums the absolute X and Y displacements.",
        "rank_proportion": "Select floor(rank × (count − 1)) in each ascending distance list. "
        "1 uses the farthest nearest neighbor. Lower ranks ignore larger distances.",
        "check_distance": "Fail when the symmetric distance exceeds the maximum.",
        "maximum_distance": "Inclusive distance limit in the supplied coordinate units.",
    }
    input_help = {
        "reference": "Contour collection or [x,y] point list in the comparison grid.",
        "sample": "Contour collection or [x,y] point list in the reference grid.",
    }
    output_help = {
        "distance": "Larger of the two directional distances, in the input coordinate units.",
        "reference_to_sample": "Ranked nearest-point distance from reference to sample.",
        "sample_to_reference": "Ranked nearest-point distance from sample to reference.",
    }

    def process(self, inputs):
        p = self.parameters
        reference, _ = read_ordered_points(
            self, inputs, "reference", p["reference_index"], minimum=1
        )
        sample, _ = read_ordered_points(self, inputs, "sample", p["sample_index"], minimum=1)
        reference = np.unique(reference, axis=0)
        sample = np.unique(sample, axis=0)
        metric = METRICS[p["metric"]]
        rank = float(p["rank_proportion"])
        # OpenCV 5.0.0 sorts the column of distances one row at a time. Selecting
        # the rank directly keeps the distance independent of contour point order.
        forward = ranked_nearest_distance(reference, sample, metric, rank)
        reverse = ranked_nearest_distance(sample, reference, metric, rank)
        distance = max(forward, reverse)
        limit = float(p["maximum_distance"])
        passed = not p["check_distance"] or distance <= limit
        return Result(
            {"distance": distance, "reference_to_sample": forward, "sample_to_reference": reverse},
            passed=passed,
            message=f"Hausdorff distance {distance:.6g}",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "contour_distance_out_of_tolerance",
                    "message": f"Hausdorff distance {distance:.6g} exceeds {limit:.6g}",
                }
            ],
        )
