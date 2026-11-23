"""Geometry-only OpenCV shape-context distance for ordered closed contours."""

import cv2
import numpy as np

from .contour_analysis import read_ordered_points
from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError

HISTOGRAM_COSTS = {
    "chi-square": "createChiHistogramCostExtractor",
    "norm": "createNormHistogramCostExtractor",
    "earth mover": "createEMDHistogramCostExtractor",
    "earth mover L1": "createEMDL1HistogramCostExtractor",
}
NORMS = {"L1": cv2.DIST_L1, "L2": cv2.DIST_L2}
ALIGNMENTS = ("thin-plate spline", "full affine", "partial affine")


def sample_closed_contour(points, count):
    # Center before conversion to float32 to retain precision for offset contours.
    centered = points - points[0]
    area = cv2.contourArea(centered.astype(np.float32), oriented=True)
    if area == 0:
        raise ValidationError("Shape context requires a contour enclosing a nonzero area")
    if area < 0:
        centered = centered[::-1]
    start = np.lexsort((centered[:, 1], centered[:, 0]))[0]
    ordered = np.roll(centered, -start, axis=0) - centered[start]
    edges = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    keep = edges > 0
    vertices = ordered[keep]
    arc = np.concatenate(([0.0], edges[keep].cumsum()))
    closed = np.vstack((vertices, vertices[0]))
    positions = np.linspace(0.0, arc[-1], count, endpoint=False)
    sampled = np.column_stack([np.interp(positions, arc, closed[:, axis]) for axis in range(2)])
    native = (sampled - sampled.mean(axis=0)).astype(np.float32).reshape(1, -1, 2)
    if cv2.contourArea(native) == 0:
        raise ValidationError("Resampled contour has zero area. Increase the sample count.")
    return native


class ShapeContextDistance(Block):
    type_id, title = "inspect.shape_context_distance", "Compare contours (shape context)"
    category = block_type = "Inspection"
    subtype = "Contour matching"
    description = "Compare closed silhouettes after iterative alignment using OpenCV shape context."
    inputs = (Port("reference", "data"), Port("sample", "data"))
    outputs = (Port("distance", "number"),)
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "reference_index": 0,
        "sample_index": 0,
        "samples": 64,
        "angular_bins": 12,
        "radial_bins": 4,
        "inner_radius": 0.2,
        "outer_radius": 2.0,
        "iterations": 3,
        "rotation_invariant": False,
        "histogram_cost": "chi-square",
        "histogram_norm": "L2",
        "dummy_count": 12,
        "dummy_cost": 0.2,
        "alignment": "thin-plate spline",
        "shape_weight": 1.0,
        "bending_weight": 0.3,
        "check_distance": False,
        "maximum_distance": 0.05,
    }
    parameter_choices = {
        "histogram_cost": tuple(HISTOGRAM_COSTS),
        "histogram_norm": tuple(NORMS),
        "alignment": ALIGNMENTS,
    }
    parameter_ranges = {
        "reference_index": NumericRange(0, whole=True),
        "sample_index": NumericRange(0, whole=True),
        "samples": NumericRange(16, 128, whole=True),
        "angular_bins": NumericRange(4, 24, whole=True),
        "radial_bins": NumericRange(2, 8, whole=True),
        "inner_radius": NumericRange(0.01, 2),
        "outer_radius": NumericRange(0.1, 8),
        "iterations": NumericRange(1, 6, whole=True),
        "dummy_count": NumericRange(1, 32, whole=True),
        "dummy_cost": NumericRange(0.01, 10),
        "shape_weight": NumericRange(0, 10),
        "bending_weight": NumericRange(0, 10),
        "maximum_distance": NumericRange(0),
    }
    parameter_help = {
        "reference_index": "Zero-based collection item. Ignored for a point list.",
        "sample_index": "Zero-based collection item. Ignored for a point list.",
        "samples": "Number of equally spaced perimeter samples on each closed contour.",
        "angular_bins": "Angular sectors in the log-polar shape descriptor.",
        "radial_bins": "Logarithmic radius bins in the shape descriptor.",
        "inner_radius": "Smallest descriptor radius in mean point-separation units.",
        "outer_radius": "Largest descriptor radius in mean point-separation units.",
        "iterations": "Number of rounds of descriptor matching and alignment.",
        "rotation_invariant": "Orient each descriptor relative to the contour centroid.",
        "histogram_cost": "Distance used to compare each pair of local shape histograms.",
        "histogram_norm": "Ground distance for the norm and earth mover histogram costs.",
        "dummy_count": "Dummy assignments available for unmatched contour samples.",
        "dummy_cost": "Cost of assigning a point to a dummy instead of a real match.",
        "alignment": "Transform applied between rounds of correspondence matching.",
        "shape_weight": "Weight of the final histogram matching cost.",
        "bending_weight": "Weight of the accumulated deformation cost for thin-plate spline.",
        "check_distance": "Fail an inspection if the measured distance exceeds the maximum.",
        "maximum_distance": "Inclusive limit for the chosen settings. Smaller means closer.",
    }
    input_help = {
        "reference": "Closed reference silhouette as ordered [x,y] points or a contour collection.",
        "sample": "Closed sample silhouette as ordered [x,y] points or a contour collection.",
    }
    output_help = {
        "distance": "Raw reference-to-sample OpenCV distance. Lower values indicate closer shapes."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["inner_radius"]) >= float(p["outer_radius"]):
            raise ValidationError(
                "Inner radius must be smaller than outer radius", "inner_radius", "outer_radius"
            )
        if float(p["shape_weight"]) == 0 and (
            p["alignment"] != ALIGNMENTS[0] or float(p["bending_weight"]) == 0
        ):
            raise ValidationError(
                "Set a positive shape weight or use a weighted spline",
                "shape_weight",
                "bending_weight",
                "alignment",
            )
        if p["dummy_count"] > p["samples"] - 3:
            raise ValidationError("Leave at least three real assignments", "dummy_count", "samples")
        work = p["samples"] ** 2 * p["angular_bins"] * p["radial_bins"] * p["iterations"]
        if p["histogram_cost"].startswith("earth mover") and work > 1_500_000:
            raise ValidationError(
                "Earth mover cost exceeds the work limit. Reduce samples, bins, or iterations.",
                "samples",
                "angular_bins",
                "radial_bins",
                "iterations",
            )

    def process(self, inputs):
        p = self.parameters
        cost_name = p["histogram_cost"]
        spline = p["alignment"] == ALIGNMENTS[0]
        create = getattr(cv2, "createShapeContextDistanceExtractor", None)
        cost_factory = getattr(cv2, HISTOGRAM_COSTS[cost_name], None)
        transform_name = (
            "createThinPlateSplineShapeTransformer" if spline else "createAffineTransformer"
        )
        transform_factory = getattr(cv2, transform_name, None)
        if not all((create, cost_factory, transform_factory)):
            raise ValidationError("Shape context requires the OpenCV contrib shape module")
        shape = {}
        for name in ("reference", "sample"):
            points, _ = read_ordered_points(self, inputs, name, p[f"{name}_index"])
            shape[name] = sample_closed_contour(points, int(p["samples"]))
        cost_args = (int(p["dummy_count"]), float(p["dummy_cost"]))
        if cost_name in {"norm", "earth mover"}:
            cost_args = (NORMS[p["histogram_norm"]], *cost_args)
        transformer = (
            transform_factory() if spline else transform_factory(p["alignment"] == "full affine")
        )
        distance_extractor = create(
            int(p["angular_bins"]),
            int(p["radial_bins"]),
            float(p["inner_radius"]),
            float(p["outer_radius"]),
            int(p["iterations"]),
            cost_factory(*cost_args),
            transformer,
        )
        distance_extractor.setRotationInvariant(p["rotation_invariant"])
        distance_extractor.setShapeContextWeight(float(p["shape_weight"]))
        distance_extractor.setBendingEnergyWeight(float(p["bending_weight"]) if spline else 0.0)
        distance = distance_extractor.computeDistance(shape["reference"], shape["sample"])
        if not np.isfinite(distance):
            raise ValidationError("OpenCV returned a nonfinite shape-context distance")
        limit = float(p["maximum_distance"])
        passed = not p["check_distance"] or distance <= limit
        return Result(
            {"distance": distance},
            passed=passed,
            message=f"Shape-context distance {distance:.6g}",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "contour_distance_out_of_tolerance",
                    "message": f"Shape-context distance {distance:.6g} exceeds {limit:.6g}",
                }
            ],
        )
