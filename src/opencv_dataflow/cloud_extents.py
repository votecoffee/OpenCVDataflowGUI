"""Measure oriented cloud extents and qualify the associated rigid pose."""

from itertools import product

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, pose_matrix, pose_record, rigid
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import render_cloud

_CORNERS = np.array(list(product((0, 1), repeat=3)))
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]
_PROPER_SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
_RANK_TOLERANCE = 1e-8


def _reference_axes(value, cloud_frame):
    if (
        not isinstance(value, dict)
        or value.get("target_frame") != cloud_frame
        or not isinstance(value.get("source_frame"), str)
        or not value["source_frame"].strip()
        or value.get("units") not in ("m", "mm")
    ):
        raise ValidationError(
            "Reference pose must map a named frame into the metric cloud frame", "reference_pose"
        )
    try:
        rotation = rigid(value)[:3, :3]
    except ValueError as error:
        raise ValidationError(str(error), "reference_pose") from error
    # Normalize the small numerical drift accepted by the shared rigid pose contract.
    left, _, right = np.linalg.svd(rotation)
    return left @ right


def _pca_axes(points, reference, minimum_gap):
    mean, vectors, values = cv2.PCACompute2(points, mean=None)
    variances = np.maximum(values.ravel(), 0)
    rank = int(np.count_nonzero(variances > variances[0] * _RANK_TOLERANCE))
    ratios = np.divide(
        variances[:-1] - variances[1:],
        variances[:-1],
        out=np.zeros(2),
        where=variances[:-1] > 0,
    )
    basis = vectors[:2].copy()
    signs = basis[np.arange(2), np.argmax(np.abs(basis), axis=1)] < 0
    basis[signs] *= -1
    axes = np.column_stack([basis[0], basis[1], np.cross(*basis)])
    diagnostics = {
        "principal_variances": variances.tolist(),
        "eigen_gap_ratios": ratios.tolist(),
        "minimum_eigen_gap_ratio": minimum_gap,
        "numerical_rank": rank,
        "axis_direction_source": "canonical_cloud_axes",
    }
    if rank < 2:
        return mean.ravel(), axes, diagnostics, "rank_deficient"
    if np.min(ratios) < minimum_gap:
        return mean.ravel(), axes, diagnostics, "ambiguous_principal_axes"
    if reference is None:
        return mean.ravel(), axes, diagnostics, None
    dots = np.einsum("ij,ij->j", axes, reference)
    scores = _PROPER_SIGNS @ dots
    order = np.argsort(scores, kind="stable")
    gap = float(scores[order[-1]] - scores[order[-2]])
    diagnostics.update(axis_direction_source="reference_signs", reference_sign_score_gap=gap)
    if gap < 1e-6:
        return mean.ravel(), axes, diagnostics, "ambiguous_reference_signs"
    axes *= _PROPER_SIGNS[order[-1]]
    return mean.ravel(), axes, diagnostics, None


def _box(points, anchor, axes):
    local = (points - anchor) @ axes
    low, high = local.min(axis=0), local.max(axis=0)
    midpoint = (low + high) / 2
    center = anchor + axes @ midpoint
    dimensions = high - low
    corners = np.where(_CORNERS != 0, high, low) @ axes.T + anchor
    return center, dimensions, corners


class MeasureOrientedCloud(Tool3D):
    type_id = "3d.cloud_oriented_extents"
    title, subtype = "Measure oriented point-cloud extents", "Point clouds"
    description = "Measure a box enclosing observed XYZ, with checks for pose ambiguity."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "cloud"
    inputs = (
        Port("cloud", "cloud"),
        Port(
            "reference_pose",
            "data",
            required=False,
            required_when_connected=True,
            visual_type="pose",
        ),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("dimensions", "data", required=False),
        Port("report", "data"),
    )
    defaults = {"orientation": "pca", "minimum_eigen_gap_ratio": 0.05, "box_frame": "oriented_box"}
    parameter_choices = {"orientation": ("pca", "reference")}
    parameter_ranges = {"minimum_eigen_gap_ratio": NumericRange(1e-6, 1.0)}
    parameter_help = {
        "orientation": "Fit PCA axes or measure along the connected reference pose's rotation.",
        "minimum_eigen_gap_ratio": "Minimum relative gap between adjacent PCA variances.",
        "box_frame": "Name of the fitted box frame, distinct from the point-cloud frame.",
    }
    input_help = {
        "cloud": "Observed XYZ in meters or millimeters, in a named frame.",
        "reference_pose": "Axes in the cloud frame. In PCA mode this resolves axis signs only.",
    }
    output_help = {
        "pose": "Right-handed box center and orientation mapping the box frame into the cloud.",
        "dimensions": "Observed extents along the fitted box's X, Y, Z axes, in the cloud units.",
        "report": "Fit method, PCA quality when applicable, box bounds and any ambiguity.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["box_frame"], str) or not p["box_frame"].strip():
            raise ValidationError("Name the fitted box coordinate frame", "box_frame")

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in ("m", "mm"):
            raise ValidationError("Oriented extents require cloud units m or mm", "cloud")
        box_frame = p["box_frame"]
        if box_frame == cloud.frame:
            raise ValidationError("Box frame must differ from the cloud frame", "box_frame")
        reference = inputs.get("reference_pose")
        ref_axes = _reference_axes(reference, cloud.frame) if reference is not None else None
        method = p["orientation"]
        if method == "reference" and ref_axes is None:
            raise ValidationError("Reference orientation needs a reference pose", "reference_pose")
        report = {
            "orientation": method,
            "cloud_frame": cloud.frame,
            "box_frame": box_frame,
            "units": cloud.units,
            "point_count": len(cloud.points),
        }
        if reference is not None:
            report["reference_frame"] = reference["source_frame"]

        def failure(reason, message, graphics=None):
            report.update(pose_available=False, status=reason)
            return Result(
                {"pose": NO_VALUE, "dimensions": NO_VALUE, "report": report},
                passed=False,
                fail_reasons=[f"cloud_extents_{reason}"],
                graphics=graphics or {},
                message=message,
            )

        if not len(cloud.points) or (method == "pca" and len(cloud.points) < 3):
            return failure("insufficient_points", "Insufficient cloud points for oriented extents")
        points = cloud.points.astype(np.float64)
        issue = None
        if method == "pca":
            anchor, axes, quality, issue = _pca_axes(
                points, ref_axes, float(p["minimum_eigen_gap_ratio"])
            )
            report.update(quality)
        else:
            anchor, axes = points.mean(axis=0), ref_axes
            report["axis_direction_source"] = "reference_rotation"
        center, dimensions, corners = _box(points, anchor, axes)
        box = {
            "center": center.tolist(),
            "extent_xyz": dimensions.tolist(),
            "corners_xyz": corners.tolist(),
            "volume": float(np.prod(dimensions)),
        }
        graphics = {}
        if self.capture_graphics:
            graphics["Oriented box"] = render_cloud(
                cloud,
                limit=8000,
                segments=corners[np.asarray(_EDGES)],
                segment_color=(65, 190, 245) if issue else (80, 235, 125),
            )
        if issue:
            report["provisional_pca_box"] = box
            messages = {
                "rank_deficient": "Cloud has fewer than two independent spatial axes",
                "ambiguous_principal_axes": "Adjacent PCA variances give an ambiguous orientation",
                "ambiguous_reference_signs": "Reference does not resolve the PCA axis signs",
            }
            return failure(issue, messages[issue], graphics)
        report.update(box=box, pose_available=True, status="ok")
        pose = pose_record(pose_matrix(axes, center), box_frame, cloud.frame, cloud.units)
        return Result(
            {"pose": pose, "dimensions": dimensions.tolist(), "report": report},
            graphics=graphics,
            message=f"Measured {len(points):,} cloud points in {cloud.units}",
        )
