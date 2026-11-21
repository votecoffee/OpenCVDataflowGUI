"""Fit a native OpenCV line to measured XYZ and qualify its observed support."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, rigid
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud
from .shape_fitting import LINE_DISTANCES

_MAX_NORMALIZED_EXTENT = 1_000_000
_SCALE_DEPENDENT = {"fair", "welsch", "huber"}


def _reference_axis(reference, frame):
    if (
        not isinstance(reference, dict)
        or reference.get("model") != "pose_3d_v1"
        or reference.get("target_frame") != frame
    ):
        raise ValidationError("Reference pose must map into the cloud frame", "reference_pose")
    return rigid(reference)[:3, 2]


def _native_fit(points, scale, p):
    center = points.mean(axis=0)
    local = (points - center) / scale
    if np.max(np.abs(local)) > _MAX_NORMALIZED_EXTENT:
        raise ValidationError(
            "Cloud extent exceeds the native fit range at this inlier tolerance", "inlier_tolerance"
        )
    native = local.astype(np.float32)
    if not np.any(native != native[0]):
        return None
    result = cv2.fitLine(
        native[:, None, :],
        LINE_DISTANCES[p["distance"]],
        float(p["robust_scale"]) / scale,
        float(p["radius_epsilon"]) / scale,
        float(p["angle_epsilon_rad"]),
    )
    fit = np.asarray(result, dtype=np.float64).ravel()
    if fit.shape != (6,) or not np.isfinite(fit).all():
        raise ValueError("OpenCV returned invalid XYZ line coefficients")
    norm = np.linalg.norm(fit[:3])
    if norm < 1e-9:
        return None
    axis = fit[:3] / norm
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return center + fit[3:] * scale, axis


def _linearity(support):
    centered = support - support.mean(axis=0)
    spectrum = np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0)
    return float(1 - spectrum[-2] / spectrum[-1]) if spectrum[-1] else 0.0


def _preview(cloud, keep, ends, passed):
    selected = np.linspace(0, len(keep) - 1, min(len(keep), 12_000), dtype=int)
    colors = np.array([[235, 101, 83], [106, 218, 144]], np.uint8)[keep[selected].astype(int)]
    view = PointCloud(cloud.points[selected], cloud.units, cloud.frame, colors=colors)
    segment = None if ends is None else ends[None]
    return render_cloud(
        view,
        segments=segment,
        segment_color=(90, 234, 145) if passed else (75, 184, 242),
    )


class FitCloudLine(Tool3D):
    type_id = "3d.cloud_fit_line"
    title, subtype = "Fit 3D line to point cloud", "Point clouds"
    description = "Fit a 3D axis using OpenCV and measure the support from the retained scan rows."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
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
        Port("line", "data", required=False),
        Port("endpoints", "data", required=False),
        Port("span", "number", required=False),
        Port("inliers", "cloud", required=False),
        Port("outliers", "cloud", required=False),
        Port("inlier_indices", "data", required=False),
        Port("outlier_indices", "data", required=False),
        Port("radial_distances", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "distance": "welsch",
        "robust_scale": 0.0,
        "radius_epsilon": 0.0,
        "angle_epsilon_rad": 0.01,
        "inlier_tolerance": 0.002,
        "minimum_inliers": 3,
        "minimum_inlier_fraction": 0.5,
        "minimum_linearity": 0.8,
        "minimum_span": 0.0,
        "minimum_reference_alignment": 0.05,
        "maximum_fit_samples": 4096,
        "max_input_points": 250_000,
        "random_seed": 0,
    }
    parameter_choices = {"distance": tuple(LINE_DISTANCES)}
    parameter_ranges = {
        "robust_scale": NumericRange(0, 1e12),
        "radius_epsilon": NumericRange(0, 1e12),
        "angle_epsilon_rad": NumericRange(0, np.pi / 2),
        "inlier_tolerance": NumericRange(1e-12, 1e12),
        "minimum_inliers": NumericRange(2, 1_000_000, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_linearity": NumericRange(1e-6, 1),
        "minimum_span": NumericRange(0, 1e12),
        "minimum_reference_alignment": NumericRange(1e-6, 1),
        "maximum_fit_samples": NumericRange(2, 16_384, whole=True),
        "max_input_points": NumericRange(2, 1_000_000, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "distance": "OpenCV loss for point-to-line residuals, from least squares to robust loss.",
        "robust_scale": "Fair, Welsch or Huber scale in cloud units. Zero uses the native default.",
        "radius_epsilon": "Convergence distance in cloud units. Zero uses the native default.",
        "angle_epsilon_rad": "Angular convergence in radians. Zero uses the native default.",
        "inlier_tolerance": "Inclusive radial threshold and fit scale, both in cloud units.",
        "minimum_inliers": "Minimum number of supported input rows.",
        "minimum_inlier_fraction": "Minimum supported fraction of all input rows.",
        "minimum_linearity": "Minimum 1 − second/first principal variance of the supporting rows.",
        "minimum_span": "Smallest accepted support span along the line, in cloud units.",
        "minimum_reference_alignment": "Minimum absolute cosine with the reference pose's +Z.",
        "maximum_fit_samples": "Maximum rows sent to the fitter. Classification uses every row.",
        "max_input_points": "Maximum cloud row count to classify.",
        "random_seed": "Seed for choosing fit rows when the cloud exceeds the sample budget.",
    }
    input_help = {
        "cloud": "Measured XYZ, with its native frame, length units and optional point attributes.",
        "reference_pose": "Use this pose's +Z in the cloud frame to resolve the line direction.",
    }
    output_help = {
        "line": "Point, direction and support limits in the cloud frame. Check orientation status.",
        "endpoints": "Ordered XYZ limits of the inliers projected onto the fitted line.",
        "span": "Distance between the projected inlier endpoints in the cloud's units.",
        "inliers": "Supporting rows in original order with their original attributes.",
        "outliers": "Rows outside the radial threshold with their original attributes.",
        "inlier_indices": "Zero-based input row indices. Source scan IDs stay on the cloud.",
        "outlier_indices": "Zero-based input row indices of the rejected points.",
        "radial_distances": "Unsigned radial distances in input order, in cloud units.",
        "report": "Support, residuals, direction convention and the acceptance status.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_inliers"]) > int(p["max_input_points"]):
            raise ValidationError(
                "Minimum inliers must fit within the cloud budget",
                "minimum_inliers",
                "max_input_points",
            )
        scale = float(p["robust_scale"])
        if scale and p["distance"] not in _SCALE_DEPENDENT:
            raise ValidationError(
                "Only Fair, Welsch and Huber use a robust scale", "robust_scale", "distance"
            )
        if scale and not 1e-6 <= scale / float(p["inlier_tolerance"]) <= 1e6:
            raise ValidationError(
                "Robust scale is outside the native range relative to the inlier tolerance",
                "robust_scale",
                "inlier_tolerance",
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        reference = (
            _reference_axis(inputs["reference_pose"], cloud.frame)
            if "reference_pose" in inputs
            else None
        )
        count = len(cloud.points)
        if count > int(p["max_input_points"]):
            raise ValidationError("Cloud exceeds the input point budget", "max_input_points")
        tolerance = float(p["inlier_tolerance"])
        report = {
            "model": "cloud_line_fit_report_v1",
            "distance": p["distance"],
            "frame": cloud.frame,
            "units": cloud.units,
            "input_points": count,
            "native_input_scale": tolerance,
            "inlier_tolerance": tolerance,
            "fit_samples": 0,
            "inlier_count": 0,
            "inlier_fraction": 0.0,
        }
        outputs = dict.fromkeys((port.name for port in self.outputs), NO_VALUE)
        outputs["report"] = report
        keep, ends = None, None

        def failure(code, message):
            report.update(status=code, passed=False)
            graphics = {}
            if keep is not None and self.capture_graphics:
                graphics["Line support"] = _preview(cloud, keep, ends, False)
            return Result(
                outputs,
                passed=False,
                message=message,
                score=report["inlier_fraction"],
                fail_reasons=[f"cloud_line_{code}"],
                graphics=graphics,
            )

        if count < int(p["minimum_inliers"]):
            return failure(
                "insufficient_points", "Cloud has fewer points than the required support"
            )
        if self._cancelled():
            return failure("cancelled", "Line fitting cancelled")
        points = cloud.points.astype(np.float64)
        if np.max(np.ptp(points, axis=0)) == 0:
            return failure("coincident", "The cloud has no distinct spatial positions")
        if np.max(np.ptp(points, axis=0)) > 2 * _MAX_NORMALIZED_EXTENT * tolerance:
            raise ValidationError(
                "Cloud extent exceeds the native fit range at this inlier tolerance",
                "inlier_tolerance",
            )
        sample_count = min(count, int(p["maximum_fit_samples"]))
        report["fit_samples"] = sample_count
        if sample_count == count:
            sample = points
        else:
            rng = np.random.default_rng(int(p["random_seed"]))
            sample = points[np.sort(rng.choice(count, sample_count, replace=False))]
        fit = _native_fit(sample, tolerance, p)
        if self._cancelled():
            return failure("cancelled", "Line fitting cancelled")
        if fit is None:
            return failure("no_native_axis", "Native fit rows do not define an axis")

        origin, axis = fit
        alignment = float(np.clip(axis @ reference, -1, 1)) if reference is not None else None
        oriented = alignment is not None and abs(alignment) >= float(
            p["minimum_reference_alignment"]
        )
        if oriented and alignment < 0:
            axis = -axis
        centered = points - origin
        along = centered @ axis
        errors = np.linalg.norm(centered - np.outer(along, axis), axis=1)
        keep = errors <= tolerance
        supported = int(keep.sum())
        fraction = supported / count
        report.update(
            inlier_count=supported,
            inlier_fraction=fraction,
            radial_rms=float(np.sqrt(np.mean(errors**2))),
            reference_alignment_abs=abs(alignment) if alignment is not None else None,
            direction_is_oriented=oriented,
        )
        if supported < int(p["minimum_inliers"]) or fraction < float(p["minimum_inlier_fraction"]):
            return failure(
                "insufficient_support", "Fitted line has too few rows within radial tolerance"
            )

        limits = np.array([along[keep].min(), along[keep].max()])
        span = float(limits[1] - limits[0])
        ends = origin + limits[:, None] * axis
        linearity = _linearity(points[keep])
        report.update(
            span=span,
            support_linearity=linearity,
            candidate_endpoints=ends.tolist(),
            inlier_radial_rms=float(np.sqrt(np.mean(errors[keep] ** 2))),
        )
        if span <= 0 or span < float(p["minimum_span"]):
            return failure(
                "insufficient_span", "Supporting points span too little of the fitted axis"
            )
        if linearity < float(p["minimum_linearity"]):
            return failure(
                "ambiguous_axis", "Supporting rows do not constrain one dominant linear direction"
            )
        if reference is not None and not oriented:
            return failure(
                "ambiguous_sign", "Reference +Z does not resolve the fitted line direction"
            )
        midpoint = origin + limits.mean() * axis
        convention = "reference_z" if oriented else "canonical_dominant_component"
        report.update(status="ok", passed=True, direction_convention=convention)
        outputs.update(
            line={
                "schema": "point_cloud_model_v1",
                "kind": "line",
                "frame": cloud.frame,
                "units": cloud.units,
                "point_on_axis": midpoint.tolist(),
                "axis_direction": axis.tolist(),
                "direction_is_oriented": oriented,
                "direction_convention": convention,
                "axial_limits": [-span / 2, span / 2],
                "observed_endpoints": ends.tolist(),
                "span": span,
            },
            endpoints=ends.tolist(),
            span=span,
        )
        for name, selected in (("inlier", keep), ("outlier", ~keep)):
            if f"{name}s" in self.exposed_outputs:
                outputs[f"{name}s"] = cloud.subset(selected)
            if f"{name}_indices" in self.exposed_outputs:
                outputs[f"{name}_indices"] = np.flatnonzero(selected).tolist()
        if "radial_distances" in self.exposed_outputs:
            outputs["radial_distances"] = errors.tolist()
        graphics = (
            {"Line support": _preview(cloud, keep, ends, True)} if self.capture_graphics else {}
        )
        return Result(
            outputs,
            score=fraction,
            graphics=graphics,
            message=f"{supported}/{count} support rows · span {span:.4g} {cloud.units}",
        )
