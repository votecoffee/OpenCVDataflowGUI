"""Causal stabilization of an inspection camera's image sequence."""

import math

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import gray_image
from .selection import selection_mask
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_OUTPUT_NAMES = ("image", "coverage", "transform")
_VALIDATION_INPUTS = ("validation_mask", "validation_region")


def _inside(points, shape):
    height, width = shape
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )


def _correspondences(previous, current, points, mask, parameters):
    window = int(parameters["tracking_window"])
    settings = {"winSize": (window, window), "maxLevel": int(parameters["max_pyramid_level"])}
    found, status, patch_error = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, **settings
    )
    if found is None or status is None or patch_error is None:
        return np.empty(0, dtype=np.intp), points[:0, 0]
    original, tracked = points[:, 0], found[:, 0]
    keep = status.ravel().astype(bool) & _inside(tracked, current.shape)
    keep &= patch_error.ravel() <= float(parameters["max_patch_error"])
    indices = np.flatnonzero(keep)
    original, tracked = original[keep], tracked[keep]
    if mask is not None:
        pixels = np.rint(tracked).astype(np.intp)
        keep = mask[pixels[:, 1], pixels[:, 0]] != 0
        indices = indices[keep]
        original, tracked = original[keep], tracked[keep]
    if not len(tracked):
        return indices, tracked
    back, status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, tracked[:, None, :], None, **settings
    )
    if back is None or status is None:
        return indices[:0], tracked[:0]
    error = np.linalg.norm(back[:, 0] - original, axis=1)
    keep = status.ravel().astype(bool) & (error <= float(parameters["forward_backward_pixels"]))
    return indices[keep], tracked[keep]


def _features(gray, mask, parameters, limit):
    return cv2.goodFeaturesToTrack(
        gray,
        int(parameters[limit]),
        float(parameters["feature_quality"]),
        float(parameters["feature_spacing"]),
        mask=mask,
    )


def _validate_motion(previous, current, state, mask, accumulated, parameters, details):
    report = details["validation"] = {"tracked": 0, "accepted": False}
    points = state["validation_points"]
    if points is None:
        return "insufficient_validation_tracks"
    indices, tracked = _correspondences(previous, current, points, mask, parameters)
    report["tracked"] = len(tracked)
    if len(tracked) < int(parameters["min_validation_tracks"]):
        return "insufficient_validation_tracks"
    anchor = state["validation_anchor"][indices]
    predicted = anchor @ accumulated[:2, :2].T + accumulated[:2, 2]
    scale = math.hypot(accumulated[0, 0], accumulated[1, 0])
    error = np.linalg.norm(tracked - predicted, axis=1) / scale
    inliers = int(np.count_nonzero(error <= float(parameters["max_validation_error_pixels"])))
    ratio = inliers / len(error)
    accepted = ratio >= float(parameters["min_validation_inlier_ratio"])
    report.update(
        inliers=inliers,
        inlier_ratio=ratio,
        median_error_pixels=float(np.median(error)),
        p90_error_pixels=float(np.percentile(error, 90)),
        accepted=accepted,
    )
    if not accepted:
        return "validation_motion_rejected"
    # Keep the original anchor coordinates for each surviving track.
    state.update(validation_points=tracked[:, None, :], validation_anchor=anchor)
    return None


def _fit_motion(previous, current, model, threshold):
    if model == "translation":
        displacements = (current - previous).astype(np.float64)
        offset = np.median(displacements, axis=0)
        inliers = np.linalg.norm(displacements - offset, axis=1) <= threshold
        matrix = np.eye(3)
        matrix[:2, 2] = offset
        return matrix, int(np.count_nonzero(inliers))
    affine, inliers = cv2.estimateAffinePartial2D(
        previous, current, method=cv2.RANSAC, ransacReprojThreshold=threshold
    )
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None, 0
    return np.vstack((affine, [0, 0, 1])), int(np.count_nonzero(inliers))


def _pose_matrix(trajectory, center):
    x, y, angle, log_scale = trajectory
    cosine, sine = math.cos(angle), math.sin(angle)
    linear = math.exp(log_scale) * np.array([[cosine, -sine], [sine, cosine]])
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = center + [x, y] - linear @ center
    return transform


def _estimate_pose(previous, current, points, mask, center, previous_pose, p, details):
    minimum = int(p["min_inliers"])
    if points is None or len(points) < minimum:
        return None, None, "insufficient_features"
    indices, new = _correspondences(previous, current, points, mask, p)
    details["tracked"] = len(new)
    if len(new) < minimum:
        return None, None, "insufficient_tracks"
    step, inliers = _fit_motion(
        points[indices, 0], new, p["motion_model"], float(p["inlier_pixels"])
    )
    ratio = inliers / len(new)
    details.update(inliers=inliers, inlier_ratio=ratio)
    if step is None or inliers < minimum or ratio < float(p["min_inlier_ratio"]):
        return None, None, "motion_fit_rejected"
    movement = step[:2, :2] @ center + step[:2, 2] - center
    angle = math.atan2(step[1, 0], step[0, 0])
    scale = math.hypot(step[0, 0], step[1, 0])
    scale_limit = float(p["max_step_scale_ratio"])
    if (
        np.linalg.norm(movement) > float(p["max_step_pixels"])
        or abs(math.degrees(angle)) > float(p["max_step_degrees"])
        or not 1 / scale_limit <= scale <= scale_limit
    ):
        return None, None, "motion_step_rejected"
    accumulated = step @ _pose_matrix(previous_pose, center)
    translation = accumulated[:2, :2] @ center + accumulated[:2, 2] - center
    observed = np.array(
        [*translation, previous_pose[2] + angle, previous_pose[3] + math.log(scale)]
    )
    if not np.isfinite(observed).all() or abs(observed[3]) > math.log(2):
        return None, None, "trajectory_out_of_range"
    return observed, accumulated, None


def _warp_correction(source, correction, p):
    size = source.shape[1::-1]
    interpolation = _INTERPOLATION[p["interpolation"]]
    # The same interpolation and constant border define valid sample support.
    support = cv2.warpAffine(
        np.ones(source.shape[:2], dtype=np.float32),
        correction[:2],
        size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = support == 1
    covered = float(valid.mean())
    if covered == 0 or covered < float(p["min_coverage_fraction"]):
        return None, None, covered
    output = cv2.warpAffine(
        np.asarray(source),
        correction[:2],
        size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
    output[~valid] = 0
    if output.dtype.kind == "f" and not np.isfinite(output).all():
        raise ValidationError("Stabilization produced nonfinite samples")
    return output, valid.astype(np.uint8) * 255, covered


def _transform_payload(correction, size, payload, spatial):
    return {
        "model": "image_stabilization_v1",
        "source_to_target": correction.tolist(),
        "target_to_source": np.linalg.inv(correction).tolist(),
        "source_image_size": list(size),
        "target_image_size": list(size),
        "source_spatial": payload.get("spatial"),
        "source_pyramid": payload.get("pyramid"),
        "target_spatial": spatial,
    }


class StabilizeCameraMotion(SequenceBlock):
    type_id, title = "sequence.stabilize_camera", "Stabilize camera motion"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("validation_mask", required=False, required_when_connected=True),
        Port("validation_region", "data", required=False, required_when_connected=True),
        *SequenceBlock.inputs[3:],
    )
    data_outputs = (
        Port("image"),
        Port("coverage", visual_type="region"),
        Port("transform", "data"),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "motion_model": "similarity",
        "follow": 0.0,
        "max_features": 240,
        "feature_quality": 0.01,
        "feature_spacing": 8.0,
        "tracking_window": 21,
        "max_pyramid_level": 3,
        "forward_backward_pixels": 1.0,
        "max_patch_error": 30.0,
        "inlier_pixels": 2.0,
        "min_inliers": 12,
        "min_inlier_ratio": 0.5,
        "max_step_pixels": 40.0,
        "max_step_degrees": 12.0,
        "max_step_scale_ratio": 1.15,
        "min_coverage_fraction": 0.65,
        "max_validation_features": 120,
        "min_validation_tracks": 12,
        "max_validation_error_pixels": 2.0,
        "min_validation_inlier_ratio": 0.8,
        "interpolation": "linear",
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "motion_model": ("translation", "similarity"),
        "interpolation": tuple(_INTERPOLATION),
    }
    parameter_ranges = {
        "follow": NumericRange(0, 1),
        "max_features": NumericRange(3, 5000, whole=True),
        "feature_quality": NumericRange(1e-6, 1),
        "feature_spacing": NumericRange(0, 512),
        "tracking_window": NumericRange(3, 255, whole=True, odd=True),
        "max_pyramid_level": NumericRange(0, 8, whole=True),
        "forward_backward_pixels": NumericRange(1e-6, 100),
        "max_patch_error": NumericRange(0, 255),
        "inlier_pixels": NumericRange(1e-6, 100),
        "min_inliers": NumericRange(3, 5000, whole=True),
        "min_inlier_ratio": NumericRange(0, 1),
        "max_step_pixels": NumericRange(1e-6, 16384),
        "max_step_degrees": NumericRange(0, 180),
        "max_step_scale_ratio": NumericRange(1, 2),
        "min_coverage_fraction": NumericRange(0, 1),
        "max_validation_features": NumericRange(1, 5000, whole=True),
        "min_validation_tracks": NumericRange(1, 5000, whole=True),
        "max_validation_error_pixels": NumericRange(1e-6, 16384),
        "min_validation_inlier_ratio": NumericRange(1e-6, 1),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "motion_model": "Translation shifts the view. Similarity also tracks rotation and scale.",
        "follow": "Zero locks to the anchor. Intermediate values smooth the observed trajectory.",
        "forward_backward_pixels": "Maximum error when a tracked point is mapped back.",
        "max_patch_error": "Maximum mean patch error on the 8-bit tracking image.",
        "inlier_pixels": "Reprojection tolerance for a track to support the motion estimate.",
        "max_step_pixels": "Largest accepted center displacement between consecutive frames.",
        "max_step_degrees": "Largest accepted rotation between consecutive frames.",
        "max_step_scale_ratio": "Allowed scale range is the reciprocal of this ratio to the ratio.",
        "min_coverage_fraction": "Re-anchor when too little of the corrected canvas is covered.",
        "max_validation_features": "Maximum features seeded inside the check area at each anchor.",
        "min_validation_tracks": "Minimum surviving anchor tracks required in the check area.",
        "max_validation_error_pixels": "Allowed residual displacement in anchor pixels.",
        "min_validation_inlier_ratio": "Fraction of check tracks that must meet the error limit.",
    }
    input_help = {
        "image": "Camera sequence in one consistent pixel grid.",
        "mask": "Optional static-background selection for feature centers.",
        "region": "Optional static-background region for motion estimation.",
        "validation_mask": "Check the fitted motion on these pixels and exclude them from fitting.",
        "validation_region": "Static area to validate against the anchor. Excluded from fitting.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["min_inliers"]) > int(p["max_features"]):
            raise ValidationError(
                "Minimum inliers exceeds maximum features", "min_inliers", "max_features"
            )
        if int(p["min_validation_tracks"]) > int(p["max_validation_features"]):
            raise ValidationError(
                "Minimum validation tracks exceeds maximum validation features",
                "min_validation_tracks",
                "max_validation_features",
            )

    def extra_context(self, inputs):
        return "validation" if any(name in inputs for name in _VALIDATION_INPUTS) else None

    def estimate_state_bytes(self, image):
        p = self.parameters
        validation_bytes = (
            16 * int(p["max_validation_features"]) if self.extra_context(self.input_payloads) else 0
        )
        return (
            image.shape[0] * image.shape[1] + 8 * int(p["max_features"]) + 4096 + validation_bytes
        )

    def _reanchor(self, state, reason, details, validation):
        state["observed"] = np.zeros(4)
        state["smoothed"] = np.zeros(4)
        state["anchor"] = state.get("anchor", -1) + 1
        state["learning"] = 0
        if validation is not None:
            points = _features(
                state["previous"], validation, self.parameters, "max_validation_features"
            )
            state["validation_points"] = points
            state["validation_anchor"] = points[:, 0] if points is not None else None
            report = details.setdefault("validation", {})
            report["new_anchor_features"] = 0 if points is None else len(points)
        return SequenceStep(
            dict.fromkeys(_OUTPUT_NAMES, NO_VALUE),
            ready=False,
            details={**details, "reason": reason, "anchor": state["anchor"]},
        )

    def step(self, inputs, state, mask):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 8 or height * width > 16_777_216:
            raise ValidationError("Use at least 8 × 8 pixels and at most 16,777,216 pixels")
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Stabilization requires grayscale or RGB/BGR encoding")
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        p = self.parameters
        validation = None
        if self.extra_context(inputs):
            validation = selection_mask(
                {
                    "image": source,
                    "mask": inputs.get("validation_mask"),
                    "region": inputs.get("validation_region"),
                },
                source,
            )
            if validation is None:
                raise ValidationError("Connect a validation mask or region")
            outside_validation = cv2.bitwise_not(validation)
            mask = outside_validation if mask is None else cv2.bitwise_and(mask, outside_validation)
        points = _features(gray, mask, p, "max_features")
        previous, previous_points = state.get("previous"), state.get("points")
        state.update(previous=gray.copy(), points=points)
        details = {"tracked": 0, "inliers": 0, "inlier_ratio": 0.0}
        if previous is None:
            return self._reanchor(state, "initial_frame", details, validation)
        center = np.array([(width - 1) / 2, (height - 1) / 2])
        observed, accumulated, reason = _estimate_pose(
            previous, gray, previous_points, mask, center, state["observed"], p, details
        )
        if reason is not None:
            return self._reanchor(state, reason, details, validation)
        if validation is not None:
            reason = _validate_motion(previous, gray, state, validation, accumulated, p, details)
            if reason is not None:
                return self._reanchor(state, reason, details, validation)
        smoothed = state["smoothed"] + float(p["follow"]) * (observed - state["smoothed"])
        correction = _pose_matrix(smoothed, center) @ np.linalg.inv(_pose_matrix(observed, center))
        size = (width, height)
        output, coverage, covered = _warp_correction(source, correction, p)
        details["coverage_fraction"] = covered
        if output is None:
            return self._reanchor(state, "insufficient_coverage", details, validation)
        state.update(observed=observed, smoothed=smoothed)
        anchor = state["anchor"]
        spatial = frame(size, f"{self.id}:stabilized:{state['generation']}:{anchor}")
        payload = self.input_payloads["image"]
        transform = _transform_payload(correction, size, payload, spatial)
        return SequenceStep(
            {
                "image": ImageArray(output, encoding),
                "coverage": coverage,
                "transform": transform,
            },
            details={
                **details,
                "reason": "ready",
                "anchor": anchor,
                "observed_translation": observed[:2].tolist(),
                "observed_angle_degrees": math.degrees(observed[2]),
            },
            output_frames=dict.fromkeys(_OUTPUT_NAMES, spatial),
        )
