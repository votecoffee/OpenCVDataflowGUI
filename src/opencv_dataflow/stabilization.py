"""Causal stabilization of an inspection camera's image sequence."""

import math

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import gray_image
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_OUTPUT_NAMES = ("image", "coverage", "transform")


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
        return points[:0, 0], points[:0, 0]
    original, tracked = points[:, 0], found[:, 0]
    keep = status.ravel().astype(bool) & _inside(tracked, current.shape)
    keep &= patch_error.ravel() <= float(parameters["max_patch_error"])
    original, tracked = original[keep], tracked[keep]
    if mask is not None:
        pixels = np.rint(tracked).astype(np.intp)
        keep = mask[pixels[:, 1], pixels[:, 0]] != 0
        original, tracked = original[keep], tracked[keep]
    if not len(tracked):
        return original, tracked
    back, status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, tracked[:, None, :], None, **settings
    )
    if back is None or status is None:
        return original[:0], tracked[:0]
    error = np.linalg.norm(back[:, 0] - original, axis=1)
    keep = status.ravel().astype(bool) & (error <= float(parameters["forward_backward_pixels"]))
    return original[keep], tracked[keep]


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


class StabilizeCameraMotion(SequenceBlock):
    type_id, title = "sequence.stabilize_camera", "Stabilize camera motion"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
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
    }
    input_help = {
        "image": "Camera sequence in one consistent pixel grid.",
        "mask": "Optional static-background selection for feature centers.",
        "region": "Optional static-background region for motion estimation.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["min_inliers"]) > int(p["max_features"]):
            raise ValidationError(
                "Minimum inliers exceeds maximum features", "min_inliers", "max_features"
            )

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] + 8 * int(self.parameters["max_features"]) + 4096

    def _reanchor(self, state, reason, details):
        state["observed"] = np.zeros(4)
        state["smoothed"] = np.zeros(4)
        state["anchor"] = state.get("anchor", -1) + 1
        state["learning"] = 0
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
        points = cv2.goodFeaturesToTrack(
            gray,
            int(p["max_features"]),
            float(p["feature_quality"]),
            float(p["feature_spacing"]),
            mask=mask,
        )
        previous, previous_points = state.get("previous"), state.get("points")
        state.update(previous=gray.copy(), points=points)
        details = {"tracked": 0, "inliers": 0, "inlier_ratio": 0.0}
        if previous is None:
            return self._reanchor(state, "initial_frame", details)
        minimum = int(p["min_inliers"])
        if previous_points is None or len(previous_points) < minimum:
            return self._reanchor(state, "insufficient_features", details)

        old, new = _correspondences(previous, gray, previous_points, mask, p)
        details["tracked"] = len(old)
        if len(old) < minimum:
            return self._reanchor(state, "insufficient_tracks", details)
        step, inliers = _fit_motion(old, new, p["motion_model"], float(p["inlier_pixels"]))
        ratio = inliers / len(old)
        details.update(inliers=inliers, inlier_ratio=ratio)
        if step is None or inliers < minimum or ratio < float(p["min_inlier_ratio"]):
            return self._reanchor(state, "motion_fit_rejected", details)

        center = np.array([(width - 1) / 2, (height - 1) / 2])
        movement = step[:2, :2] @ center + step[:2, 2] - center
        angle = math.atan2(step[1, 0], step[0, 0])
        scale = math.hypot(step[0, 0], step[1, 0])
        scale_limit = float(p["max_step_scale_ratio"])
        if (
            np.linalg.norm(movement) > float(p["max_step_pixels"])
            or abs(math.degrees(angle)) > float(p["max_step_degrees"])
            or not 1 / scale_limit <= scale <= scale_limit
        ):
            return self._reanchor(state, "motion_step_rejected", details)

        previous_pose = state["observed"]
        accumulated = step @ _pose_matrix(previous_pose, center)
        translation = accumulated[:2, :2] @ center + accumulated[:2, 2] - center
        observed = np.array(
            [*translation, previous_pose[2] + angle, previous_pose[3] + math.log(scale)]
        )
        if not np.isfinite(observed).all() or abs(observed[3]) > math.log(2):
            return self._reanchor(state, "trajectory_out_of_range", details)
        smoothed = state["smoothed"] + float(p["follow"]) * (observed - state["smoothed"])
        correction = _pose_matrix(smoothed, center) @ np.linalg.inv(_pose_matrix(observed, center))
        size = (width, height)
        interpolation = _INTERPOLATION[p["interpolation"]]
        # The same interpolation and constant border define valid sample support.
        support = cv2.warpAffine(
            np.ones((height, width), dtype=np.float32),
            correction[:2],
            size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = support == 1
        covered = float(valid.mean())
        details["coverage_fraction"] = covered
        if covered == 0 or covered < float(p["min_coverage_fraction"]):
            return self._reanchor(state, "insufficient_coverage", details)
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
        state.update(observed=observed, smoothed=smoothed)
        anchor = state["anchor"]
        spatial = frame(size, f"{self.id}:stabilized:{state['generation']}:{anchor}")
        payload = self.input_payloads["image"]
        transform = {
            "model": "image_stabilization_v1",
            "source_to_target": correction.tolist(),
            "target_to_source": np.linalg.inv(correction).tolist(),
            "source_image_size": list(size),
            "target_image_size": list(size),
            "source_spatial": payload.get("spatial"),
            "source_pyramid": payload.get("pyramid"),
            "target_spatial": spatial,
        }
        return SequenceStep(
            {
                "image": ImageArray(output, encoding),
                "coverage": valid.astype(np.uint8) * 255,
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
