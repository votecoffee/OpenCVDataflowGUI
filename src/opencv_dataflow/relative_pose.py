"""Relative pose from independent pinhole, fisheye, or Mei camera models."""

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration_revisions import REVISION, check_capture
from .core import Port, Result
from .epipolar_refinement import sampson_pixels
from .geometry3d_common import (
    Tool3D,
    camera,
    checked_camera_rays,
    forward_camera_domain,
    points,
    pose_matrix,
    pose_record,
    project_camera_pixels,
)
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .pointclouds import PointCloud
from .result_graphics import ACCEPTED, REJECTED

_MAX_PAIRS = 4096
_K = np.eye(3)
_P_LEFT = np.eye(3, 4)
_ZERO = np.zeros(3)
_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-12)
_METHODS = {"ransac": cv2.RANSAC, "lmeds": cv2.LMEDS}


@dataclass(frozen=True)
class _View:
    model: dict
    k: np.ndarray
    d: np.ndarray
    size: tuple
    xy: np.ndarray
    rays: np.ndarray
    valid: np.ndarray
    image: object
    payload: dict


@dataclass(frozen=True)
class _Candidate:
    rotation: np.ndarray
    direction: np.ndarray
    essential: np.ndarray
    kept: np.ndarray
    xyz: np.ndarray
    errors: tuple
    median_parallax: float | None
    cheirality: int
    parallax_count: int

    @property
    def score(self):
        count = int(self.kept.sum())
        error = sum(np.mean(view[self.kept] ** 2) for view in self.errors) if count else np.inf
        return count, -error


def _essential(rotation, direction):
    x, y, z = direction
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ rotation


def _normalized_rank(pixels):
    xy = pixels - pixels.mean(axis=0)
    singular = np.linalg.svd(xy, compute_uv=False)
    return singular[0] > 0 and singular[1] > singular[0] * 1e-7


def _measure(rotation, direction, views, indices, p):
    first, second = views
    a, b = first.rays[indices], second.rays[indices]
    essential = _essential(rotation, direction)
    plausible = sampson_pixels(a, b, essential) <= float(p["normalized_threshold"])
    projection_right = np.column_stack((rotation, direction))
    reduced = np.flatnonzero(plausible)
    xyz = np.full((len(a), 3), np.nan)
    if len(reduced):
        h = cv2.triangulatePoints(_P_LEFT, projection_right, a[reduced].T, b[reduced].T).T
        finite = np.isfinite(h).all(axis=1) & (np.abs(h[:, 3]) > 1e-12)
        xyz[reduced[finite]] = h[finite, :3] / h[finite, 3:4]
    right_xyz = xyz @ rotation.T + direction
    depth = float(p["maximum_depth_baselines"])
    cosine = float(p["minimum_ray_cosine"])
    forward = (
        plausible
        & forward_camera_domain(xyz, cosine)
        & forward_camera_domain(right_xyz, cosine)
        & (xyz[:, 2] <= depth)
        & (right_xyz[:, 2] <= depth)
    )
    first_bearing = np.c_[a, np.ones(len(a))]
    second_bearing = np.c_[b, np.ones(len(b))] @ rotation
    angular = np.degrees(
        np.arctan2(
            np.linalg.norm(np.cross(first_bearing, second_bearing), axis=1),
            np.einsum("ij,ij->i", first_bearing, second_bearing),
        )
    )
    enough_parallax = forward & (angular >= float(p["minimum_parallax_deg"]))
    errors = []
    for view, local in zip(views, (xyz, right_xyz), strict=True):
        r = np.full(len(a), np.nan)
        if np.any(forward):
            projected = project_camera_pixels(
                local[forward], view.model, view.k, view.d, _ZERO, _ZERO
            )
            r[forward] = np.linalg.norm(projected - view.xy[indices[forward]], axis=1)
        errors.append(r)
    kept = enough_parallax & (np.maximum(*errors) <= float(p["maximum_reprojection_px"]))
    return _Candidate(
        rotation,
        direction,
        essential,
        kept,
        xyz,
        tuple(errors),
        float(np.median(angular[kept])) if np.any(kept) else None,
        int(forward.sum()),
        int(enough_parallax.sum()),
    )


def _recover(essential, proposal, views, indices, p):
    a, b = (view.rays[indices] for view in views)
    _, rotation, translation, _, _ = cv2.recoverPose(
        essential,
        a,
        b,
        _K,
        distanceThresh=float(p["maximum_depth_baselines"]),
        mask=proposal.astype(np.uint8).reshape(-1, 1),
    )
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("OpenCV returned nonfinite relative pose coefficients")
    magnitude = np.linalg.norm(translation)
    if not magnitude or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("OpenCV returned an invalid relative rotation or direction")
    return _measure(rotation, translation.ravel() / magnitude, views, indices, p)


def _eight_point_refit(views, indices):
    a, b = (view.rays[indices] for view in views)
    matrix, _ = cv2.findFundamentalMat(a, b, cv2.FM_8POINT)
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    u, s, v = np.linalg.svd(matrix)
    if s[1] <= s[0] * 1e-9:
        return None
    scale = (s[0] + s[1]) / 2
    return u @ np.diag([scale, scale, 0]) @ v


def _finite(values, indices, count):
    full = np.full(count, np.nan)
    full[indices] = values
    return [float(x) if np.isfinite(x) else None for x in full]


class IndependentRelativePose(Tool3D):
    type_id, title = "3d.independent_relative_pose", "Relative pose from two calibrated cameras"
    subtype = "Reconstruction"
    description = "Recover a left-to-right pose from matched pixels on two independent lens models."
    inputs = (
        Port("points_left", "data"),
        Port("points_right", "data"),
        Port("camera_left", "data"),
        Port("camera_right", "data"),
        Port("left_image", required=False, required_when_connected=True),
        Port("right_image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("essential", "data", required=False),
        Port("inlier_indices", "data"),
        Port("cloud", "cloud", required=False),
        Port("report", "data"),
        Port("left_image", required=False),
        Port("right_image", required=False),
    )
    defaults = {
        "method": "ransac",
        "confidence": 0.999,
        "iterations": 3000,
        "normalized_threshold": 0.0015,
        "maximum_reprojection_px": 1.5,
        "minimum_inliers": 12,
        "minimum_inlier_fraction": 0.55,
        "minimum_parallax_deg": 0.5,
        "maximum_depth_baselines": 100.0,
        "minimum_ray_cosine": 0.05,
        "maximum_inversion_error_px": 0.1,
        "refit": True,
        "baseline_length": 0.0,
        "baseline_units": "m",
    }
    parameter_choices = {"method": tuple(_METHODS), "baseline_units": ("m", "mm")}
    parameter_ranges = {
        "confidence": NumericRange(0.5001, 0.9999),
        "iterations": NumericRange(1, 20_000, whole=True),
        "normalized_threshold": NumericRange(1e-7, 1.0),
        "maximum_reprojection_px": NumericRange(1e-5, 1000.0),
        "minimum_inliers": NumericRange(8, _MAX_PAIRS, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_parallax_deg": NumericRange(0.01, 90.0),
        "maximum_depth_baselines": NumericRange(1.0, 1e5),
        "minimum_ray_cosine": NumericRange(0.001, 0.999),
        "maximum_inversion_error_px": NumericRange(1e-6, 10.0),
        "baseline_length": NumericRange(1e-9, 1e9, zero_allowed=True),
    }
    parameter_help = {
        "method": "Robust estimator for the normalized essential matrix.",
        "confidence": "Confidence of the native estimator.",
        "iterations": "Proposal limit for the native estimator.",
        "normalized_threshold": "Inlier tolerance in normalized rays, at unit focal length.",
        "maximum_reprojection_px": "Largest final error allowed in either original pixel grid.",
        "minimum_inliers": "Required correspondences passing cheirality and both lens checks.",
        "minimum_inlier_fraction": "Required fraction of all supplied pairs.",
        "minimum_parallax_deg": "Minimum angle between matched rays in a common frame.",
        "maximum_depth_baselines": "Maximum triangulated Z in each view, in baseline lengths.",
        "minimum_ray_cosine": "Minimum positive Z of a unit ray in each camera.",
        "maximum_inversion_error_px": "Maximum ray inversion round-trip error in source pixels.",
        "refit": "Try an eight-point refit on the source-pixel consensus.",
        "baseline_length": "Measured camera center separation. Zero keeps the pose unitless.",
        "baseline_units": "Units of a supplied measured baseline.",
    }
    input_help = {
        "points_left": "Left raw-image XY rows, ordered to match the right rows.",
        "points_right": "Right raw-image XY rows, ordered to match the left rows.",
        "camera_left": "Calibrated lens and pixel grid for the left observations.",
        "camera_right": "Independently calibrated lens and pixel grid for the right observations.",
        "left_image": "Optional original left image for accepted/rejected pair graphics.",
        "right_image": "Optional original right image for accepted/rejected pair graphics.",
    }
    output_help = {
        "pose": "Left-camera coordinates to right-camera coordinates. Unitless unless anchored.",
        "essential": "Normalized essential matrix for the left-to-right constraint.",
        "inlier_indices": "Original row indices accepted by both lenses, in input order.",
        "cloud": "Triangulated inliers in the left frame. Source indices record pair row IDs.",
        "report": "Separate image geometry, support, residuals and baseline direction.",
    }
    payload_reference = "points_left"
    preview_safe = parallel_safe = process_inputs_readonly = True
    composes_result_image = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["refit"], bool):
            raise ValidationError("Refit must be a boolean", "refit")

    def _view(self, inputs, side):
        model = inputs[f"camera_{side}"]
        k, d, size = camera(model)
        model = {
            **model,
            "frame": model.get("frame", "camera"),
            "lens": model.get("lens", "pinhole"),
        }
        if model["lens"] == "omnidir":
            xi = float(model["xi"])
            if xi < 0:
                raise ValidationError(
                    f"{side.title()} omnidirectional camera requires nonnegative xi"
                )
            model["xi"] = xi
        key, image_key = f"points_{side}", f"{side}_image"
        xy = points(inputs[key], 2, 8)
        if len(xy) > _MAX_PAIRS or (xy < 0).any() or (xy >= size).any():
            raise ValidationError(f"Use up to {_MAX_PAIRS} points inside the {side} camera grid")
        payload = self.input_payloads[key]
        spatial = payload.get("spatial")
        if spatial and spatial["image_size"] != list(size):
            raise ValidationError(f"{side.title()} points use a different calibrated image grid")
        image = inputs.get(image_key)
        if image is not None:
            image_payload = self.input_payloads[image_key]
            if image.shape[:2][::-1] != size:
                raise ValidationError(f"{side.title()} image does not match its camera grid")
            require_same_frame(
                (ValuePacket(xy, payload), ValuePacket(image, image_payload)),
                message=f"{side.title()} points and image use different coordinate frames",
            )
            if payload.get("ticket") is not None and image_payload.get("ticket") is not None:
                if payload["ticket"] != image_payload["ticket"]:
                    raise ValidationError(
                        f"{side.title()} points and image must describe the same capture"
                    )
            capture = image_payload
            payload = {**payload, **image_payload}
        else:
            capture = payload
        check_capture(model.get(REVISION), capture, size)
        rays, valid = checked_camera_rays(
            xy,
            model,
            k,
            d,
            _CRITERIA,
            float(self.parameters["minimum_ray_cosine"]),
            float(self.parameters["maximum_inversion_error_px"]),
        )
        return _View(model, k, d, size, xy, rays, valid, image, payload)

    def process(self, inputs):
        p = self.parameters
        views = left, right = tuple(self._view(inputs, side) for side in ("left", "right"))
        count = len(left.xy)
        if count != len(right.xy):
            raise ValidationError("Left and right correspondence counts differ")
        first, second = left.model["frame"], right.model["frame"]
        if first == second:
            raise ValidationError("Use distinct camera frame names")
        indices = np.flatnonzero(left.valid & right.valid)
        minimum = max(
            int(p["minimum_inliers"]), int(np.ceil(float(p["minimum_inlier_fraction"]) * count))
        )
        report = {
            "model": "independent_relative_pose_report_v1",
            "left_frame": first,
            "right_frame": second,
            "left_lens": left.model["lens"],
            "right_lens": right.model["lens"],
            "left_image_size": list(left.size),
            "right_image_size": list(right.size),
            "input_pairs": count,
            "method": p["method"],
            "invalid_left_ray_indices": np.flatnonzero(~left.valid).tolist(),
            "invalid_right_ray_indices": np.flatnonzero(~right.valid).tolist(),
            "refit_adopted": False,
        }
        output = {
            "pose": NO_VALUE,
            "essential": NO_VALUE,
            "inlier_indices": [],
            "cloud": NO_VALUE,
            "report": report,
            "left_image": left.image if left.image is not None else NO_VALUE,
            "right_image": right.image if right.image is not None else NO_VALUE,
        }

        def finish(code, message):
            report.update(status=code or "ok", passed=code is None)
            payloads = {
                key: left.payload for key in output if key not in ("left_image", "right_image")
            }
            drawings = {}
            kept = np.zeros(count, dtype=bool)
            kept[output["inlier_indices"]] = True
            for side, view in zip(("left", "right"), views, strict=True):
                if view.image is not None:
                    image_key = f"{side}_image"
                    payloads[image_key] = view.payload
                    if self.generate_graphics:
                        stride = max(1, (count + 127) // 128)
                        drawings[f"{side.title()} correspondences"] = GraphicDrawing(
                            tuple(
                                graphic_object(
                                    "marker", [view.xy[i]], ACCEPTED if kept[i] else REJECTED
                                )
                                for i in range(0, count, stride)
                            ),
                            image_key,
                        )
            return Result(
                output,
                passed=code is None,
                message=message,
                fail_reasons=[f"relative_pose_{code}"] if code else [],
                payloads=payloads,
                drawings=drawings,
            )

        if count < minimum:
            return finish(
                "insufficient_pairs", "Too few pairs to meet the configured support minimum"
            )
        if len(indices) < minimum:
            return finish("ray_domain", "Too few pairs invert inside both forward lens domains")
        a, b = (view.rays[indices] for view in views)
        if not _normalized_rank(a) or not _normalized_rank(b):
            return finish("degenerate", "The correspondences lack two-dimensional support")
        if self._cancelled():
            return finish("cancelled", "Relative pose estimation cancelled")
        e, mask = cv2.findEssentialMat(
            a,
            b,
            _K,
            method=_METHODS[p["method"]],
            prob=float(p["confidence"]),
            threshold=float(p["normalized_threshold"]),
            maxIters=int(p["iterations"]),
        )
        if e is None or mask is None:
            return finish("no_essential", "OpenCV could not find an essential matrix")
        if (
            e.ndim != 2
            or e.shape[1] != 3
            or not 3 <= e.shape[0] <= 30
            or e.shape[0] % 3
            or not np.isfinite(e).all()
        ):
            raise ValueError("OpenCV returned malformed essential matrices")
        proposal = mask.ravel().astype(bool)
        if proposal.shape != (len(indices),):
            raise ValueError("OpenCV returned a malformed proposal mask")
        report["native_inlier_indices"] = indices[proposal].tolist()
        if int(proposal.sum()) < 8:
            return finish(
                "insufficient_proposal", "The essential matrix has too little native support"
            )
        recovered = [_recover(part, proposal, views, indices, p) for part in e.reshape(-1, 3, 3)]
        fit = max(recovered, key=lambda candidate: candidate.score)
        if p["refit"] and fit.score[0] >= 8 and not self._cancelled():
            refined = _eight_point_refit(views, indices[fit.kept])
            if refined is not None:
                candidate = _recover(refined, fit.kept, views, indices, p)
                if candidate.score > fit.score:
                    fit = candidate
                    report["refit_adopted"] = True
        if self._cancelled():
            return finish("cancelled", "Relative pose estimation cancelled")

        kept = indices[fit.kept]
        support = len(kept)
        output["inlier_indices"] = kept.tolist()
        report.update(
            cheirality_count=fit.cheirality,
            parallax_count=fit.parallax_count,
            inlier_count=support,
            median_parallax_deg=fit.median_parallax,
            errors_left_px=_finite(fit.errors[0], indices, count),
            errors_right_px=_finite(fit.errors[1], indices, count),
        )
        if support < minimum:
            if fit.cheirality < minimum:
                return finish(
                    "cheirality",
                    "Too few pairs triangulate in front of both cameras within depth limits",
                )
            if fit.parallax_count < minimum:
                return finish(
                    "parallax", "The matches do not have enough parallax for a supported baseline"
                )
            return finish(
                "reprojection", "Too few inliers pass the original pixel reprojection limits"
            )

        baseline = float(p["baseline_length"])
        scale = baseline or 1.0
        units = p["baseline_units"] if baseline else "unitless"
        status = "measured_baseline" if baseline else "unknown"
        right_center = -fit.rotation.T @ fit.direction
        report.update(
            translation_direction_in_right=fit.direction.tolist(),
            right_center_direction_in_left=right_center.tolist(),
            scale=status,
            units=units,
            baseline_length=baseline or None,
            rms_left_px=float(np.sqrt(np.mean(fit.errors[0][fit.kept] ** 2))),
            rms_right_px=float(np.sqrt(np.mean(fit.errors[1][fit.kept] ** 2))),
        )
        output.update(
            pose=pose_record(
                pose_matrix(fit.rotation, scale * fit.direction), first, second, units, scale=status
            ),
            essential=fit.essential.tolist(),
        )
        if "cloud" in self.exposed_outputs:
            output["cloud"] = PointCloud(
                fit.xyz[fit.kept] * scale,
                units,
                first,
                attributes={"source_index": kept},
            )
        return finish(None, f"{support}/{count} pairs · {fit.median_parallax:.3g}° median parallax")
