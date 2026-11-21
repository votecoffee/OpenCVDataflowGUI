"""Object pose for fisheye and forward-domain omnidirectional cameras."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, check_capture
from .core import Port, Result
from .geometry3d import PNP
from .geometry3d_common import (
    Tool3D,
    camera,
    points,
    pose_matrix,
    pose_record,
    project_camera_pixels,
    rigid,
    undistort_camera_pixels,
)
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .result_graphics import ACCEPTED, REJECTED

_MAX_POINTS = 4096
_ZERO = np.zeros(3)


def _scale_and_rank(rows):
    centered = rows - rows.mean(axis=0)
    scale = np.max(np.abs(centered))
    if not scale:
        return scale, 0
    local = centered / scale
    singular = np.linalg.svd(local, compute_uv=False)
    return scale, int(np.count_nonzero(singular > singular[0] * 1e-7))


def _forward(xyz, minimum_cosine):
    norm = np.linalg.norm(xyz, axis=1)
    return np.isfinite(norm) & (norm > 0) & (xyz[:, 2] >= minimum_cosine * norm)


def _rays(xy, model, k, d, criteria, minimum_cosine, inversion_limit):
    if model["lens"] == "fisheye":
        undeformed = cv2.fisheye.undistortPoints(xy[:, None], k, d, criteria=criteria).reshape(
            -1, 2
        )
    else:
        undeformed = undistort_camera_pixels(xy, model, k, d)
    xyz = np.column_stack((undeformed, np.ones(len(xy))))
    valid = _forward(xyz, minimum_cosine)
    if np.any(valid):
        # Mei's inverse returns x/z and y/z without the hemisphere sign. Forward
        # reprojection rejects rays whose inverse belongs to the opposite chart.
        rebuilt = project_camera_pixels(xyz[valid], model, k, d, _ZERO, _ZERO)
        valid[valid] = np.linalg.norm(rebuilt - xy[valid], axis=1) <= inversion_limit
    return undeformed, valid


def _guess(record, model, source, center, scale):
    if (
        not isinstance(record, dict)
        or record.get("model") != "pose_3d_v1"
        or record.get("source_frame") != source
        or record.get("target_frame") != model["frame"]
        or record.get("units") != model["units"]
    ):
        raise ValidationError(
            "Initial pose must have the requested object frame, camera frame and units"
        )
    transform = rigid(record)
    rotation = transform[:3, :3]
    translation = (transform[:3, 3] + rotation @ center) / scale
    return cv2.Rodrigues(rotation)[0], translation.reshape(3, 1)


def _solve(xyz, pixels, rays, model, k, d, p, criteria, guess):
    fisheye = model["lens"] == "fisheye"
    if fisheye:
        observed, distortion = pixels[:, None], d
    else:
        observed, distortion = rays * k.diagonal()[:2] + k[:2, 2], None
    rv, tv = guess if guess is not None else (None, None)
    options = {"flags": PNP[p["method"]], "useExtrinsicGuess": guess is not None}
    if fisheye:
        options["criteria"] = criteria
    if p["ransac"]:
        solve = cv2.fisheye.solvePnPRansac if fisheye else cv2.solvePnPRansac
        ok, rotation, translation, native_ids = solve(
            xyz,
            observed,
            k,
            distortion,
            rv,
            tv,
            iterationsCount=int(p["iterations"]),
            reprojectionError=float(p["ransac_rectified_px"]),
            confidence=float(p["confidence"]),
            **options,
        )
        native_ids = np.asarray(native_ids).ravel() if native_ids is not None else np.empty(0, int)
        if np.any(native_ids < 0) or np.any(native_ids >= len(xyz)):
            raise ValueError("OpenCV returned invalid wide-angle pose inlier IDs")
    else:
        solve = cv2.fisheye.solvePnP if fisheye else cv2.solvePnP
        ok, rotation, translation = solve(xyz, observed, k, distortion, rv, tv, **options)
        native_ids = np.arange(len(xyz))
    if not ok:
        return None, np.empty(0, int)
    pose = np.concatenate((rotation.ravel(), translation.ravel()))
    return (pose if np.isfinite(pose).all() else None), native_ids


def _score(xyz, measured, valid, model, k, d, pose, p):
    r, t = pose[:3], pose[3:]
    rotation = cv2.Rodrigues(r)[0]
    projected = np.full(measured.shape, np.nan)
    domain = valid & _forward(xyz @ rotation.T + t, float(p["minimum_ray_cosine"]))
    if np.any(domain):
        projected[domain] = project_camera_pixels(xyz[domain], model, k, d, r, t)
    errors = np.linalg.norm(projected - measured, axis=1)
    inliers = np.isfinite(errors)
    if p["ransac"]:
        inliers &= errors <= float(p["inlier_threshold_px"])
    selected = np.flatnonzero(inliers)
    rms = float(np.sqrt(np.mean(errors[selected] ** 2))) if len(selected) else None
    return selected, rms, projected, errors


def _refine(xyz, pixels, model, k, d, seed, steps):
    from scipy.optimize import least_squares

    def residual(pose):
        return (project_camera_pixels(xyz, model, k, d, pose[:3], pose[3:]) - pixels).ravel()

    result = least_squares(residual, seed, method="lm", diff_step=1e-6, max_nfev=steps)
    return result.x if np.isfinite(result.x).all() else seed, result.nfev


class SolveWideAnglePose(Tool3D):
    type_id, title, subtype = "3d.wide_pnp", "Wide-angle object pose (PnP)", "Camera models"
    description = (
        "Recover a pose and check residuals in the source fisheye or omnidirectional image."
    )
    inputs = (
        Port("object_points", "data"),
        Port("image_points", "data"),
        Port("camera", "data"),
        Port(
            "initial_pose", "data", required=False, required_when_connected=True, visual_type="pose"
        ),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("inliers", "data"),
        Port("rms", "number", required=False),
        Port("report", "data"),
        Port("image", required=False),
    )
    defaults = {
        "method": "iterative",
        "ransac": True,
        "iterations": 500,
        "confidence": 0.995,
        "ransac_rectified_px": 8.0,
        "inlier_threshold_px": 2.0,
        "minimum_inliers": 4,
        "minimum_inlier_fraction": 0.5,
        "max_rms_px": 1.0,
        "minimum_ray_cosine": 0.05,
        "max_inversion_error_px": 0.1,
        "undistort_iterations": 50,
        "undistort_epsilon": 1e-10,
        "refine_evaluations": 30,
        "source_frame": "object",
    }
    parameter_choices = {"method": tuple(PNP)}
    parameter_ranges = {
        "iterations": NumericRange(1, 10_000, whole=True),
        "confidence": NumericRange(0.5001, 0.9999),
        "ransac_rectified_px": NumericRange(1e-4, 10_000),
        "inlier_threshold_px": NumericRange(1e-4, 1000),
        "minimum_inliers": NumericRange(4, _MAX_POINTS, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "max_rms_px": NumericRange(1e-6, 1000),
        "minimum_ray_cosine": NumericRange(0.001, 0.999),
        "max_inversion_error_px": NumericRange(1e-6, 10),
        "undistort_iterations": NumericRange(1, 200, whole=True),
        "undistort_epsilon": NumericRange(1e-15, 0.1),
        "refine_evaluations": NumericRange(0, 200, whole=True),
    }
    parameter_help = {
        "method": "Native pose algorithm. IPPE needs coplanar points.",
        "ransac": "Propose a robust fit in the rectified grid, then classify in source pixels.",
        "iterations": "Maximum native RANSAC iterations.",
        "confidence": "Native RANSAC confidence, strictly below 1.",
        "ransac_rectified_px": "Proposal threshold in rectified pixels with the same intrinsics.",
        "inlier_threshold_px": "Final inlier tolerance in raw pixels for RANSAC.",
        "minimum_inliers": "Minimum number of retained source correspondences.",
        "minimum_inlier_fraction": "Minimum retained fraction of all supplied pairs.",
        "max_rms_px": "Maximum source-pixel RMS over the retained pairs.",
        "minimum_ray_cosine": "Minimum positive Z of a unit ray, excluding the horizon.",
        "max_inversion_error_px": "Maximum unproject/project round trip error in source pixels.",
        "undistort_iterations": "Fisheye inverse solver iteration limit.",
        "undistort_epsilon": "Fisheye inverse solver convergence tolerance.",
        "refine_evaluations": "Source-lens least-squares evaluation limit. Zero skips refinement.",
        "source_frame": "Coordinate frame of the object points.",
    }
    input_help = {
        "object_points": "Ordered XYZ in the named object frame and camera length units.",
        "image_points": "Matching XY coordinates in the original, distorted image.",
        "camera": "Calibrated fisheye or Mei omnidirectional camera model.",
        "initial_pose": "Optional object-to-camera initial guess for the Iterative method.",
        "image": "Optional original image for source-pixel measurement graphics.",
    }
    output_help = {
        "pose": "Accepted object-to-camera transform in the camera length units.",
        "inliers": "Original row indices rechecked in the source lens model.",
        "rms": "Root mean squared reprojection error in source pixels.",
        "report": "Native proposal indices, source-pixel residuals and diagnostics for all pairs.",
        "image": "Original source image with independent result graphics.",
    }
    payload_reference = "image"
    composes_result_image = True
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_frame"], str) or not p["source_frame"].strip():
            raise ValidationError("Name the object point frame", "source_frame")
        if not isinstance(p["ransac"], bool):
            raise ValidationError("RANSAC must be a boolean", "ransac")

    def process(self, inputs):
        p = self.parameters
        model = inputs["camera"]
        k, d, size = camera(model)
        lens = model.get("lens")
        if lens not in ("fisheye", "omnidir"):
            raise ValidationError("Connect a fisheye or omnidirectional camera")
        if lens == "omnidir":
            xi = float(model["xi"])
            if xi < 0:
                raise ValidationError(
                    "The forward omnidirectional pose path requires nonnegative xi"
                )
            model = {**model, "xi": xi}
        source = p["source_frame"]
        target, units = model.get("frame", "camera"), model.get("units", "m")
        model = {**model, "frame": target, "units": units}
        if source == target:
            raise ValidationError("Object and camera frames must differ", "source_frame")
        xyz, xy = points(inputs["object_points"], 3, 4), points(inputs["image_points"], 2, 4)
        if len(xyz) != len(xy) or len(xyz) > _MAX_POINTS or np.max(np.abs(xyz)) > 1e15:
            raise ValidationError("Use 4 to 4096 corresponding bounded XYZ and XY rows")
        if np.any(xy < 0) or np.any(xy >= size):
            raise ValidationError(
                "Image point coordinates must be inside the calibrated image grid"
            )
        obj_units = self.input_payloads["object_points"].get("object_point_units")
        if obj_units is not None and obj_units != units:
            raise ValidationError("Object point units must match the camera")
        point_payload = self.input_payloads["image_points"]
        image = inputs.get("image")
        capture = self.input_payloads["image"] if image is not None else point_payload
        spatial = point_payload.get("spatial")
        if spatial and spatial["image_size"] != list(size):
            raise ValidationError("Image point grid does not match the calibrated camera")
        if image is not None:
            if image.shape[:2][::-1] != size:
                raise ValidationError("Source image dimensions do not match the camera")
            require_same_frame(
                [image, ValuePacket(xy, point_payload)],
                message="Image and points must share a calibrated coordinate frame",
            )
            if (
                "ticket" in point_payload
                and "ticket" in capture
                and point_payload["ticket"] != capture["ticket"]
            ):
                raise ValidationError("Image and points must describe the same capture")
        check_capture(model.get(REVISION), capture, size)

        report = {
            "model": "wide_angle_pose_report_v1",
            "lens": lens,
            "camera_frame": target,
            "source_frame": source,
            "units": units,
            "image_size": list(size),
            "input_count": len(xyz),
            "method": p["method"],
            "ransac": p["ransac"],
            "native_backend": "fisheye" if lens == "fisheye" else "omnidir_forward_rays",
        }
        output = {
            "pose": NO_VALUE,
            "inliers": [],
            "rms": NO_VALUE,
            "report": report,
            "image": image if image is not None else NO_VALUE,
        }

        def finish(code, message, drawings=None):
            report.update(status=code or "ok", passed=code is None)
            return Result(
                output,
                passed=code is None,
                message=message,
                drawings=drawings or {},
                fail_reasons=[f"wide_pose_{code}"] if code else [],
                payloads={key: point_payload for key in output if key != "image"}
                if image is None
                else {},
            )

        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            int(p["undistort_iterations"]),
            float(p["undistort_epsilon"]),
        )
        rays, valid = _rays(
            xy,
            model,
            k,
            d,
            criteria,
            float(p["minimum_ray_cosine"]),
            float(p["max_inversion_error_px"]),
        )
        raw_ids = np.flatnonzero(valid)
        report["unsupported_ray_indices"] = np.flatnonzero(~valid).tolist()
        report["minimum_ray_cosine"] = float(p["minimum_ray_cosine"])
        report["max_inversion_error_px"] = float(p["max_inversion_error_px"])
        report["undistort_criteria"] = list(criteria)[1:] if lens == "fisheye" else None
        minimum = max(
            int(p["minimum_inliers"]), int(np.ceil(float(p["minimum_inlier_fraction"]) * len(xyz)))
        )
        if len(raw_ids) < minimum or (not p["ransac"] and not valid.all()):
            return finish(
                "ray_domain", "Insufficient observations invert in the forward camera domain"
            )
        scale, rank = _scale_and_rank(xyz[valid])
        _, pixel_rank = _scale_and_rank(rays[valid])
        if min(rank, pixel_rank) < 2:
            return finish("degenerate", "Object or image points do not constrain a pose")
        center = xyz[valid].mean(axis=0)
        local = (xyz - center) / scale
        if p["method"] == "ippe" and rank != 2:
            raise ValidationError("IPPE requires coplanar object points", "method")
        guess = None
        if "initial_pose" in inputs:
            if p["method"] != "iterative":
                raise ValidationError("Initial pose is supported by the Iterative method", "method")
            guess = _guess(inputs["initial_pose"], model, source, center, scale)
        if (
            p["method"] == "iterative"
            and not p["ransac"]
            and rank == 3
            and len(raw_ids) < 6
            and guess is None
        ):
            return finish(
                "degenerate", "Iterative nonplanar pose needs six rows or an initial pose"
            )
        if self._cancelled():
            return finish("cancelled", "Wide-angle pose cancelled")
        seed, native = _solve(local[valid], xy[valid], rays[valid], model, k, d, p, criteria, guess)
        if seed is None:
            return finish("no_pose", "OpenCV did not find a valid pose proposal")
        report["native_inlier_indices"] = raw_ids[native].tolist()
        report["proposal_threshold_rectified_px"] = (
            float(p["ransac_rectified_px"]) if p["ransac"] else None
        )
        report["source_threshold_px"] = float(p["inlier_threshold_px"]) if p["ransac"] else None
        accepted, rms, projected, errors = _score(local, xy, valid, model, k, d, seed, p)
        report["refine_evaluations"] = 0
        report["refinement_adopted"] = False
        if len(accepted) >= 4 and p["refine_evaluations"] and not self._cancelled():
            candidate, evaluations = _refine(
                local[accepted],
                xy[accepted],
                model,
                k,
                d,
                seed,
                int(p["refine_evaluations"]),
            )
            new_ids, new_rms, new_projection, new_errors = _score(
                local, xy, valid, model, k, d, candidate, p
            )
            report["refine_evaluations"] = evaluations
            if len(new_ids) > len(accepted) or (len(new_ids) == len(accepted) and new_rms <= rms):
                seed, accepted, rms, projected, errors = (
                    candidate,
                    new_ids,
                    new_rms,
                    new_projection,
                    new_errors,
                )
                report["refinement_adopted"] = True
        if self._cancelled():
            return finish("cancelled", "Wide-angle pose cancelled")
        report.update(
            inlier_count=len(accepted),
            rms_px=rms,
            projected_pixels=[
                point.tolist() if np.isfinite(point).all() else None for point in projected
            ],
            errors_px=[float(error) if np.isfinite(error) else None for error in errors],
        )
        output.update(inliers=accepted.tolist(), rms=rms if rms is not None else NO_VALUE)
        kept = np.zeros(len(xyz), bool)
        kept[accepted] = True
        drawings = {}
        if image is not None and self.generate_graphics:
            for name, selected, color in (
                ("Accepted points", accepted, ACCEPTED),
                ("Rejected points", np.flatnonzero(~kept), REJECTED),
            ):
                if len(selected):
                    stride = max(1, (len(selected) + 127) // 128)
                    drawings[name] = GraphicDrawing(
                        tuple(
                            graphic_object("marker", [xy[index]], color)
                            for index in selected[::stride]
                        )
                    )
        if len(accepted) < minimum or (not p["ransac"] and len(accepted) != len(xy)):
            return finish(
                "insufficient_support", "Pose has too few forward source-pixel inliers", drawings
            )
        if _scale_and_rank(local[accepted])[1] < 2 or _scale_and_rank(rays[accepted])[1] < 2:
            return finish("degenerate_support", "Retained points do not constrain a pose", drawings)
        if rms > float(p["max_rms_px"]):
            return finish(
                "rms", f"Source reprojection RMS {rms:.4g} px exceeds the limit", drawings
            )
        rotation = cv2.Rodrigues(seed[:3])[0]
        transform = pose_matrix(rotation, seed[3:] * scale - rotation @ center)
        output["pose"] = pose_record(transform, source, target, units, rms_pixels=rms, lens=lens)
        return finish(
            None, f"{len(accepted)}/{len(xy)} inliers · source RMS {rms:.4g} px", drawings
        )
