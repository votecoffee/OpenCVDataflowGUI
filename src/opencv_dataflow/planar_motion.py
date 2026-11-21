"""Recover calibrated relative camera motion from one observed plane."""

import cv2
import numpy as np

from .calibration_revisions import REVISION
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, pose_matrix, pose_record, require_pinhole
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, require_same_frame
from .result_graphics import ACCEPTED, REJECTED

_METHODS = {"direct": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "usac_magsac": cv2.USAC_MAGSAC}
_MAX_POINTS = 16_384
_DEPTH_EPS = 1e-9


def _normal(values, field):
    raw = np.asarray(values, np.float64)
    if raw.shape != (3,) or not np.isfinite(raw).all() or not np.any(raw):
        raise ValidationError("Use a finite nonzero plane normal with three components", field)
    scaled = raw / np.abs(raw).max()
    return scaled / np.linalg.norm(scaled)


def _plane_anchor(record, source_frame, settings):
    if record is None:
        supplied = settings["source_normal"]
        normal = _normal(supplied, "source_normal") if supplied != [] else None
        distance = float(settings["plane_distance_m"])
        return normal, distance or None
    if settings["source_normal"] != [] or settings["plane_distance_m"]:
        raise ValidationError(
            "Use either the connected plane or the configured plane prior", "source_plane"
        )
    if (
        not isinstance(record, dict)
        or record.get("frame") != source_frame
        or record.get("units") not in METERS
    ):
        raise ValidationError(
            "Plane must use the source camera frame and metric units", "source_plane"
        )
    if record.get("model") == "camera_depth_plane_v1":
        values, offset = record.get("normal"), record.get("offset")
    elif record.get("schema") == "point_cloud_model_v1" and record.get("kind") == "plane":
        coefficients = np.asarray(record.get("coefficients"), np.float64)
        if coefficients.shape != (4,):
            raise ValidationError("The fitted cloud plane needs four coefficients", "source_plane")
        values, offset = coefficients[:3], coefficients[3]
    else:
        raise ValidationError("Connect a fitted camera-depth or point-cloud plane", "source_plane")
    normal = _normal(values, "source_plane")
    raw = np.asarray(values, np.float64)
    scale = np.abs(raw).max()
    distance = -float(offset) / scale / np.linalg.norm(raw / scale) * METERS[record["units"]]
    if not np.isfinite(distance) or abs(distance) <= 1e-9:
        raise ValidationError(
            "Plane needs a finite nonzero distance from the camera", "source_plane"
        )
    return (normal if distance > 0 else -normal), abs(distance)


def _pairs(inputs):
    if "matches" in inputs:
        if "source_points" in inputs or "target_points" in inputs:
            raise ValidationError("Use a match record or two point inputs", "matches")
        match = inputs["matches"]
        if not isinstance(match, dict):
            raise ValidationError("Connect a query/train match record", "matches")
        values = [match.get("query_points"), match.get("train_points")]
        sizes = [match.get("query_size"), match.get("train_size")]
    else:
        if not all(key in inputs for key in ("source_points", "target_points")):
            raise ValidationError("Connect both source and target point lists or a match record")
        values = [inputs["source_points"], inputs["target_points"]]
        sizes = [None, None]
    if any(
        not isinstance(value, (list, tuple, np.ndarray))
        or isinstance(value, np.ndarray)
        and value.ndim != 2
        or len(value) > _MAX_POINTS
        for value in values
    ):
        raise ValidationError(f"Provide two ordered point lists of at most {_MAX_POINTS} rows")
    arrays = [
        np.asarray(value, np.float64).reshape(0, 2)
        if len(value) == 0
        else np.asarray(value, np.float64)
        for value in values
    ]
    if arrays[0].shape != arrays[1].shape or any(
        a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() for a in arrays
    ):
        raise ValidationError("Correspondences need two matching finite Nx2 pixel arrays")
    return arrays, sizes


def _view(block, inputs, side, xy, match_size):
    model = inputs[f"{side}_camera"]
    k, d, size = require_pinhole(model)
    label = model.get("frame")
    if not isinstance(label, str) or not label.strip():
        raise ValidationError("Both camera models need explicit frame names")
    image = inputs.get(f"{side}_image")
    context = block.input_payloads.get(f"{side}_points", {})
    if match_size is not None and (
        not isinstance(match_size, (list, tuple)) or size != tuple(match_size)
    ):
        raise ValidationError(f"{side.title()} matches use a different calibrated size")
    check_spatial(context, size)
    if image is not None:
        camera_for_image(model, image)
        optical = block.input_payloads[f"{side}_image"].get("optical_frame")
        if optical is not None and optical != label:
            raise ValidationError(f"{side.title()} image uses a different optical frame")
        require_same_frame(
            (image, ValuePacket(xy, context)),
            message=f"{side.title()} image and points use different frames",
        )
    elif REVISION in model:
        raise ValidationError(f"Connect the {side} image to verify the calibration revision")
    if (xy < 0).any() or (xy >= size).any():
        raise ValidationError(f"{side.title()} points must lie inside the calibrated raster")
    if not len(xy):
        return (k, d, label, size), image, xy, xy
    normalized = cv2.undistortPoints(
        xy[:, None], k, d, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
    )[:, 0]
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 1e5):
        raise ValidationError(f"{side.title()} undistortion produced unusable rays")
    ideal = normalized * k.diagonal()[:2] + k[:2, 2]
    return (k, d, label, size), image, normalized, ideal


def _spans_plane(points):
    if len(np.unique(points, axis=0)) < 4:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return singular[1] > singular[0] * 1e-6


def _fit_homography(source, target, optics, settings):
    method = _METHODS[settings["method"]]
    h, mask = cv2.findHomography(
        source,
        target,
        method,
        float(settings["ransac_pixels"]),
        maxIters=int(settings["max_iterations"]),
        confidence=float(settings["confidence"]),
    )
    if h is None or mask is None or not np.isfinite(h).all():
        return None, None
    k1, k2 = optics[0][0], optics[1][0]
    # Native decomposition takes one K. Work in each camera's normalized rays.
    normalized = np.linalg.solve(k2, h @ k1)
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] <= 1e-10 * singular[0]:
        return None, None
    normalized /= singular[1]
    if np.linalg.det(normalized) < 0:
        normalized = -normalized
    return normalized, mask.reshape(-1).astype(bool)


def _native_candidates(h, source_rays, target_rays, inliers):
    count, rotations, translations, normals = cv2.decomposeHomographyMat(h, np.eye(3))
    if count not in (1, 2, 4) or any(
        len(values) != count for values in (rotations, translations, normals)
    ):
        raise ValueError("OpenCV returned unexpected homography candidates")
    rotation_only = count == 1 and not np.any(translations[0]) and not np.any(normals[0])
    if rotation_only:
        return rotations, translations, normals, {0}, True
    visible = cv2.filterHomographyDecompByVisibleRefpoints(
        rotations,
        normals,
        np.ascontiguousarray(source_rays[:, None], np.float32),
        np.ascontiguousarray(target_rays[:, None], np.float32),
        pointsMask=np.ascontiguousarray(inliers[:, None], np.uint8),
    )
    indices = set() if visible is None else set(visible.ravel().tolist())
    if any(not 0 <= i < count for i in indices):
        raise ValueError("OpenCV returned an invalid visibility index")
    return rotations, translations, normals, indices, False


def _physical_candidate(
    rotation,
    tau,
    normal,
    rays,
    observed,
    ideal_target,
    native_ok,
    rotation_only,
    anchor,
    distance,
    frames,
    settings,
):
    info = {"native_visible": native_ok}
    rotation, tau = np.asarray(rotation, np.float64), np.asarray(tau, np.float64).reshape(3)
    if rotation_only and np.isfinite(rotation).all():
        left, _, right = np.linalg.svd(rotation)
        rotation = left @ right
    try:
        pose_matrix(rotation, tau)
    except ValueError:
        return {**info, "status": "invalid_rigid_motion"}, None
    target_normal = None
    if rotation_only:
        transformed = rays @ rotation.T
        normal = None
    else:
        raw_normal = np.asarray(normal, np.float64).reshape(3)
        scale = np.linalg.norm(raw_normal)
        if not np.isfinite(scale) or scale <= 1e-12:
            return {**info, "status": "invalid_normal"}, None
        normal, tau = raw_normal / scale, tau * scale
        denominators = rays @ normal
        info["minimum_source_plane_dot"] = float(denominators.min())
        if np.any(denominators <= _DEPTH_EPS):
            return {**info, "status": "nonpositive_source_depth"}, None
        source_xyz = rays / denominators[:, None]
        transformed = source_xyz @ rotation.T + tau
        target_normal = rotation @ normal
    depth = float(transformed[:, 2].min())
    info["minimum_target_depth_over_distance"] = None if rotation_only else depth
    if not np.isfinite(transformed).all() or depth <= _DEPTH_EPS:
        return {**info, "status": "nonpositive_target_depth"}, None
    if not native_ok:
        return {**info, "status": "native_visibility"}, None
    with np.errstate(over="ignore", invalid="ignore"):
        ideal = (transformed[:, :2] / transformed[:, 2:3]) * ideal_target.diagonal()[:2]
        ideal += ideal_target[:2, 2]
        error = np.linalg.norm(ideal - observed, axis=1)
        rms = float(np.sqrt(np.mean(error**2)))
    if not np.isfinite(rms):
        return {**info, "status": "nonfinite_reprojection"}, None
    info.update(rms_px=rms, maximum_error_px=float(error.max()))
    if rms > float(settings["maximum_rms_px"]):
        return {**info, "status": "reprojection_limit"}, None
    if anchor is not None and normal is not None:
        angle = float(np.degrees(np.arccos(np.clip(normal @ anchor, -1, 1))))
        info["source_normal_error_deg"] = angle
        if angle > float(settings["maximum_normal_angle_deg"]):
            return {**info, "status": "plane_normal_limit"}, None
    metric_pose = None
    if distance is not None and normal is not None:
        metric_pose = pose_record(pose_matrix(rotation, tau * distance), *frames, units="m")
    return {**info, "status": "accepted"}, {
        "model": "planar_camera_motion_hypothesis_v1",
        "source_frame": frames[0],
        "target_frame": frames[1],
        "rotation": rotation.tolist(),
        "translation_over_plane_distance": None if rotation_only else tau.tolist(),
        "normal_source": None if normal is None else normal.tolist(),
        "normal_target": None if target_normal is None else target_normal.tolist(),
        "source_plane_distance_m": None if normal is None else distance,
        "metric_pose": metric_pose,
        "rms_pixels": rms,
    }


def _draw_matches(xy, keep, limit, port):
    selected = range(min(limit, len(keep)))
    objects = tuple(
        graphic_object("marker", [xy[index]], ACCEPTED if keep[index] else REJECTED)
        for index in selected
    )
    return GraphicDrawing(objects, port, from_output=True) if objects else None


class RecoverPlanarCameraMotion(Tool3D):
    type_id = "3d.planar_camera_motion"
    title, subtype = "Recover planar camera motion", "Camera models"
    description = "Recover visible camera motion and plane alternatives from paired image pixels."
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (
        Port("source_camera", "data"),
        Port("target_camera", "data"),
        Port("matches", "data", required=False, required_when_connected=True),
        Port("source_points", "data", required=False, required_when_connected=True),
        Port("target_points", "data", required=False, required_when_connected=True),
        Port("source_plane", "data", required=False, required_when_connected=True),
        Port("source_image", required=False, required_when_connected=True),
        Port("target_image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("hypotheses", "data"),
        Port("ambiguous", "boolean"),
        Port("inliers", "data"),
        Port("source_image", required=False),
        Port("target_image", required=False),
        Port("report", "data"),
    )
    defaults = {
        "method": "ransac",
        "ransac_pixels": 2.0,
        "max_iterations": 2000,
        "confidence": 0.995,
        "minimum_inliers": 8,
        "minimum_inlier_fraction": 0.55,
        "maximum_rms_px": 2.0,
        "plane_distance_m": 0.0,
        "source_normal": [],
        "maximum_normal_angle_deg": 8.0,
        "maximum_drawn_matches": 96,
    }
    parameter_choices = {"method": tuple(_METHODS)}
    parameter_ranges = {
        "ransac_pixels": NumericRange(1e-6, 1024),
        "max_iterations": NumericRange(1, 20_000, whole=True),
        "confidence": NumericRange(1e-6, 0.999999),
        "minimum_inliers": NumericRange(4, _MAX_POINTS, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "maximum_rms_px": NumericRange(0, 1024),
        "plane_distance_m": NumericRange(0, 1e6),
        "maximum_normal_angle_deg": NumericRange(0, 180),
        "maximum_drawn_matches": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "method": "Homography estimator on undistorted ideal pixels.",
        "ransac_pixels": "Homography inlier tolerance in ideal target pixels.",
        "max_iterations": "Maximum iterations for a robust homography fit.",
        "confidence": "Target confidence for robust homography estimation.",
        "minimum_inliers": "Required number of matched points supporting the homography.",
        "minimum_inlier_fraction": "Required fraction of matched points supporting the homography.",
        "maximum_rms_px": "Maximum candidate reprojection RMS in ideal target pixels.",
        "plane_distance_m": "Perpendicular source camera-to-plane distance. Zero is unknown.",
        "source_normal": "Normal in source camera axes directed towards the plane.",
        "maximum_normal_angle_deg": "Maximum angular discrepancy from the supplied plane normal.",
        "maximum_drawn_matches": "Limit inlier and outlier markers on the attached images.",
    }
    input_help = {
        "source_camera": "Source pinhole intrinsics, distortion and optical frame.",
        "target_camera": "Target pinhole intrinsics, distortion and optical frame.",
        "matches": "Matched query/source and train/target pixels from a feature matcher.",
        "source_points": "Alternative ordered source image pixels, with target points.",
        "target_points": "Corresponding target image pixels in the same row order.",
        "source_plane": "Optional metric depth-plane or cloud-plane record in the source camera.",
        "source_image": "Optional source image for markers and calibration revision checks.",
        "target_image": "Optional target image for markers and calibration revision checks.",
    }
    output_help = {
        "pose": "A source-to-target pose in meters when exactly one anchored candidate survives.",
        "hypotheses": "All accepted rotations, plane normals and translations divided by distance.",
        "ambiguous": "True when multiple plausible motions remain.",
        "inliers": "Homography fit mask in incoming correspondence order.",
        "report": "Homography, visibility tests, reprojection, normal checks and scale status.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        settings = self.parameters if parameters is None else parameters
        if settings["source_normal"] != []:
            _normal(settings["source_normal"], "source_normal")

    def process(self, inputs):
        settings = self.parameters
        raw_points, sizes = _pairs(inputs)
        optics, images, normalized, ideal = zip(
            *(
                _view(self, inputs, side, xy, size)
                for side, xy, size in zip(("source", "target"), raw_points, sizes, strict=True)
            ),
            strict=True,
        )
        frames = tuple(view[2] for view in optics)
        if frames[0] == frames[1]:
            raise ValidationError("Use distinct source and target optical frame names")
        anchor, distance = _plane_anchor(inputs.get("source_plane"), frames[0], settings)
        n = len(raw_points[0])
        keep = np.zeros(n, bool)
        report = {
            "model": "planar_camera_motion_report_v1",
            "source_frame": frames[0],
            "target_frame": frames[1],
            "pairs": n,
            "fit_inliers": 0,
            "fit_inlier_fraction": 0.0,
            "native_candidates": 0,
            "plausible_count": 0,
            "candidates": [],
            "scale_status": "unknown" if distance is None else "measured_plane_distance",
            "source_plane_distance_m": distance,
            "status": "failed",
        }
        outputs = {
            "pose": NO_VALUE,
            "hypotheses": [],
            "ambiguous": False,
            "inliers": keep.tolist(),
            "source_image": NO_VALUE if images[0] is None else images[0],
            "target_image": NO_VALUE if images[1] is None else images[1],
            "report": report,
        }
        payloads = {port: {} for port in ("pose", "hypotheses", "ambiguous", "inliers", "report")}
        for name in ("source_image", "target_image"):
            if name in inputs:
                payloads[name] = self.input_payloads[name]

        def result(reason, message):
            drawings = {}
            if self.generate_graphics:
                for port, image, xy in zip(
                    ("source_image", "target_image"), images, raw_points, strict=True
                ):
                    if image is not None:
                        drawing = _draw_matches(
                            xy, keep, int(settings["maximum_drawn_matches"]), port
                        )
                        if drawing is not None:
                            drawings[port.replace("_", " ").title() + " matches"] = drawing
            return Result(
                outputs,
                payloads=payloads,
                drawings=drawings,
                passed=reason is None,
                fail_reasons=[reason] if reason else [],
                score=report["fit_inlier_fraction"],
                message=message,
            )

        if n < int(settings["minimum_inliers"]) or not all(
            _spans_plane(a) for a in (*raw_points, *ideal)
        ):
            return result("planar_motion_support", "Too few distinct, noncollinear matches")
        h, inliers = _fit_homography(*ideal, optics, settings)
        if h is None:
            return result("planar_motion_fit", "No stable calibrated homography")
        keep = inliers
        outputs["inliers"] = keep.tolist()
        count = int(keep.sum())
        report.update(
            fit_inliers=count, fit_inlier_fraction=count / n, normalized_homography=h.tolist()
        )
        if (
            count < int(settings["minimum_inliers"])
            or count / n < settings["minimum_inlier_fraction"]
        ):
            return result("planar_motion_support", f"Only {count}/{n} matches support the plane")
        if not all(_spans_plane(a[keep]) for a in (*raw_points, *ideal)):
            return result(
                "planar_motion_support", "Inlier points do not span a plane in both images"
            )
        if self._cancelled():
            return result("planar_motion_cancelled", "Planar motion recovery cancelled")

        rs, ts, ns, native_indices, rotation_only = _native_candidates(h, *normalized, keep)
        report.update(native_candidates=len(rs), native_visible_indices=sorted(native_indices))
        if rotation_only:
            report["scale_status"] = "rotation_only_model"
        rays = np.column_stack((normalized[0][keep], np.ones(count)))
        hypotheses = []
        for index, (rotation, tau, normal) in enumerate(zip(rs, ts, ns, strict=True)):
            if self._cancelled():
                return result("planar_motion_cancelled", "Planar motion recovery cancelled")
            info, candidate = _physical_candidate(
                rotation,
                tau,
                normal,
                rays,
                ideal[1][keep],
                optics[1][0],
                index in native_indices,
                rotation_only,
                anchor,
                distance,
                frames,
                settings,
            )
            report["candidates"].append({"solver_index": index, **info})
            if candidate is not None:
                hypotheses.append({"solver_index": index, **candidate})
        hypotheses.sort(key=lambda item: (item["rms_pixels"], item["solver_index"]))
        count = len(hypotheses)
        report["plausible_count"] = count
        if not count:
            return result(
                "planar_motion_no_candidate", "No motion passed visibility and plane checks"
            )
        outputs.update(hypotheses=hypotheses, ambiguous=count > 1)
        if count == 1 and hypotheses[0]["metric_pose"] is not None:
            outputs["pose"] = hypotheses[0]["metric_pose"]
        report["status"] = "ok"
        if rotation_only:
            message = "Rotation-only homography model"
        elif count > 1:
            message = f"{count} plausible source-to-target motions"
        else:
            message = (
                "One metric motion" if outputs["pose"] is not NO_VALUE else "One unscaled motion"
            )
        return result(None, message)
