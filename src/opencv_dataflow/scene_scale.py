"""Recover metric scale and world registration from reconstructed landmark IDs."""

import cv2
import numpy as np

from .calibration_revisions import REVISION
from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, pose_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud
from .sparse_multiview import MAX_LANDMARKS, MAX_VIEWS, _identifier, _parse_view
from .tickets import ticket_key

_MAX_XYZ = float(np.finfo(np.float32).max)


def _control_points(records, label, unit_scale):
    if not isinstance(records, (list, tuple)) or len(records) > MAX_LANDMARKS:
        raise ValidationError(f"{label} must contain up to {MAX_LANDMARKS} landmark records")
    found = {}
    for record in records:
        if not isinstance(record, dict) or not _identifier(record.get("id")):
            raise ValidationError(f"Each {label.lower()} record needs a landmark ID")
        key = record["id"]
        if key in found:
            raise ValidationError(f"{label} landmark IDs must be unique")
        try:
            point = np.asarray(record.get("xyz"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"{label} coordinates must be numeric XYZ") from exc
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValidationError(f"{label} coordinates must be finite XYZ")
        point = point * unit_scale
        if np.any(np.abs(point) > _MAX_XYZ):
            raise ValidationError(f"{label} positions exceed the metric point-cloud range")
        found[key] = point
    return found


def _controls(value):
    if (
        not isinstance(value, dict)
        or value.get("model") != "metric_landmark_controls_v1"
        or not _identifier(value.get("frame"))
        or not isinstance(value["frame"], str)
        or value.get("units") not in ("m", "mm")
    ):
        raise ValidationError("Provide metric landmark controls with a named frame and units")
    scale = 1.0 if value["units"] == "m" else 0.001
    anchors = _control_points(value.get("anchors"), "Anchors", scale)
    holdouts = _control_points(value.get("holdouts", []), "Holdouts", scale)
    if len(anchors) < 3 or len(anchors) + len(holdouts) > MAX_LANDMARKS:
        raise ValidationError("Use at least three anchors and at most 512 total controls")
    if anchors.keys() & holdouts.keys():
        raise ValidationError("Anchor and holdout landmark IDs must be disjoint")
    return value["frame"], anchors, holdouts


def _centered(points):
    center = points.mean(axis=0)
    centered = points - center
    span = np.max(np.abs(centered))
    if not span:
        return None
    return centered / span, center, span


def _estimate(source, target):
    a, b = _centered(source), _centered(target)
    if a is None or b is None:
        return None
    src, src_center, src_span = a
    dst, dst_center, dst_span = b
    # Noncollinear individual sets can still have a degenerate cross-covariance.
    singular = np.linalg.svd(dst.T @ src, compute_uv=False)
    if singular[1] <= 1e-7 * singular[0]:
        return None
    # Select the native uniformly scaled, proper-rotation overload.
    affine, local_scale = cv2.estimateAffine3D(src, dst, force_rotation=True)
    if affine is None or not np.isfinite(affine).all() or not local_scale > 0:
        return None
    rotation = affine[:, :3]
    scale = float(local_scale * dst_span / src_span)
    translation = dst_center + affine[:, 3] * dst_span - scale * (rotation @ src_center)
    if not np.isfinite(scale) or not scale > 0 or not np.isfinite(translation).all():
        return None
    return scale, rotation, translation


def _transform(points, fit):
    scale, rotation, translation = fit
    return scale * (points @ rotation.T) + translation


def _residual(source, target, fit):
    return np.linalg.norm(_transform(source, fit) - target, axis=1)


def _fit_anchors(source, target, tolerance, trials, seed):
    if len(source) < 3:
        return None, np.zeros(len(source), bool)
    rng = np.random.default_rng(seed)
    best, best_score, model = np.zeros(len(source), bool), (0, -np.inf), None
    for trial in range(trials + 1):
        subset = slice(None) if not trial else rng.choice(len(source), size=3, replace=False)
        fit = _estimate(source[subset], target[subset])
        if fit is None:
            continue
        error = _residual(source, target, fit)
        accepted = error <= tolerance
        count = int(accepted.sum())
        score = (count, -float(np.sum(error[accepted] ** 2)))
        if count >= 3 and score > best_score:
            best, best_score, model = accepted, score, fit
        if count == len(source) or len(source) == 3:
            break
    if model is None:
        return None, best
    for _ in range(4):
        fit = _estimate(source[best], target[best])
        if fit is None:
            return None, np.zeros(len(source), bool)
        error = _residual(source, target, fit)
        accepted = error <= tolerance
        count = int(accepted.sum())
        if count < 3:
            break
        if np.array_equal(accepted, best):
            return fit, accepted
        score = (count, -float(np.sum(error[accepted] ** 2)))
        if score <= best_score:
            break
        best, best_score, model = accepted, score, fit
    return model, best


def _views(value, source_frame, target_frame):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not 3 <= len(value) <= MAX_VIEWS:
        raise ValidationError("Supply three to sixteen reconstructed views")
    views = [_parse_view(record) for record in value]
    if len({view.record["id"] for view in views}) != len(views):
        raise ValidationError("Reconstructed view IDs must be unique")
    for view in views:
        camera = view.record["camera"]
        if view.frame != source_frame or view.units != "unitless":
            raise ValidationError("Views must use the reconstructed cloud's unitless frame")
        if target_frame == camera.get("frame", "camera"):
            raise ValidationError("Metric world frame must differ from each optical frame")
        if REVISION in camera:
            raise ValidationError("A sealed camera profile cannot be relabeled with metric units")
    return views


def _convert_views(views, fit, metric_frame):
    if views is None:
        return NO_VALUE
    scale, alignment, shift = fit
    result = []
    for view in views:
        record = view.record
        camera = record["camera"]
        rotation = view.transform[:3, :3] @ alignment.T
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = scale * view.transform[:3, 3] - rotation @ shift
        result.append(
            {
                **record,
                "camera": {**camera, "units": "m"},
                "world_to_camera": pose_record(
                    matrix, metric_frame, camera.get("frame", "camera"), "m"
                ),
            }
        )
    return result


def _control_rows(controls, landmarks, fit, accepted):
    rows, errors = [], []
    for key, surveyed in controls.items():
        row = {"id": key, "reference_xyz_m": surveyed.tolist(), "error_m": None}
        if key not in landmarks:
            row["status"] = "missing"
        elif fit is None:
            row["status"] = "not_fitted"
        else:
            position = _transform(landmarks[key], fit)
            error = float(np.linalg.norm(position - surveyed))
            matched = accepted is None or key in accepted
            if matched:
                errors.append(error)
            row.update(
                reconstructed_xyz_m=position.tolist(),
                error_m=error,
                status="evaluated" if accepted is None else "inlier" if matched else "outlier",
            )
        rows.append(row)
    rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    return rows, rms, max(errors, default=None)


class RecoverSceneScale(Tool3D):
    type_id, title, subtype = (
        "3d.recover_scene_scale",
        "Recover metric scene scale",
        "Reconstruction",
    )
    inputs = (
        Port("cloud", "cloud"),
        Port("views", "data", required=False, required_when_connected=True),
        Port("controls", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("cloud", "cloud", required=False),
        Port("views", "data", required=False),
        Port("similarity", "data", required=False),
        Port("scale_m_per_unit", "number", required=False),
        Port("report", "data"),
    )
    defaults = {
        "controls": {},
        "anchor_tolerance_m": 0.005,
        "minimum_anchor_inlier_ratio": 0.65,
        "maximum_holdout_error_m": 0.01,
        "maximum_trials": 192,
        "random_seed": 0,
    }
    parameter_ranges = {
        "anchor_tolerance_m": NumericRange(1e-7, 1000),
        "minimum_anchor_inlier_ratio": NumericRange(0, 1),
        "maximum_holdout_error_m": NumericRange(1e-7, 1000),
        "maximum_trials": NumericRange(1, 1024, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "controls": "Metric frame, units, anchors and optional disjoint holdouts.",
        "anchor_tolerance_m": "Maximum metric residual for a surveyed training landmark to agree.",
        "minimum_anchor_inlier_ratio": "Minimum share of declared anchors accepted by the fit.",
        "maximum_holdout_error_m": "Largest allowed error at any independent holdout landmark.",
        "maximum_trials": "Maximum random three-anchor fits after trying all matched anchors.",
        "random_seed": "Seed used for reproducible sampling of anchor triplets.",
    }
    input_help = {
        "cloud": "Unitless reconstructed cloud with a unique landmark ID for each point.",
        "views": "Optional reconstructed views in the same unitless frame. Re-emitted in meters.",
        "controls": "Wired control record. Takes precedence over the Controls parameter.",
    }
    payload_reference = "cloud"
    flow_primary = ("cloud", "similarity")
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["controls"], dict):
            raise ValidationError("Controls must be a metric landmark object", "controls")
        if p["controls"]:
            _controls(p["controls"])

    def process(self, inputs):
        p = self.parameters
        cloud = inputs["cloud"]
        ids = cloud.metadata.get("landmark_ids")
        if (
            cloud.units != "unitless"
            or not isinstance(ids, (list, tuple))
            or not 1 <= len(ids) == len(cloud.points) <= MAX_LANDMARKS
            or not all(_identifier(key) for key in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValidationError("Supply a unitless cloud with up to 512 unique landmark IDs")
        target_frame, anchors, holdouts = _controls(inputs.get("controls", p["controls"]))
        views = _views(inputs.get("views"), cloud.frame, target_frame)
        tickets = {ticket_key(v["ticket"]) for v in self.input_payloads.values() if "ticket" in v}
        if len(tickets) > 1:
            raise ValidationError("Metric scale inputs must refer to the same ticket")

        landmarks = dict(zip(ids, cloud.points.astype(np.float64), strict=True))
        matched = [key for key in anchors if key in landmarks]
        fit, inliers = _fit_anchors(
            np.asarray([landmarks[key] for key in matched]),
            np.asarray([anchors[key] for key in matched]),
            float(p["anchor_tolerance_m"]),
            int(p["maximum_trials"]),
            int(p["random_seed"]),
        )
        accepted = {key for key, inlier in zip(matched, inliers, strict=True) if inlier}
        anchor_rows, rms, _ = _control_rows(anchors, landmarks, fit, accepted)
        holdout_rows, holdout_rms, holdout_max = _control_rows(holdouts, landmarks, fit, None)
        ratio = len(accepted) / len(anchors)
        missing = sum(key not in landmarks for key in holdouts)
        fail = []
        if fit is None:
            fail.append("scene_scale_no_supported_fit")
        elif ratio < float(p["minimum_anchor_inlier_ratio"]):
            fail.append("scene_scale_anchor_support_below_minimum")
        if missing:
            fail.append("scene_scale_holdout_missing")
        if holdout_max is not None and holdout_max > float(p["maximum_holdout_error_m"]):
            fail.append("scene_scale_holdout_error_above_maximum")

        report = {
            "model": "scene_scale_report_v1",
            "source_frame": cloud.frame,
            "metric_frame": target_frame,
            "declared_anchors": len(anchors),
            "matched_anchors": len(matched),
            "inlier_anchors": len(accepted),
            "anchor_inlier_ratio": ratio,
            "anchor_inlier_rms_m": rms,
            "holdout_count": len(holdouts),
            "missing_holdouts": missing,
            "holdout_rms_m": holdout_rms,
            "holdout_max_error_m": holdout_max,
            "anchors": anchor_rows,
            "holdouts": holdout_rows,
        }
        outputs = {name: NO_VALUE for name in ("cloud", "views", "similarity", "scale_m_per_unit")}
        graphics = {}
        if not fail:
            scale, rotation, translation = fit
            metric_points = _transform(cloud.points.astype(np.float64), fit)
            if not np.isfinite(metric_points).all() or np.any(np.abs(metric_points) > _MAX_XYZ):
                raise ValidationError("Recovered metric points exceed the point-cloud range")
            similarity = {
                "model": "scene_similarity_v1",
                "source_frame": cloud.frame,
                "target_frame": target_frame,
                "scale_m_per_unit": scale,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
            }
            metric = PointCloud(
                metric_points,
                "m",
                target_frame,
                cloud.colors.copy() if cloud.colors is not None else None,
                cloud.normals @ rotation.T if cloud.normals is not None else None,
                cloud.pixels.copy() if cloud.pixels is not None else None,
                cloud.source_size,
                {**cloud.metadata, "scene_similarity": similarity},
            )
            outputs.update(
                cloud=metric,
                views=_convert_views(views, fit, target_frame),
                similarity=similarity,
                scale_m_per_unit=scale,
            )
            graphics = cloud_graphics(self, metric)
        message = f"{len(accepted)}/{len(anchors)} inlier anchors · {len(holdouts)} holdouts"
        if fail:
            reason = fail[0].removeprefix("scene_scale_").replace("_", " ")
            message = f"Metric result withheld. {reason.capitalize()}. {message}"
        return Result(
            {**outputs, "report": report},
            graphics=graphics,
            passed=not fail,
            score=ratio,
            fail_reasons=fail,
            message=message,
        )
