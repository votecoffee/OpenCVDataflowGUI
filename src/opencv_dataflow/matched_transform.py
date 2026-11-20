"""Robust metric transforms from known 3D correspondence rows."""

from dataclasses import dataclass

import cv2
import numpy as np

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, fit_similarity_3d, pose_matrix, pose_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud

_MAX_MATCHES = 4096
_MINIMUM = {"translation": 4, "rigid": 3, "similarity": 3, "affine": 4}
_REQUIRED_RANK = {"translation": 0, "rigid": 2, "similarity": 2, "affine": 3}
_RANK_TOLERANCE = 1e-7


@dataclass(frozen=True)
class Fit:
    linear: np.ndarray
    translation: np.ndarray
    scale: float = 1.0

    def apply(self, source):
        return source @ self.linear.T + self.translation


def _match_rows(pairs, source_size, target_size):
    if pairs is None:
        if source_size != target_size or source_size > _MAX_MATCHES:
            raise ValidationError(
                "Row alignment needs equal cloud counts up to 4096, or explicit pairs"
            )
        rows = np.arange(source_size, dtype=np.int32)
        return np.column_stack((rows, rows))
    if not isinstance(pairs, (list, tuple)) or len(pairs) > _MAX_MATCHES:
        raise ValidationError("Pairs must contain up to 4096 [source, target] row indices", "pairs")
    if not pairs:
        return np.empty((0, 2), np.int32)
    if any(
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or any(type(value) is not int for value in pair)
        or not 0 <= pair[0] < source_size
        or not 0 <= pair[1] < target_size
        for pair in pairs
    ):
        raise ValidationError("Each pair needs valid zero-based integer row indices", "pairs")
    rows = np.asarray(pairs, dtype=np.int32)
    if any(len(np.unique(column)) != len(rows) for column in rows.T):
        raise ValidationError("Each source and target row can occur in at most one pair", "pairs")
    return rows


def _has_rank(samples, rank):
    if not rank:
        return True
    singular = np.linalg.svd(samples, compute_uv=False)
    return bool(singular[rank - 1] > _RANK_TOLERANCE * singular[0])


def _supports_geometry(source, target, model):
    rank = _REQUIRED_RANK[model]
    return _has_rank(source - source.mean(axis=0), rank) and _has_rank(
        target - target.mean(axis=0), rank
    )


def _refit(source, target, model):
    if model == "translation":
        return Fit(np.eye(3), (target - source).mean(axis=0))
    if model in ("rigid", "similarity"):
        estimated = fit_similarity_3d(source, target)
        if estimated is None:
            return None
        scale, rotation, translation = estimated
        if model == "rigid":
            translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
            return Fit(rotation, translation)
        return Fit(scale * rotation, translation, scale)
    source_origin, target_origin = source.mean(axis=0), target.mean(axis=0)
    fit, _, rank, _ = np.linalg.lstsq(
        source - source_origin, target - target_origin, rcond=_RANK_TOLERANCE
    )
    if rank != 3 or not _has_rank(fit, 3):
        return None
    linear = fit.T
    return Fit(linear, target_origin - linear @ source_origin)


def _native_inliers(source, target, model, tolerance, confidence):
    # Both native RANSAC paths cast coordinates to float32. Remove the large
    # frame origins before that cast and express the threshold at the same scale.
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    span = max(np.abs(a).max(), np.abs(b).max()) or 1.0
    if model == "translation":
        estimate = cv2.estimateTranslation3D
    else:
        estimate = cv2.estimateAffine3D
    success, _, mask = estimate(
        a / span, b / span, ransacThreshold=tolerance / span, confidence=confidence
    )
    if not success or mask is None:
        return None
    if mask.size != len(source):
        raise RuntimeError("OpenCV returned an invalid correspondence mask")
    return mask.reshape(-1) != 0


def _consensus(source, target, parameters, cancelled):
    model = parameters["model"]
    minimum = _MINIMUM[model]
    tolerance = float(parameters["inlier_threshold_m"])
    best, best_mask, best_error, best_score = None, None, None, (0, -np.inf)

    def consider(fit):
        nonlocal best, best_mask, best_error, best_score
        if fit is None:
            return False
        error = np.linalg.norm(fit.apply(source) - target, axis=1)
        mask = error <= tolerance
        count = int(mask.sum())
        score = (count, -float(np.sum(error[mask] ** 2)))
        if count < minimum or score <= best_score:
            return False
        if not _supports_geometry(source[mask], target[mask], model):
            return False
        best, best_mask, best_error, best_score = fit, mask, error, score
        return True

    consider(_refit(source, target, model))
    if cancelled():
        raise InterruptedError("Matched 3D estimation cancelled")
    if best_score[0] < len(source):
        if model in ("translation", "affine"):
            mask = _native_inliers(
                source, target, model, tolerance, float(parameters["ransac_confidence"])
            )
            if mask is not None and mask.sum() >= minimum:
                consider(_refit(source[mask], target[mask], model))
        else:
            rng = np.random.default_rng(int(parameters["random_seed"]))
            for _ in range(int(parameters["maximum_trials"])):
                if cancelled():
                    raise InterruptedError("Matched 3D estimation cancelled")
                rows = rng.choice(len(source), size=3, replace=False)
                consider(_refit(source[rows], target[rows], model))
                if best_score[0] == len(source):
                    break
    if cancelled():
        raise InterruptedError("Matched 3D estimation cancelled")
    if best is None:
        return None, None, None
    for _ in range(4):
        prior = best_mask
        improved = consider(_refit(source[prior], target[prior], model))
        if not improved or np.array_equal(prior, best_mask):
            break
    return best, best_mask, best_error


def _preview(predicted, target, inliers, frame):
    selected = np.linspace(0, len(target) - 1, min(len(target), 256), dtype=int)
    expected, observed = predicted[selected], target[selected]
    rejected = ~inliers[selected]
    points = np.concatenate((expected, observed))
    if np.max(np.abs(points)) > np.finfo(np.float32).max:
        return {}
    colors = np.vstack(
        (
            np.tile([77, 161, 224], (len(selected), 1)),
            np.where(rejected[:, None], [246, 116, 62], [93, 225, 143]),
        )
    )
    segments = (
        np.stack((expected[rejected], observed[rejected]), axis=1) if rejected.any() else None
    )
    cloud = PointCloud(points, "m", frame, colors=colors)
    return {"Matched 3D residuals": render_cloud(cloud, segments=segments, point_size=3)}


class EstimateMatchedTransform(Tool3D):
    type_id = "3d.estimate_matched_transform"
    title, subtype = "Estimate transform from matched 3D points", "3D registration"
    description = "Fit and verify a metric transform from known cloud-row correspondences."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "target"
    inputs = (
        Port("source", "cloud"),
        Port("target", "cloud"),
        Port("pairs", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("similarity", "data", required=False),
        Port("affine", "data", required=False),
        Port("inlier_pairs", "data"),
        Port("report", "data"),
    )
    defaults = {
        "model": "rigid",
        "output_units": "m",
        "inlier_threshold_m": 0.001,
        "minimum_inlier_fraction": 0.6,
        "ransac_confidence": 0.99,
        "maximum_trials": 256,
        "random_seed": 0,
    }
    parameter_choices = {"model": tuple(_MINIMUM), "output_units": tuple(METERS)}
    parameter_ranges = {
        "inlier_threshold_m": NumericRange(1e-7, 1000),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "ransac_confidence": NumericRange(0.5, 0.9999),
        "maximum_trials": NumericRange(1, 2048, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "model": "Choose translation, rigid motion, uniform similarity, or full affine.",
        "output_units": "Linear units for both sides of the resulting transform record.",
        "inlier_threshold_m": "Inclusive Euclidean residual limit in meters.",
        "minimum_inlier_fraction": "Fraction of all matched rows required for acceptance.",
        "ransac_confidence": "OpenCV confidence for translation and affine consensus.",
        "maximum_trials": "Maximum sampled triples for rigid and similarity consensus.",
        "random_seed": "Reproducible local sampler seed for rigid and similarity modes.",
    }
    input_help = {
        "source": "Metric landmarks in the source frame.",
        "target": "Metric landmarks in the target frame. Units may differ.",
        "pairs": "Optional one-to-one [source row, target row] pairs. Default pairs by row order.",
    }
    output_help = {
        "pose": "Standard rigid pose for translation and rigid models.",
        "similarity": "Proper rotation, positive uniform scale, and translation.",
        "affine": "Invertible full 3×3 linear map and translation.",
        "inlier_pairs": "Verified [source row, target row] pairs in original inputs.",
        "report": "Frame, units, support, and geometric residuals.",
    }

    def process(self, inputs):
        source_cloud, target_cloud = inputs["source"], inputs["target"]
        p, units = self.parameters, self.parameters["output_units"]
        if source_cloud.units not in METERS or target_cloud.units not in METERS:
            raise ValidationError(
                "Both landmark clouds need metric units m or mm", "source", "target"
            )
        pairs = _match_rows(inputs.get("pairs"), len(source_cloud.points), len(target_cloud.points))
        source = source_cloud.points[pairs[:, 0]].astype(np.float64) * METERS[source_cloud.units]
        target = target_cloud.points[pairs[:, 1]].astype(np.float64) * METERS[target_cloud.units]
        model = p["model"]
        report = {
            "model": model,
            "source_frame": source_cloud.frame,
            "target_frame": target_cloud.frame,
            "source_units": source_cloud.units,
            "target_units": target_cloud.units,
            "output_units": units,
            "pair_count": len(pairs),
            "inlier_count": 0,
            "inlier_fraction": 0.0,
            "inlier_rmse_m": None,
            "max_inlier_error_m": None,
        }
        outputs = {
            "pose": NO_VALUE,
            "similarity": NO_VALUE,
            "affine": NO_VALUE,
            "inlier_pairs": [],
            "report": report,
        }
        if len(pairs) < _MINIMUM[model]:
            issue, fit = "insufficient_pairs", None
        elif not _supports_geometry(source, target, model):
            issue, fit = "degenerate_geometry", None
        else:
            fit, mask, errors = _consensus(source, target, p, self._cancelled)
            issue = "no_consensus" if fit is None else None
        graphics = {}
        if fit is not None:
            residual = errors[mask]
            fraction = len(residual) / len(pairs)
            report.update(
                inlier_count=len(residual),
                inlier_fraction=fraction,
                inlier_rmse_m=float(np.sqrt(np.mean(residual**2))),
                max_inlier_error_m=float(residual.max()),
            )
            outputs["inlier_pairs"] = pairs[mask].tolist()
            if self.capture_graphics:
                graphics = _preview(fit.apply(source), target, mask, target_cloud.frame)
            if fraction < float(p["minimum_inlier_fraction"]):
                issue = "insufficient_inlier_fraction"
            else:
                translation = fit.translation / METERS[units]
                fields = {
                    "source_frame": source_cloud.frame,
                    "target_frame": target_cloud.frame,
                    "units": units,
                    "translation": translation.tolist(),
                }
                if model in ("translation", "rigid"):
                    outputs["pose"] = pose_record(
                        pose_matrix(fit.linear, translation),
                        source_cloud.frame,
                        target_cloud.frame,
                        units,
                    )
                elif model == "similarity":
                    outputs["similarity"] = {
                        "model": "similarity_3d_v1",
                        "scale": fit.scale,
                        "rotation": (fit.linear / fit.scale).tolist(),
                        **fields,
                    }
                else:
                    outputs["affine"] = {
                        "model": "affine_3d_v1",
                        "linear": fit.linear.tolist(),
                        **fields,
                    }
        report["status"] = issue or "ok"
        return Result(
            outputs,
            passed=issue is None,
            graphics=graphics,
            score=report["inlier_fraction"],
            fail_reasons=[f"matched_transform_{issue}"] if issue else [],
            message=f"{report['inlier_count']}/{len(pairs)} matched rows · {report['status']}",
        )
