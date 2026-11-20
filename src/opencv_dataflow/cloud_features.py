"""Register partially overlapping point clouds with local metric shape descriptors."""

import numpy as np
from scipy.spatial import cKDTree

from .core import Port, Result
from .geometry3d_common import Tool3D, pose_record, transformed_cloud
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud

_METERS = {"m": 1.0, "mm": 0.001}
_MAX_POINTS = 250_000


def _sample_indices(count, limit, rng):
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, limit, replace=False))


def _local_features(xyz, tree, rng, settings):
    limit, k = int(settings["maximum_keypoints"]), int(settings["neighbors"])
    radius = float(settings["feature_radius_m"])
    ids = _sample_indices(len(xyz), limit, rng)
    distances, neighbors = tree.query(xyz[ids], k=k + 1, distance_upper_bound=radius)
    valid = np.isfinite(distances[:, -1]) & (distances[:, 1] > 0)
    ids, neighbors = ids[valid], neighbors[valid, 1:]
    if not len(ids):
        return ids, np.empty((0, k + 8))
    radial = distances[valid, 1:] / (radius * np.sqrt(k))
    offsets = (xyz[neighbors] - xyz[ids, None, :]) / radius
    shape = []
    for window in (offsets[:, : k // 2], offsets):
        mean = window.mean(axis=1)
        centered = window - mean[:, None, :]
        cov = centered.transpose(0, 2, 1) @ centered / window.shape[1]
        eigen = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
        shape.extend((eigen, np.linalg.norm(mean, axis=1, keepdims=True)))
    return ids, np.column_stack((radial, *shape))


def _match_features(source, target, ratio):
    if len(source) < 2 or len(target) < 2:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0)
    distances, target_ids = cKDTree(target).query(source, k=2)
    _, reverse = cKDTree(source).query(target, k=1)
    i = np.arange(len(source))
    near, other = distances.T
    j = target_ids[:, 0]
    eligible = (other > 1e-12) & (near < ratio * other) & (reverse[j] == i)
    return i[eligible], j[eligible], near[eligible]


def _fit_rigid(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    left, singular, right = np.linalg.svd((source - a).T @ (target - b))
    if singular[0] == 0 or singular[1] <= 1e-6 * singular[0]:
        return None
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return rotation, b - rotation @ a


def _residuals(source, target, fit):
    rotation, translation = fit
    return np.linalg.norm(source @ rotation.T + translation - target, axis=1)


def _consensus(source, target, rng, iterations, limit):
    best, errors, keep, best_score = None, None, None, (-np.inf, 2)

    def score(residuals, good):
        # Truncated squared loss prefers a tight consensus over loose threshold grazes.
        loss = np.square(np.minimum(residuals / limit, 1)).sum()
        return -float(loss), int(good.sum())

    for _ in range(iterations):
        seed = rng.choice(len(source), 3, replace=False)
        fit = _fit_rigid(source[seed], target[seed])
        if fit is None:
            continue
        residuals = _residuals(source, target, fit)
        good = residuals <= limit
        candidate_score = score(residuals, good)
        if candidate_score[1] >= 3 and candidate_score > best_score:
            best, errors, keep, best_score = fit, residuals, good, candidate_score
    if best is None:
        return None, None, None

    for _ in range(4):
        refined = _fit_rigid(source[keep], target[keep])
        if refined is None:
            break
        residuals = _residuals(source, target, refined)
        good = residuals <= limit
        candidate_score = score(residuals, good)
        if candidate_score[1] < 3 or candidate_score < best_score:
            break
        settled = np.array_equal(keep, good)
        best, errors, keep, best_score = refined, residuals, good, candidate_score
        if settled:
            break
    return best, errors, keep


def _measure_overlap(source, target, sources, targets, fit, rng, p):
    limit, threshold = int(p["maximum_overlap_samples"]), float(p["inlier_distance_m"])
    rotation, translation = fit
    rows = (
        ("source", source, targets, rotation, translation),
        ("target", target, sources, rotation.T, -rotation.T @ translation),
    )
    result = {}
    for name, local, other, r, t in rows:
        sample = local[_sample_indices(len(local), limit, rng)]
        distances, _ = other.query(
            sample @ r.T + t, distance_upper_bound=np.nextafter(threshold, np.inf)
        )
        close = distances[np.isfinite(distances)]
        result[name] = {
            "samples": len(sample),
            "within_threshold": len(close),
            "fraction": len(close) / len(sample),
            "overlap_rms_m": float(np.sqrt(np.mean(close**2))) if len(close) else None,
        }
    return result


def _comparison_graphic(aligned, target):
    a, b = (
        cloud.points[:: max(1, (len(cloud.points) + 2999) // 3000)] for cloud in (aligned, target)
    )
    rgb = np.vstack(
        (
            np.broadcast_to((250, 168, 65), a.shape),
            np.broadcast_to((81, 213, 242), b.shape),
        )
    )
    comparison = PointCloud(np.vstack((a, b)), target.units, target.frame, colors=rgb)
    return render_cloud(comparison)


class RegisterCloudFeatures(Tool3D):
    type_id, title, subtype = (
        "3d.register_cloud_features",
        "Register clouds by local features",
        "Surface matching",
    )
    inputs = (Port("source", "cloud"), Port("target", "cloud"))
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("cloud", "cloud", required=False),
        Port("inlier_correspondences", "data"),
        Port("report", "data"),
    )
    defaults = {
        "feature_radius_m": 0.10,
        "neighbors": 12,
        "maximum_keypoints": 4096,
        "descriptor_ratio": 0.92,
        "inlier_distance_m": 0.004,
        "max_rms_m": 0.003,
        "minimum_inliers": 12,
        "minimum_inlier_fraction": 0.2,
        "minimum_overlap_fraction": 0.15,
        "maximum_overlap_samples": 8192,
        "ransac_iterations": 1024,
        "random_seed": 0,
    }
    parameter_ranges = {
        "feature_radius_m": NumericRange(1e-6, 1000),
        "neighbors": NumericRange(6, 64, whole=True),
        "maximum_keypoints": NumericRange(3, 8192, whole=True),
        "descriptor_ratio": NumericRange(1e-6, 1),
        "inlier_distance_m": NumericRange(1e-6, 1000),
        "max_rms_m": NumericRange(0, 1000),
        "minimum_inliers": NumericRange(3, 8192, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_overlap_fraction": NumericRange(0, 1),
        "maximum_overlap_samples": NumericRange(3, 32_768, whole=True),
        "ransac_iterations": NumericRange(1, 8192, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "feature_radius_m": "Maximum source and target neighborhood radius in meters.",
        "neighbors": "Nearest samples per descriptor. Sparse neighborhoods are skipped.",
        "maximum_keypoints": "Maximum sampled centers per input cloud.",
        "descriptor_ratio": "Maximum nearest-to-second descriptor-distance ratio.",
        "inlier_distance_m": "3D distance cutoff for correspondence and overlap checks.",
        "max_rms_m": "Maximum geometric RMS of inlier correspondences. Zero disables.",
        "minimum_inliers": "Minimum accepted correspondence count.",
        "minimum_inlier_fraction": "Minimum fraction of reciprocal descriptor matches that fit.",
        "minimum_overlap_fraction": "Minimum geometric overlap fraction in each direction.",
        "maximum_overlap_samples": "Maximum samples per cloud used to estimate overlap.",
        "ransac_iterations": "Maximum three-correspondence pose trials.",
        "random_seed": "Repeatable keypoint, pose-trial and overlap samples.",
    }
    input_help = {
        "source": "Moving cloud with explicit meters or millimeters.",
        "target": "Fixed cloud in the same physical units as Source.",
    }
    payload_reference = "source"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_inliers"]) > int(p["maximum_keypoints"]):
            raise ValidationError("Minimum inliers exceeds keypoint limit", "minimum_inliers")

    def process(self, inputs):
        source, target = inputs["source"], inputs["target"]
        p = self.parameters
        if source.units not in _METERS or source.units != target.units:
            raise ValidationError("Source and target need matching metric cloud units")
        if max(len(source.points), len(target.points)) > _MAX_POINTS:
            raise ValidationError(
                "Use at most 250,000 points per view. Voxel sample larger clouds."
            )
        scale = _METERS[source.units]
        a, b = (cloud.points.astype(np.float64) * scale for cloud in (source, target))
        rng = np.random.default_rng(int(p["random_seed"]))
        a_tree, b_tree = cKDTree(a), cKDTree(b)
        a_ids, a_features = _local_features(a, a_tree, rng, p)
        b_ids, b_features = _local_features(b, b_tree, rng, p)
        li, ri, descriptor_distance = _match_features(
            a_features, b_features, float(p["descriptor_ratio"])
        )
        li, ri = a_ids[li], b_ids[ri]
        pairs, reasons = [], []
        rms = overlap = None
        fraction = 0.0
        pose = aligned = NO_VALUE
        min_inliers = int(p["minimum_inliers"])
        if len(li) < min_inliers:
            reasons.append("cloud_features_insufficient_matches")
        else:
            fit, errors, keep = _consensus(
                a[li], b[ri], rng, int(p["ransac_iterations"]), float(p["inlier_distance_m"])
            )
            if fit is None:
                reasons.append("cloud_features_no_pose_consensus")
            else:
                pairs = [
                    {
                        "source_index": int(li[i]),
                        "target_index": int(ri[i]),
                        "descriptor_distance": float(descriptor_distance[i]),
                        "error_m": float(errors[i]),
                        **(
                            {"source_pixel": source.pixels[li[i]].tolist()}
                            if source.pixels is not None
                            else {}
                        ),
                        **(
                            {"target_pixel": target.pixels[ri[i]].tolist()}
                            if target.pixels is not None
                            else {}
                        ),
                    }
                    for i in np.flatnonzero(keep)
                ]
                fraction = len(pairs) / len(li)
                rms = float(np.sqrt(np.mean(errors[keep] ** 2)))
                overlap = _measure_overlap(a, b, a_tree, b_tree, fit, rng, p)
                if len(pairs) < min_inliers or fraction < float(p["minimum_inlier_fraction"]):
                    reasons.append("cloud_features_insufficient_inliers")
                if (limit := float(p["max_rms_m"])) and rms > limit:
                    reasons.append("cloud_features_fit_rms")
                if min(row["fraction"] for row in overlap.values()) < float(
                    p["minimum_overlap_fraction"]
                ):
                    reasons.append("cloud_features_low_overlap")
                if not reasons:
                    transform = np.eye(4)
                    transform[:3, :3], transform[:3, 3] = fit[0], fit[1] / scale
                    pose = pose_record(transform, source.frame, target.frame, source.units)
                    aligned = transformed_cloud(source, pose, target.frame)

        accepted = not reasons
        report = {
            "model": "local_cloud_registration_v1",
            "source_frame": source.frame,
            "target_frame": target.frame,
            "source_keypoints": len(a_ids),
            "target_keypoints": len(b_ids),
            "reciprocal_matches": len(li),
            "inliers": len(pairs),
            "inlier_fraction": fraction,
            "inlier_rms_m": rms,
            "overlap": overlap,
            "accepted": accepted,
        }
        return Result(
            {"pose": pose, "cloud": aligned, "inlier_correspondences": pairs, "report": report},
            graphics={"Registered views": _comparison_graphic(aligned, target)}
            if accepted and self.generate_graphics
            else {},
            passed=accepted,
            score=fraction,
            fail_reasons=reasons,
            message=f"{len(pairs)}/{len(li)} reciprocal matches agree",
        )
