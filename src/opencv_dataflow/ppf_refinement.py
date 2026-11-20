"""Rank bounded PPF pose hypotheses using independent geometric support."""

import cv2
import numpy as np

from .cloud_comparison import nearest_cloud_rows
from .geometry3d_common import pose_record, rigid
from .icp_diagnostics import evaluate_icp, reciprocal_cloud_rows
from .pointclouds import PointCloud, render_cloud


def _geometry(xyz, scene, p, cancelled):
    radius = float(p["correspondence_radius"])
    evaluated = evaluate_icp(xyz, scene, radius, cancelled)
    if evaluated is None:
        return None
    metrics = evaluated[0]
    overlap = metrics["source_overlap_fraction"]
    rmse = metrics["inlier_rmse"]
    failures = []
    if overlap < float(p["minimum_model_overlap"]):
        failures.append("model_overlap")
    if metrics["inlier_count"] < p["minimum_inliers"]:
        failures.append("inlier_count")
    if rmse is not None and rmse > float(p["maximum_inlier_rmse"]):
        failures.append("inlier_rmse")
    return {
        "model_overlap": overlap,
        "model_overlap_count": metrics["source_overlap_count"],
        "inlier_count": metrics["inlier_count"],
        "inlier_rmse": rmse,
        "point_to_plane_rmse": metrics["point_to_plane_rmse"],
        "score": overlap * max(0.0, 1 - rmse / radius) if rmse is not None else 0.0,
        "failed_checks": failures,
    }


def _order(quality):
    rmse = quality["inlier_rmse"]
    return (
        bool(quality["failed_checks"]),
        -quality["score"],
        -quality["model_overlap"],
        rmse if rmse is not None else float("inf"),
    )


def _local_pairs(xyz, scene_xyz, radius, cancelled):
    forward = nearest_cloud_rows(xyz, scene_xyz, radius, 0, cancelled)
    if forward is None or cancelled():
        return None
    reverse = nearest_cloud_rows(scene_xyz, xyz, radius, 0, cancelled)
    if reverse is None or cancelled():
        return None
    return reciprocal_cloud_rows(forward, reverse)


def _registration(icp, source, target, seed):
    try:
        status, residual, delta = icp.registerModelToScene(source, target)
    except cv2.error:
        return None, {"status": "native_error"}
    native = {
        "status": "native_rejected",
        "code": int(status),
        "residual": float(residual) if np.isfinite(residual) else None,
    }
    if status != 0 or not np.isfinite(residual):
        return None, native
    try:
        matrix = rigid(rigid(delta) @ seed)
    except ValueError:
        native["status"] = "invalid_transform"
        return None, native
    native["status"] = "evaluated"
    return matrix, native


def _evaluate_candidate(candidate, model, scene, scene_xyz, p, icp, cancelled):
    report = {"candidate": candidate["candidate"], "votes": candidate["votes"]}
    try:
        seed = rigid(candidate["matrix"])
    except ValueError:
        return None, dict(report, status="invalid_seed")
    xyz = model.points @ seed[:3, :3].T + seed[:3, 3]
    metrics = _geometry(xyz, scene, p, cancelled)
    if metrics is None:
        return None, None
    chosen, chosen_metrics, selected_pass = seed, metrics, 0
    passes = []
    current = seed
    minimum_local = 6 * 2 ** (int(p["icp_pyramid_levels"]) - 1)
    for _ in range(int(p["refinement_passes"])):
        nearby = _local_pairs(xyz, scene_xyz, float(p["refinement_distance"]), cancelled)
        if nearby is None:
            return None, None
        model_rows, scene_rows = nearby
        native = {"local_pairs": len(model_rows), "status": "insufficient_local_support"}
        passes.append(native)
        if len(model_rows) < minimum_local:
            break
        normals = model.normals[model_rows] @ current[:3, :3].T
        source = np.ascontiguousarray(np.column_stack((xyz[model_rows], normals)), np.float32)
        target = np.ascontiguousarray(
            np.column_stack((scene.points[scene_rows], scene.normals[scene_rows])), np.float32
        )
        refined, result = _registration(icp, source, target, current)
        if cancelled():
            return None, None
        native.update(result)
        if refined is None:
            break
        current = refined
        xyz = model.points @ current[:3, :3].T + current[:3, 3]
        updated = _geometry(xyz, scene, p, cancelled)
        if updated is None:
            return None, None
        native["quality"] = updated
        if _order(updated) < _order(chosen_metrics):
            chosen, chosen_metrics, selected_pass = current, updated, len(passes)
    report.update(
        seed=metrics,
        refinement=passes,
        selected_pass=selected_pass,
        quality=chosen_metrics,
    )
    if chosen_metrics["failed_checks"]:
        return None, dict(report, status="rejected")
    return chosen, dict(report, status="accepted")


def _is_duplicate(matrix, other, center, distance, angle):
    separation = (matrix[:3, :3] - other[:3, :3]) @ center + matrix[:3, 3] - other[:3, 3]
    if np.linalg.norm(separation) > distance:
        return False
    cosine = (np.sum(matrix[:3, :3] * other[:3, :3]) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1))) <= angle


def rank_surface_candidates(candidates, model, scene, p, icp, cancelled):
    """Evaluate quality, keep the stronger fit per pose and preserve candidate receipts."""
    scene_xyz = scene.points.astype(np.float64)
    reports, accepted = [], []
    for candidate in candidates:
        if cancelled():
            return None
        matrix, report = _evaluate_candidate(candidate, model, scene, scene_xyz, p, icp, cancelled)
        if report is None:
            return None
        reports.append(report)
        if matrix is not None:
            accepted.append((matrix, report))
    accepted.sort(
        key=lambda row: (*_order(row[1]["quality"]), -row[1]["votes"], row[1]["candidate"])
    )
    center = model.points.mean(axis=0, dtype=np.float64)
    distance, angle = float(p["duplicate_translation"]), float(p["duplicate_rotation_deg"])
    winners, poses = [], []
    for matrix, report in accepted:
        duplicate = next(
            (
                key
                for winner, key in winners
                if _is_duplicate(matrix, winner, center, distance, angle)
            ),
            None,
        )
        if duplicate is not None:
            report.update(status="duplicate", duplicate_of=duplicate)
        elif len(poses) == p["max_matches"]:
            report["status"] = "output_limit"
        else:
            winners.append((matrix, report["candidate"]))
            report["rank"] = len(winners)
            poses.append(
                pose_record(
                    matrix,
                    model.frame,
                    scene.frame,
                    scene.units,
                    votes=report["votes"],
                    candidate=report["candidate"],
                    refined=bool(report["selected_pass"]),
                    quality=report["quality"],
                )
            )
    return poses, reports


def render_matches(scene, model, poses):
    palette = ((60, 225, 105), (255, 168, 65), (85, 180, 255), (216, 125, 245))
    xyz = [scene.points]
    colors = [np.full((len(scene.points), 3), 95, np.uint8)]
    for i, pose in enumerate(poses):
        matrix = rigid(pose)
        xyz.append(model.points @ matrix[:3, :3].T + matrix[:3, 3])
        colors.append(np.tile(palette[i % len(palette)], (len(model.points), 1)))
    review = PointCloud(np.vstack(xyz), scene.units, scene.frame, colors=np.vstack(colors))
    return render_cloud(review)
