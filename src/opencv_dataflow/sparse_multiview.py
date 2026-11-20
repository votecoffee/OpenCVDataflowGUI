"""Sparse landmark reconstruction from calibrated views with missing observations."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, points, pose_record, require_pinhole, rigid
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .pointclouds import PointCloud

VIEW_MODEL = "landmark_view_v1"
MAX_VIEWS = 16
MAX_LANDMARKS = 512
MIN_DEPTH = 1e-8  # Dimensionless in the camera-baseline-normalized working frame.


def _identifier(value):
    return (
        type(value) is int
        and abs(value) <= 2**53 - 1
        or (isinstance(value, str) and 0 < len(value) <= 128 and bool(value.strip()))
    )


@dataclass
class _View:
    record: dict
    intrinsics: np.ndarray
    distortion: np.ndarray
    transform: np.ndarray
    pixels: np.ndarray
    ids: list
    frame: str
    units: str


def _parse_view(record):
    if (
        not isinstance(record, dict)
        or record.get("model") != VIEW_MODEL
        or not _identifier(record.get("id"))
    ):
        raise ValidationError("Each landmark view needs a model and a short text or integer ID")
    camera = record.get("camera")
    k, d, size = require_pinhole(camera)
    pose = record.get("world_to_camera")
    camera_frame, units = camera.get("frame", "camera"), camera.get("units", "m")
    if (
        not isinstance(pose, dict)
        or pose.get("model") != "pose_3d_v1"
        or pose.get("target_frame") != camera_frame
        or pose.get("units") != units
        or not isinstance(pose.get("source_frame"), str)
        or not pose["source_frame"]
        or pose["source_frame"] == camera_frame
    ):
        raise ValidationError(
            "Each view needs a world-to-camera pose matching its camera and units"
        )
    ids = record.get("landmark_ids")
    if (
        not isinstance(ids, (list, tuple))
        or not 1 <= len(ids) <= MAX_LANDMARKS
        or not all(_identifier(key) for key in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValidationError("Use 1 to 512 unique short text or integer landmark IDs per view")
    uv = points(record.get("image_points"), 2)
    if len(uv) != len(ids) or np.any(uv < 0) or np.any(uv >= size):
        raise ValidationError("Each landmark ID needs an in-frame pixel in the calibrated image")
    return _View(record, k, d, rigid(pose), uv, list(ids), pose["source_frame"], units)


class MakeLandmarkView(Tool3D):
    type_id, title, subtype = "3d.landmark_view", "Pack calibrated landmark view", "Reconstruction"
    inputs = (
        Port("camera", "data"),
        Port("world_to_camera", "data", visual_type="pose"),
        Port("landmark_ids", "data"),
        Port("image_points", "data"),
    )
    outputs = (Port("view", "data"),)
    defaults = {"view_id": "view"}
    parameter_help = {"view_id": "Unique capture ID in the group of calibrated views."}
    input_help = {
        "image_points": "Original (distorted) XY pixels in the camera model's calibrated grid.",
        "landmark_ids": "Persistent cross-view landmark IDs in point order, such as ChArUco IDs.",
    }
    preview_safe = True
    parallel_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not _identifier(p["view_id"]):
            raise ValidationError("Use a short text or integer view ID", "view_id")

    def process(self, inputs):
        record = {"model": VIEW_MODEL, "id": self.parameters["view_id"]}
        record.update({port.name: inputs[port.name] for port in self.inputs})
        view = _parse_view(record)
        spatial = self.input_payloads.get("image_points", {}).get("spatial", {})
        if spatial and spatial["image_size"] != list(record["camera"]["image_size"]):
            raise ValidationError(
                "Image point payload and calibrated camera must use the same grid"
            )
        record.update(landmark_ids=view.ids, image_points=view.pixels.tolist())
        return Result({"view": record}, message=f"{len(view.ids)} observed landmarks")


def _read_views(block, inputs):
    path = block.parameters["path"]
    values = inputs.get("views")
    if path:
        if values is not None:
            raise ValidationError("Supply a views file or the views connector")
        path = Path(path)
        if not path.is_absolute():
            if block.project_dir is None:
                raise ValidationError("Open or save the project before using a relative views file")
            path = block.project_dir / path
        with path.open("rb") as source:
            raw = source.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValidationError("Landmark views file exceeds 4 MiB")
        values = json.loads(raw)
    if isinstance(values, FlowGroup):
        if any(failed_input(item) for item in values.items):
            raise ValidationError("Every packed landmark view in the group must pass")
        values = [item.value for item in values.items]
    if not isinstance(values, (list, tuple)) or not 3 <= len(values) <= MAX_VIEWS:
        raise ValidationError("Provide three to sixteen calibrated landmark views")
    views = [_parse_view(value) for value in values]
    if len({v.record["id"] for v in views}) != len(views):
        raise ValidationError("View IDs must be unique")
    if len({(v.frame, v.units) for v in views}) != 1:
        raise ValidationError("Views must share a world frame and length units")
    return views


def _working_poses(views):
    transforms = np.stack([view.transform for view in views])
    rotations, translations = transforms[:, :3, :3], transforms[:, :3, 3]
    centers = -np.einsum("nji,nj->ni", rotations, translations)
    origin = centers[0]
    scale = float(np.max(np.linalg.norm(centers - origin, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValidationError("Camera centers must span a nonzero baseline")
    translations = np.einsum("nij,nj->ni", rotations, origin - centers) / scale
    poses = np.c_[Rotation.from_matrix(rotations).as_rotvec(), translations]
    return origin, scale, poses


def _tracks(views, poses, angle):
    tracks = {}
    rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
    projections = np.concatenate((rotations, poses[:, 3:, None]), axis=2)
    for i, view in enumerate(views):
        normalized = cv2.undistortPoints(
            view.pixels[:, None], view.intrinsics, view.distortion
        ).reshape(-1, 2)
        if not np.isfinite(normalized).all():
            raise ValidationError("Undistortion must produce finite calibrated bearings")
        for landmark, uv, xy in zip(view.ids, view.pixels, normalized, strict=True):
            tracks.setdefault(landmark, []).append((i, uv, xy))
    if len(tracks) > MAX_LANDMARKS:
        raise ValidationError("Reconstruction supports up to 512 unique landmarks per run")

    xyz, observed, reports = [], [], []
    for key, track in tracks.items():
        indices = np.asarray([i for i, _, _ in track])
        entry = {
            "id": key,
            "view_ids": [views[i].record["id"] for i in indices],
            "accepted": False,
        }
        reports.append(entry)
        if len(indices) < 2:
            entry["reason"] = "too_few_views"
            continue
        pixels = np.asarray([uv for _, uv, _ in track])
        normalized = np.asarray([xy for _, _, xy in track])
        rays = np.einsum(
            "nji,nj->ni", rotations[indices], np.c_[normalized, np.ones(len(normalized))]
        )
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        minimum_dot = np.min(np.abs(rays @ rays.T))
        p = projections[indices]
        if minimum_dot > angle:
            entry["reason"] = "weak_parallax"
            continue
        rows = normalized[:, :, None] * p[:, None, 2] - p[:, :2]
        _, singular, v = np.linalg.svd(rows.reshape(-1, 4), full_matrices=False)
        if singular[2] <= singular[0] * 1e-10 or abs(v[-1, 3]) < 1e-10:
            entry["reason"] = "degenerate_triangulation"
            continue
        point = v[-1, :3] / v[-1, 3]
        if np.any(p[:, 2, :3] @ point + p[:, 2, 3] <= MIN_DEPTH):
            entry["reason"] = "nonpositive_depth"
            continue
        entry["solver_index"] = len(xyz)
        observed.extend(zip(indices, [len(xyz)] * len(indices), pixels, strict=True))
        xyz.append(point)
    return np.asarray(xyz, float).reshape(-1, 3), observed, reports


class _Bundle:
    def __init__(self, views, poses, observed):
        self.views = views
        self.poses = poses
        self.camera_indices = np.asarray([i for i, _, _ in observed], dtype=int)
        self.point_indices = np.asarray([j for _, j, _ in observed], dtype=int)
        self.pixels = np.asarray([uv for _, _, uv in observed])
        self.by_view = [np.flatnonzero(self.camera_indices == i) for i in range(len(views))]

    def project(self, xyz, poses):
        projected = np.empty_like(self.pixels)
        depths = np.empty(len(projected))
        rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        for i, indices in enumerate(self.by_view):
            if not len(indices):
                continue
            view = self.views[i]
            selected = xyz[self.point_indices[indices]]
            projected[indices] = cv2.projectPoints(
                selected, poses[i, :3], poses[i, 3:], view.intrinsics, view.distortion
            )[0].reshape(-1, 2)
            depths[indices] = selected @ rotations[i, 2] + poses[i, 5]
        return projected - self.pixels, depths

    def refine(self, xyz, *, joint, angle, maximum, loss_scale):
        point_count = len(xyz)
        parameter_count = xyz.size + (6 * (len(self.views) - 2) if joint else 0)
        sparsity = lil_matrix((2 * len(self.pixels), parameter_count), dtype=np.int8)
        for index, (view, point) in enumerate(zip(self.camera_indices, self.point_indices)):
            rows = slice(2 * index, 2 * index + 2)
            sparsity[rows, 3 * point : 3 * point + 3] = 1
            if joint and view >= 2:
                start = xyz.size + 6 * (view - 2)
                sparsity[rows, start : start + 6] = 1

        def unpack(values):
            points = values[: xyz.size].reshape(point_count, 3)
            poses = (
                np.vstack((self.poses[:2], values[xyz.size :].reshape(-1, 6)))
                if joint
                else self.poses
            )
            return points, poses

        def residual(values):
            return self.project(*unpack(values))[0].ravel()

        if joint:
            anchored = np.intersect1d(
                self.point_indices[self.by_view[0]], self.point_indices[self.by_view[1]]
            )
            centers = -Rotation.from_rotvec(self.poses[:2, :3]).inv().apply(self.poses[:2, 3:])
            if np.linalg.norm(centers[1] - centers[0]) <= MIN_DEPTH:
                raise ValidationError(
                    "Joint refinement requires a nonzero baseline between views 1 and 2"
                )
            rays = xyz[anchored][None] - centers[:, None]
            rays /= np.linalg.norm(rays, axis=2, keepdims=True)
            dot = np.abs(np.einsum("ij,ij->i", rays[0], rays[1]))
            anchored = anchored[dot <= angle]
            for i, index in enumerate(self.by_view[2:], 2):
                common = np.intersect1d(anchored, self.point_indices[index])
                shared = xyz[common]
                if len(shared) < 6 or np.linalg.matrix_rank(shared - shared.mean(axis=0)) < 2:
                    raise ValidationError(
                        f"View {self.views[i].record['id']!r} needs six non-collinear "
                        "landmarks with sufficient parallax in both fixed anchors"
                    )
        start = np.r_[xyz.ravel(), self.poses[2:].ravel()] if joint else xyz.ravel()
        fit = least_squares(
            residual,
            start,
            jac_sparsity=sparsity.tocsr(),
            loss="soft_l1",
            f_scale=loss_scale,
            x_scale="jac",
            max_nfev=maximum,
            ftol=1e-7,
        )
        refined_points, refined_poses = unpack(fit.x)
        return refined_points, refined_poses, fit


def _finish_tracks(entries, indices, before, after, depth, limit):
    accepted = []
    by_point = {}
    for index, point in enumerate(indices):
        by_point.setdefault(point, []).append(index)
    for entry in entries:
        point = entry.pop("solver_index", None)
        if point is None:
            continue
        rows = by_point[point]
        values = np.linalg.norm(after[rows], axis=1)
        if not np.isfinite(values).all() or not np.isfinite(before[rows]).all():
            entry["reason"] = "nonfinite_projection"
        else:
            entry["initial_rms_px"] = float(np.sqrt(np.mean(np.sum(before[rows] ** 2, axis=1))))
            entry["rms_px"] = float(np.sqrt(np.mean(values**2)))
            entry["maximum_error_px"] = float(values.max())
            if np.any(depth[rows] <= MIN_DEPTH):
                entry["reason"] = "nonpositive_depth"
            elif entry["maximum_error_px"] > limit:
                entry["reason"] = "reprojection_limit"
            else:
                entry["accepted"] = True
                accepted.append(point)
    return np.asarray(accepted, dtype=int)


class ReconstructSparseViews(Tool3D):
    type_id, title = "3d.reconstruct_sparse_views", "Reconstruct sparse multi-view landmarks"
    subtype = "Reconstruction"
    inputs = (Port("views", "any", required=False, required_when_connected=True),)
    outputs = (
        Port("cloud", "cloud", required=False),
        Port("views", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "path": "",
        "refinement": "points",
        "anchor_view_ids": [],
        "minimum_ray_angle_deg": 0.5,
        "maximum_error_px": 3.0,
        "robust_scale_px": 1.0,
        "maximum_evaluations": 100,
    }
    project_path_parameters = ("path",)
    parameter_choices = {"refinement": ("none", "points", "poses_and_points")}
    parameter_ranges = {
        "minimum_ray_angle_deg": NumericRange(0.01, 90),
        "maximum_error_px": NumericRange(0.01, 100),
        "robust_scale_px": NumericRange(0.01, 100),
        "maximum_evaluations": NumericRange(1, 250, whole=True),
    }
    parameter_help = {
        "path": "Optional project-relative or absolute JSON array of packed views.",
        "refinement": "Fit points alone or jointly with camera poses.",
        "anchor_view_ids": "Two fixed view IDs for the joint fit. Empty uses the first two views.",
        "minimum_ray_angle_deg": "Minimum maximum acute separation between a track's camera rays.",
        "maximum_error_px": "Reject a landmark if any contributing view exceeds this pixel error.",
        "robust_scale_px": "Soft-L1 scale in pixels for the optional reprojection fit.",
        "maximum_evaluations": "Maximum least-squares evaluations for the optional refinement.",
    }
    input_help = {"views": "Three to sixteen packed landmark views as a list or flow group."}
    parallel_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Use a JSON path or leave it empty", "path")
        anchors = p["anchor_view_ids"]
        if not isinstance(anchors, list) or (
            anchors
            and (
                len(anchors) != 2
                or not all(_identifier(v) for v in anchors)
                or anchors[0] == anchors[1]
            )
        ):
            raise ValidationError("Choose two distinct valid anchor view IDs", "anchor_view_ids")

    def process(self, inputs):
        p = self.parameters
        views = _read_views(self, inputs)
        mode = p["refinement"]
        joint = mode == "poses_and_points"
        anchors = p["anchor_view_ids"]
        if joint and anchors:
            by_id = {view.record["id"]: view for view in views}
            if any(key not in by_id for key in anchors):
                raise ValidationError("Both anchor view IDs must be present", "anchor_view_ids")
            views = [by_id[key] for key in anchors] + [
                view for view in views if view.record["id"] not in anchors
            ]
        origin, scale, poses = _working_poses(views)
        angle = np.cos(np.deg2rad(float(p["minimum_ray_angle_deg"])))
        xyz, observed, entries = _tracks(views, poses, angle)
        report = {
            "model": "sparse_multiview_report_v1",
            "frame": views[0].frame,
            "units": views[0].units,
            "view_ids": [view.record["id"] for view in views],
            "anchor_ids": [view.record["id"] for view in views[:2]] if joint else [],
            "refinement": mode,
            "evaluations": 0,
            "converged": None,
            "landmarks": entries,
        }
        reasons = []
        if observed:
            bundle = _Bundle(views, poses, observed)
            before, depths = bundle.project(xyz, poses)
            after = before
            if mode != "none":
                xyz, poses, fit = bundle.refine(
                    xyz,
                    joint=joint,
                    angle=angle,
                    maximum=int(p["maximum_evaluations"]),
                    loss_scale=float(p["robust_scale_px"]),
                )
                report.update(evaluations=fit.nfev, converged=bool(fit.success))
                if not fit.success:
                    reasons.append("multiview_refinement_did_not_converge")
                after, depths = bundle.project(xyz, poses)
            accepted = _finish_tracks(
                entries, bundle.point_indices, before, after, depths, float(p["maximum_error_px"])
            )
            matched = np.isin(bundle.point_indices, accepted)
            if matched.any():
                report["initial_rms_px"] = float(
                    np.sqrt(np.mean(np.sum(before[matched] ** 2, axis=1)))
                )
                report["rms_px"] = float(np.sqrt(np.mean(np.sum(after[matched] ** 2, axis=1))))
        else:
            accepted = np.empty(0, dtype=int)
        count = len(accepted)
        report["accepted_count"] = count
        if not count:
            reasons.append("multiview_no_accepted_landmarks")
        cloud, solved = NO_VALUE, NO_VALUE
        if not reasons:
            restored = xyz[accepted] * scale + origin
            ids = [entry["id"] for entry in entries if entry["accepted"]]
            cloud = PointCloud(
                restored, views[0].units, views[0].frame, metadata={"landmark_ids": ids}
            )
            solved = [dict(view.record) for view in views]
            if joint:
                rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
                for i, view in enumerate(views[2:], 2):
                    transform = np.eye(4)
                    transform[:3, :3] = rotations[i]
                    transform[:3, 3] = poses[i, 3:] * scale - rotations[i] @ origin
                    solved[i]["world_to_camera"] = pose_record(
                        transform,
                        view.frame,
                        view.record["world_to_camera"]["target_frame"],
                        view.units,
                    )
        return Result(
            {"cloud": cloud, "views": solved, "report": report},
            passed=not reasons,
            score=count / len(entries) if not reasons else 0.0,
            fail_reasons=reasons,
            graphics=cloud_graphics(self, cloud) if not reasons else {},
            message=f"{count}/{len(entries)} landmarks accepted from {len(views)} views",
        )
