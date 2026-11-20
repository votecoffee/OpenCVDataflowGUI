"""Point-cloud filtering, sampling, surface normals, ICP and PPF matching."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, pose_record, rigid, transformed_cloud
from .parameter_validation import NumericRange
from .payloads import NO_VALUE, image_with_payload
from .pointclouds import PointCloud, render_cloud
from .processing import finite, integer


def surface_module():
    module = getattr(cv2, "ppf_match_3d", None)
    if module is None:
        raise RuntimeError("This OpenCV contrib build does not expose surface matching")
    return module


class TransformCloud(Tool3D):
    type_id, title, subtype = "3d.cloud_transform", "Transform point cloud", "Point clouds"
    inputs = (Port("cloud", "cloud"), Port("transform", "data"))
    outputs = (Port("cloud", "cloud"),)

    def process(self, inputs):
        cloud, pose = inputs["cloud"], inputs["transform"]
        if cloud.frame != pose["source_frame"] or cloud.units != pose["units"]:
            raise ValueError("Cloud frame/units must match transform source frame/units")
        result = transformed_cloud(cloud, pose, pose["target_frame"])
        return Result({"cloud": result}, graphics=cloud_graphics(self, result))


class FilterCloud(Tool3D):
    type_id, title, subtype = "3d.cloud_filter", "Filter point cloud", "Point clouds"
    inputs = (
        Port("cloud", "cloud"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("cloud", "cloud"), Port("count", "number"))
    defaults = {
        "minimum": [-1e6, -1e6, -1e6],
        "maximum": [1e6, 1e6, 1e6],
        "invert": False,
        "sample_step": 1,
        "minimum_points": 1,
    }

    def process(self, inputs):
        from .selection import selection_mask

        cloud, p = inputs["cloud"], self.parameters
        low, high = (np.asarray(p[key], float) for key in ("minimum", "maximum"))
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite([low, high]).all()
            or np.any(high < low)
        ):
            raise ValueError("XYZ bounds must be finite ordered triples in cloud units")
        keep = np.all((cloud.points >= low) & (cloud.points <= high), axis=1)
        if "mask" in inputs or "region" in inputs:
            if cloud.pixels is None:
                raise ValueError(
                    "Image region filtering requires retained cloud pixel correspondence"
                )
            grid = np.zeros((cloud.source_size[1], cloud.source_size[0]), np.uint8)
            mask = selection_mask(inputs, image_with_payload(grid, cloud.metadata))
            keep &= mask[cloud.pixels[:, 1], cloud.pixels[:, 0]] != 0
        if p["invert"]:
            keep = ~keep
        selected = np.flatnonzero(keep)[:: integer(p["sample_step"], "Sample step", 1, 100000)]
        result = cloud.subset(selected)
        passed = len(selected) >= integer(p["minimum_points"], "Minimum points", 0, 2000000)
        return Result(
            {"cloud": result, "count": len(selected)},
            passed=passed,
            graphics=cloud_graphics(self, result),
            fail_reasons=[] if passed else ["cloud_count"],
        )


class VoxelCloud(Tool3D):
    type_id, title, subtype = "3d.cloud_voxel", "Voxel-sample point cloud", "Point clouds"
    description = "Group points in a voxel grid and retain a measured row or its centroid."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("cloud", "cloud"),
        Port("count", "number"),
        Port("indices", "data", required=False),
    )
    defaults = {
        "voxel_size": 0.01,
        "mode": "first",
        "spacing": "uniform",
        "voxel_size_x": 0.01,
        "voxel_size_y": 0.01,
        "voxel_size_z": 0.01,
    }
    parameter_choices = {
        "mode": ("first", "centroid", "representative"),
        "spacing": ("uniform", "per_axis"),
    }
    parameter_ranges = {
        name: NumericRange(1e-12)
        for name in ("voxel_size", "voxel_size_x", "voxel_size_y", "voxel_size_z")
    }
    parameter_help = {
        "mode": "Keep the first row, compute a centroid, or keep the nearest row to the centroid.",
        "spacing": "Use the same voxel size on every axis or separate X, Y and Z sizes.",
        "voxel_size": "Size on all axes for Uniform spacing, in the cloud's units.",
        "voxel_size_x": "Size along X for Per axis spacing, in the cloud's units.",
        "voxel_size_y": "Size along Y for Per axis spacing, in the cloud's units.",
        "voxel_size_z": "Size along Z for Per axis spacing, in the cloud's units.",
    }

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        spacing = (
            float(p["voxel_size"])
            if p["spacing"] == "uniform"
            else [float(p[f"voxel_size_{axis}"]) for axis in "xyz"]
        )
        cells = np.divide(cloud.points, spacing, dtype=np.float64)
        np.floor(cells, out=cells)
        if np.any(np.abs(cells) > 1e15):
            raise ValueError("Voxel coordinates exceed the supported range")
        indices, inverse = np.unique(
            cells.astype(np.int64), axis=0, return_index=True, return_inverse=True
        )[1:]
        del cells
        mode = p["mode"]
        needs_representative = mode == "representative" and len(indices) < len(inverse)
        if (mode == "centroid" and len(indices)) or needs_representative:
            counts = np.bincount(inverse)
            sums = np.column_stack(
                [np.bincount(inverse, weights=column) for column in cloud.points.T]
            )
            if needs_representative:
                # Within a voxel all squared distances share the same count-squared
                # denominator. Comparing numerators avoids rounding the centroid first.
                sizes = counts[inverse]
                squared = np.zeros(len(inverse), np.float64)
                for axis in range(3):
                    residual = cloud.points[:, axis].astype(np.float64) * sizes
                    residual -= sums[inverse, axis]
                    squared += np.square(residual, out=residual)
                nearest = np.full(len(indices), np.inf)
                np.minimum.at(nearest, inverse, squared)
                winners = np.flatnonzero(squared == nearest[inverse])
                indices.fill(len(inverse))
                np.minimum.at(indices, inverse[winners], winners)

        result = cloud.subset(indices)
        if mode == "centroid" and len(indices):
            result.points = (sums / counts[:, None]).astype(np.float32)
            if cloud.colors is not None:
                result.colors = np.rint(
                    np.column_stack(
                        [np.bincount(inverse, weights=column) / counts for column in cloud.colors.T]
                    )
                ).astype(np.uint8)
            # Centroid rows do not carry individual measured pixels or normals.
            result.pixels, result.source_size, result.normals = None, None, None
        source_indices = NO_VALUE
        if mode != "centroid" and "indices" in self.exposed_outputs:
            source_indices = indices.tolist()
        return Result(
            {"cloud": result, "count": len(indices), "indices": source_indices},
            graphics=cloud_graphics(self, result),
            payloads={"cloud": self.input_payloads["cloud"]},
        )


class MeasureCloud(Tool3D):
    type_id, title, subtype = "3d.cloud_measure", "Measure point cloud", "Point clouds"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("measurements", "data"),)

    def process(self, inputs):
        cloud = inputs["cloud"]
        result = cloud.describe()
        if len(cloud.points):
            result.update(
                centroid=cloud.points.mean(axis=0).tolist(),
                extent=np.ptp(cloud.points, axis=0).tolist(),
            )
            if len(cloud.points) >= 3:
                mean, eigenvectors, eigenvalues = cv2.PCACompute2(
                    cloud.points.astype(np.float64), mean=None
                )
                result.update(
                    principal_axes=eigenvectors.tolist(),
                    principal_variances=eigenvalues.reshape(-1).tolist(),
                )
        return Result({"measurements": result}, passed=bool(len(cloud.points)))


class CloudNormals(Tool3D):
    type_id, title, subtype = "3d.cloud_normals", "Estimate point-cloud normals", "Surface matching"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("cloud", "cloud"),)
    defaults = {"neighbors": 20, "flip_viewpoint": True, "viewpoint": [0.0, 0.0, 0.0]}

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        neighbors = integer(p["neighbors"], "Neighbors", 3, 10000)
        if len(cloud.points) <= neighbors:
            raise ValueError("Normal estimation needs more cloud points than neighbors")
        viewpoint = np.asarray(p["viewpoint"], float)
        if viewpoint.shape != (3,) or not np.isfinite(viewpoint).all():
            raise ValueError("Normal viewpoint must be finite XYZ in cloud units")
        status, normals = surface_module().computeNormalsPC3d(
            cloud.points, neighbors, p["flip_viewpoint"], tuple(viewpoint)
        )
        if status not in (0, 1) or normals is None or normals.shape != (len(cloud.points), 6):
            raise ValueError("OpenCV could not estimate cloud normals")
        normal = normals[:, 3:6]
        lengths = np.linalg.norm(normal, axis=1)
        keep = np.isfinite(normal).all(axis=1) & (lengths > 1e-8)
        result = cloud.subset(keep)
        result.normals = normal[keep] / lengths[keep, None]
        return Result(
            {"cloud": result}, passed=bool(keep.any()), graphics=cloud_graphics(self, result)
        )


def cloud_with_normals(cloud):
    if cloud.normals is None or len(cloud.points) < 6:
        raise ValueError("Surface matching requires at least six points with unit normals")
    return np.ascontiguousarray(np.c_[cloud.points, cloud.normals], np.float32)


class ICPCloud(Tool3D):
    type_id, title, subtype = "3d.icp", "Point-to-plane ICP registration", "Surface matching"
    inputs = (
        Port("source", "cloud"),
        Port("target", "cloud"),
        Port("initial_pose", "data", required=False),
    )
    outputs = (Port("pose", "data"), Port("cloud", "cloud"), Port("residual", "number"))
    defaults = {
        "iterations": 100,
        "tolerance": 0.005,
        "rejection_scale": 2.5,
        "pyramid_levels": 4,
        "max_residual": 0.05,
    }

    def process(self, inputs):
        source, target, p = inputs["source"], inputs["target"], self.parameters
        if source.units != target.units:
            raise ValueError("ICP cloud length units must match")
        initial = inputs.get("initial_pose")
        transform = np.eye(4)
        if initial:
            if (
                initial["source_frame"] != source.frame
                or initial["target_frame"] != target.frame
                or initial["units"] != source.units
            ):
                raise ValueError("Initial ICP pose frames/units do not match the clouds")
            transform = rigid(initial)
        model = transformed_cloud(source, transform, target.frame)
        icp = surface_module().ICP(
            integer(p["iterations"], "Iterations", 1, 10000),
            finite(p["tolerance"], "Tolerance", 1e-12),
            finite(p["rejection_scale"], "Rejection scale", 1e-12),
            integer(p["pyramid_levels"], "Pyramid levels", 1, 10),
        )
        status, residual, delta = icp.registerModelToScene(
            cloud_with_normals(model), cloud_with_normals(target)
        )
        passed = (
            status == 0
            and np.isfinite(residual)
            and residual <= finite(p["max_residual"], "Maximum residual", 0)
        )
        pose, result = NO_VALUE, NO_VALUE
        if passed:
            pose = pose_record(rigid(delta) @ transform, source.frame, target.frame, source.units)
            result = transformed_cloud(source, pose, target.frame)
        return Result(
            {
                "pose": pose,
                "cloud": result,
                "residual": float(residual) if np.isfinite(residual) else 1e30,
            },
            passed=passed,
            graphics=cloud_graphics(self, result) if passed else {},
            message=f"OpenCV ICP residual {residual:.6g}",
            fail_reasons=[] if passed else ["icp_registration"],
        )


class RenderCloud(Tool3D):
    type_id, title, subtype = "3d.cloud_render", "Render point cloud", "Point clouds"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("image"),)
    defaults = {
        "width": 640,
        "height": 480,
        "yaw_degrees": 25.0,
        "pitch_degrees": -20.0,
        "max_drawn_points": 20000,
        "point_size": 2,
    }

    def process(self, inputs):
        p = self.parameters
        return Result(
            {
                "image": render_cloud(
                    inputs["cloud"],
                    integer(p["width"], "Width", 64, 4096),
                    integer(p["height"], "Height", 64, 4096),
                    finite(p["yaw_degrees"], "Yaw"),
                    finite(p["pitch_degrees"], "Pitch"),
                    integer(p["max_drawn_points"], "Max drawn points", 1, 100000),
                    integer(p["point_size"], "Point size", 1, 10),
                )
            }
        )


class TrainPPF(Tool3D):
    type_id, title, subtype = "3d.ppf_train", "Train 3D PPF surface model", "Surface matching"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("model", "data"),)
    defaults = {"relative_sampling_step": 0.05, "relative_distance_step": 0.05, "angle_bins": 30}

    def process(self, inputs):
        cloud = inputs["cloud"]
        values = cloud_with_normals(cloud)
        if len(values) > 5000:
            raise ValueError("PPF model is limited to 5000 points; voxel sample first")
        model = {
            "model": "ppf_surface_v1",
            "points_normals": values.tolist(),
            "units": cloud.units,
            "frame": cloud.frame,
            "settings": copy.deepcopy(self.parameters),
        }
        trained_ppf(model)
        return Result({"model": model}, message=f"Trained PPF with {len(values)} oriented points")


def trained_ppf(model):
    if not isinstance(model, dict) or model.get("model") != "ppf_surface_v1":
        raise ValueError("Connect a trained 3D PPF surface model")
    values = np.asarray(model["points_normals"], np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 6
        or not 6 <= len(values) <= 5000
        or not np.isfinite(values).all()
    ):
        raise ValueError("PPF model requires 6..5000 finite XYZ/normal rows")
    PointCloud(values[:, :3], model["units"], model["frame"], normals=values[:, 3:])
    p = model["settings"]
    step = finite(p["relative_sampling_step"], "Relative sampling step", 0.001)
    distance = finite(p["relative_distance_step"], "Relative distance step", 0.001)
    if max(step, distance) > 1:
        raise ValueError("Relative PPF sampling/distance steps must not exceed one")
    detector = surface_module().PPF3DDetector(
        step, distance, integer(p["angle_bins"], "Angle bins", 4, 180)
    )
    detector.trainModel(values)
    return detector


class MatchPPF(Tool3D):
    type_id, title, subtype = "3d.ppf_match", "Match 3D PPF surface model", "Surface matching"
    inputs = (Port("model", "data"), Port("scene", "cloud"))
    outputs = (Port("poses", "data"), Port("pose", "data"), Port("count", "number"))
    defaults = {
        "scene_sample_step": 0.2,
        "scene_distance": 0.05,
        "max_matches": 5,
        "minimum_votes": 1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._detector, self._key = None, None

    def reset_runtime(self):
        self._detector, self._key = None, None

    close = reset_runtime

    def process(self, inputs):
        model, cloud, p = inputs["model"], inputs["scene"], self.parameters
        if cloud.units != model["units"]:
            raise ValueError("PPF scene/model units differ")
        key = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        if key != self._key:
            self._detector, self._key = trained_ppf(model), key
        step, distance = (
            finite(p["scene_sample_step"], "Scene sample step", 0.001),
            finite(p["scene_distance"], "Scene distance", 0.001),
        )
        if max(step, distance) > 1 or len(cloud.points) > 20000:
            raise ValueError("PPF ratios must be <=1 and scene limited to 20000 oriented points")
        found = self._detector.match(cloud_with_normals(cloud), step, distance)
        minimum = integer(p["minimum_votes"], "Minimum votes", 1, 1000000000)
        poses = []
        for pose in sorted(found, key=lambda obj: -obj.numVotes):
            if pose.numVotes < minimum:
                continue
            poses.append(
                pose_record(
                    pose.pose, model["frame"], cloud.frame, cloud.units, votes=int(pose.numVotes)
                )
            )
            if len(poses) >= integer(p["max_matches"], "Max matches", 1, 100):
                break
        return Result(
            {"poses": poses, "pose": poses[0] if poses else NO_VALUE, "count": len(poses)},
            passed=bool(poses),
            fail_reasons=[] if poses else ["ppf_surface_match"],
        )


CLOUD_BLOCKS = (
    TransformCloud,
    FilterCloud,
    VoxelCloud,
    MeasureCloud,
    CloudNormals,
    ICPCloud,
    RenderCloud,
    TrainPPF,
    MatchPPF,
)
