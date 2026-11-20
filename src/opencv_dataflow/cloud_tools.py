"""Point-cloud filtering, sampling, surface normals, ICP and PPF matching."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, pose_record, rigid, transformed_cloud
from .icp_diagnostics import evaluate_icp
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, image_with_payload
from .pointclouds import PointCloud, render_cloud
from .ppf_refinement import rank_surface_candidates, render_matches
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
    description = "Refine a local pose and optionally verify overlap and geometric constraints."
    payload_reference = "source"
    inputs = (
        Port("source", "cloud"),
        Port("target", "cloud"),
        Port("initial_pose", "data", required=False),
    )
    outputs = (
        Port("pose", "data"),
        Port("cloud", "cloud"),
        Port("residual", "number"),
        Port("report", "data"),
        Port("inliers", "data", required=False),
    )
    defaults = {
        "iterations": 100,
        "tolerance": 0.005,
        "rejection_scale": 2.5,
        "pyramid_levels": 4,
        "max_residual": 0.05,
        "quality_checks": False,
        "correspondence_radius": 0.01,
        "minimum_source_overlap": 0.25,
        "minimum_target_overlap": 0.25,
        "minimum_inliers": 6,
        "maximum_inlier_rmse": 0.0,
        "minimum_constraint_ratio": 0.001,
        "max_points_per_cloud": 250_000,
    }
    parameter_ranges = {
        "iterations": NumericRange(1, 10000, whole=True),
        "tolerance": NumericRange(1e-12),
        "rejection_scale": NumericRange(1e-12),
        "pyramid_levels": NumericRange(1, 10, whole=True),
        "max_residual": NumericRange(0),
        "correspondence_radius": NumericRange(1e-12, 1e20),
        "minimum_source_overlap": NumericRange(0, 1),
        "minimum_target_overlap": NumericRange(0, 1),
        "minimum_inliers": NumericRange(6, 1_000_000, whole=True),
        "maximum_inlier_rmse": NumericRange(0, 1e20),
        "minimum_constraint_ratio": NumericRange(1e-6, 1),
        "max_points_per_cloud": NumericRange(6, 1_000_000, whole=True),
    }
    parameter_help = {
        "iterations": "Maximum native ICP iterations.",
        "tolerance": "Native ICP convergence tolerance.",
        "rejection_scale": "Native outlier rejection scale.",
        "pyramid_levels": "Number of levels in the native ICP refinement.",
        "max_residual": "Maximum residual returned by OpenCV.",
        "quality_checks": "Check the refined pose with independent nearest-point diagnostics.",
        "correspondence_radius": "Inclusive overlap radius in the input clouds' length units.",
        "minimum_source_overlap": "Minimum fraction of source rows with a nearby target.",
        "minimum_target_overlap": "Minimum fraction of target rows with a nearby source.",
        "minimum_inliers": "Minimum number of unique reciprocal nearest pairs.",
        "maximum_inlier_rmse": "Inlier RMSE limit in cloud units. Zero disables it.",
        "minimum_constraint_ratio": (
            "Minimum smallest-to-largest singular value ratio of the normalized pose Jacobian."
        ),
        "max_points_per_cloud": "Maximum points per input when quality checks are enabled.",
    }
    input_help = {
        "source": "Moving point cloud with unit normals.",
        "target": "Fixed point cloud with unit normals using the same length units.",
        "initial_pose": "Optional rigid transform from source frame to target frame.",
    }
    output_help = {
        "pose": "Accepted source-to-target pose.",
        "cloud": "Accepted source cloud transformed into the target frame.",
        "residual": "Residual returned by OpenCV, in its native convention.",
        "report": "Native status, overlap, paired errors and local pose constraints.",
        "inliers": "Reciprocal row indices with Euclidean and signed plane errors.",
    }

    def validate_parameters(self, parameters=None):
        p = self.parameters if parameters is None else parameters
        super().validate_parameters(p)
        if not isinstance(p["quality_checks"], bool):
            raise ValidationError("Quality checks must be on or off", "quality_checks")
        if p["quality_checks"] and float(p["maximum_inlier_rmse"]) > float(
            p["correspondence_radius"]
        ):
            raise ValidationError(
                "Maximum inlier RMSE must be at most the correspondence radius",
                "maximum_inlier_rmse",
                "correspondence_radius",
            )
        if p["quality_checks"] and p["minimum_inliers"] > p["max_points_per_cloud"]:
            raise ValidationError(
                "Minimum inliers must fit within the point budget",
                "minimum_inliers",
                "max_points_per_cloud",
            )

    def _check_candidate(self, matrix, source, target, outputs, reasons):
        p = self.parameters
        radius = float(p["correspondence_radius"])
        xyz = source.points @ matrix[:3, :3].T + matrix[:3, 3]
        quality = evaluate_icp(xyz, target, radius, self._cancelled)
        if quality is None or self._cancelled():
            return None
        metrics, src, dst, distances, plane, forward = quality
        outputs["report"]["quality"] = dict(metrics, correspondence_radius=radius)
        for direction in ("source", "target"):
            overlap = metrics[f"{direction}_overlap_fraction"]
            if overlap < float(p[f"minimum_{direction}_overlap"]):
                reasons.append(
                    {
                        "code": f"icp_{direction}_overlap",
                        "message": f"{direction.title()} overlap is {overlap:.1%}",
                    }
                )
        if metrics["inlier_count"] < p["minimum_inliers"]:
            reasons.append(
                {
                    "code": "icp_inliers",
                    "message": f"Only {metrics['inlier_count']} reciprocal pairs",
                }
            )
        if metrics["constraint_ratio"] < float(p["minimum_constraint_ratio"]):
            reasons.append(
                {
                    "code": "icp_degenerate",
                    "message": "Inliers weakly constrain or leave pose axes free",
                }
            )
        rmse = metrics["inlier_rmse"]
        maximum_rmse = float(p["maximum_inlier_rmse"])
        if maximum_rmse and rmse is not None and rmse > maximum_rmse:
            reasons.append(
                {
                    "code": "icp_inlier_rmse",
                    "message": f"Inlier RMSE is {rmse:.6g} {source.units}",
                }
            )
        if "inliers" in self.exposed_outputs:
            outputs["inliers"] = {
                "model": "icp_inliers_v1",
                "units": source.units,
                "source_index": src.tolist(),
                "target_index": dst.tolist(),
                "distance": distances.tolist(),
                "point_to_plane": plane.tolist(),
            }
        if not self.capture_graphics:
            return {}
        colors = np.full((len(xyz), 3), 128, np.uint8)
        colors[forward.index >= 0] = (65, 150, 235)
        colors[forward.ambiguous] = (245, 184, 60)
        colors[src] = (45, 205, 100)
        return cloud_graphics(self, PointCloud(xyz, source.units, target.frame, colors=colors))

    def process(self, inputs):
        source, target, p = inputs["source"], inputs["target"], self.parameters
        if source.units != target.units:
            raise ValidationError("ICP cloud length units must match", "source", "target")
        if (
            p["quality_checks"]
            and max(len(source.points), len(target.points)) > p["max_points_per_cloud"]
        ):
            raise ValidationError("ICP input exceeds the point budget", "max_points_per_cloud")
        initial = inputs.get("initial_pose")
        transform = np.eye(4)
        if initial is not None:
            if (
                not isinstance(initial, dict)
                or initial.get("source_frame") != source.frame
                or initial.get("target_frame") != target.frame
                or initial.get("units") != source.units
            ):
                raise ValidationError(
                    "Initial ICP pose frames/units must match the clouds", "initial_pose"
                )
            transform = rigid(initial)
        report = {
            "units": source.units,
            "source_frame": source.frame,
            "target_frame": target.frame,
            "source_points": len(source.points),
            "target_points": len(target.points),
            "native_status": None,
            "native_residual": None,
            "quality_checks": bool(p["quality_checks"]),
            "quality": None,
        }
        outputs = dict(
            pose=NO_VALUE, cloud=NO_VALUE, residual=NO_VALUE, report=report, inliers=NO_VALUE
        )

        def cancelled():
            return Result(
                outputs,
                passed=False,
                message="ICP cancelled",
                fail_reasons=["icp_cancelled"],
            )

        if self._cancelled():
            return cancelled()
        model = (
            transformed_cloud(source, transform, target.frame) if initial is not None else source
        )
        icp = surface_module().ICP(
            int(p["iterations"]),
            float(p["tolerance"]),
            float(p["rejection_scale"]),
            int(p["pyramid_levels"]),
        )
        status, residual, delta = icp.registerModelToScene(
            cloud_with_normals(model), cloud_with_normals(target)
        )
        report.update(
            native_status=int(status),
            native_residual=float(residual) if np.isfinite(residual) else None,
        )
        outputs["residual"] = report["native_residual"] if np.isfinite(residual) else 1e30
        if self._cancelled():
            return cancelled()
        reasons = []
        if status != 0 or not np.isfinite(residual) or residual > float(p["max_residual"]):
            reasons.append(
                {
                    "code": "icp_registration",
                    "message": f"OpenCV ICP status {status}, residual {residual:.6g}",
                }
            )
        matrix = None
        if status == 0 and np.isfinite(residual):
            try:
                matrix = rigid(rigid(delta) @ transform)
            except ValueError:
                reasons.append(
                    {"code": "icp_pose_invalid", "message": "OpenCV returned an invalid rigid pose"}
                )

        graphics = {}
        if matrix is not None and p["quality_checks"]:
            graphics = self._check_candidate(matrix, source, target, outputs, reasons)
            if graphics is None or self._cancelled():
                return cancelled()
        if not reasons:
            pose = pose_record(matrix, source.frame, target.frame, source.units)
            result = transformed_cloud(source, pose, target.frame)
            outputs.update(pose=pose, cloud=result)
            if not p["quality_checks"]:
                graphics = cloud_graphics(self, result)
        return Result(
            outputs,
            passed=not reasons,
            graphics=graphics,
            message=reasons[0]["message"] if reasons else f"OpenCV ICP residual {residual:.6g}",
            fail_reasons=reasons,
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
    cloud = PointCloud(values[:, :3], model["units"], model["frame"], normals=values[:, 3:])
    p = model["settings"]
    step = finite(p["relative_sampling_step"], "Relative sampling step", 0.001)
    distance = finite(p["relative_distance_step"], "Relative distance step", 0.001)
    if max(step, distance) > 1:
        raise ValueError("Relative PPF sampling/distance steps must not exceed one")
    detector = surface_module().PPF3DDetector(
        step, distance, integer(p["angle_bins"], "Angle bins", 4, 180)
    )
    detector.trainModel(values)
    return detector, cloud


class MatchPPF(Tool3D):
    type_id, title, subtype = "3d.ppf_match", "Match 3D PPF surface model", "Surface matching"
    description = "Find oriented surface poses and optionally refine and rank them by geometry."
    payload_reference = "scene"
    inputs = (Port("model", "data"), Port("scene", "cloud"))
    outputs = (
        Port("poses", "data"),
        Port("pose", "data"),
        Port("count", "number"),
        Port("report", "data"),
    )
    defaults = {
        "scene_sample_step": 0.2,
        "scene_distance": 0.05,
        "max_matches": 5,
        "minimum_votes": 1,
        "ranking": "votes",
        "candidate_limit": 8,
        "refinement_passes": 2,
        "refinement_distance": 0.015,
        "correspondence_radius": 0.005,
        "minimum_model_overlap": 0.3,
        "minimum_inliers": 6,
        "maximum_inlier_rmse": 0.003,
        "duplicate_translation": 0.005,
        "duplicate_rotation_deg": 5.0,
        "icp_iterations": 50,
        "icp_tolerance": 0.005,
        "icp_rejection_scale": 2.5,
        "icp_pyramid_levels": 1,
    }
    parameter_choices = {"ranking": ("votes", "refined_geometry")}
    parameter_ranges = {
        "scene_sample_step": NumericRange(0.001, 1),
        "scene_distance": NumericRange(0.001, 1),
        "max_matches": NumericRange(1, 100, whole=True),
        "minimum_votes": NumericRange(1, 1_000_000_000, whole=True),
        "candidate_limit": NumericRange(1, 32, whole=True),
        "refinement_passes": NumericRange(1, 4, whole=True),
        "refinement_distance": NumericRange(1e-12, 1e20),
        "correspondence_radius": NumericRange(1e-12, 1e20),
        "minimum_model_overlap": NumericRange(0, 1),
        "minimum_inliers": NumericRange(6, 5000, whole=True),
        "maximum_inlier_rmse": NumericRange(1e-12, 1e20),
        "duplicate_translation": NumericRange(0, 1e20),
        "duplicate_rotation_deg": NumericRange(0, 180),
        "icp_iterations": NumericRange(1, 1000, whole=True),
        "icp_tolerance": NumericRange(1e-12),
        "icp_rejection_scale": NumericRange(1e-12),
        "icp_pyramid_levels": NumericRange(1, 4, whole=True),
    }
    parameter_help = {
        "scene_sample_step": "Native PPF scene sampling ratio.",
        "scene_distance": "Native PPF relative scene distance step.",
        "max_matches": "Maximum poses emitted after ranking.",
        "minimum_votes": "Ignore native candidates with fewer votes.",
        "ranking": "Sort by native votes or refine and check support in the scene.",
        "candidate_limit": "Highest-vote candidates admitted to geometric ranking.",
        "refinement_passes": "Maximum number of local ICP refinements with refreshed pairs.",
        "refinement_distance": "Radius in cloud units for reciprocal pairs supplied to ICP.",
        "correspondence_radius": "Distance in cloud units for independent model coverage.",
        "minimum_model_overlap": "Minimum fraction of model points supported by the scene.",
        "minimum_inliers": "Minimum reciprocal nearest pairs.",
        "maximum_inlier_rmse": "Maximum geometric inlier RMSE in cloud units.",
        "duplicate_translation": "Maximum model-center separation in cloud units for duplicates.",
        "duplicate_rotation_deg": "Maximum relative angle in degrees for duplicate poses.",
        "icp_iterations": "Maximum native ICP iterations for each candidate.",
        "icp_tolerance": "Native ICP convergence tolerance.",
        "icp_rejection_scale": "Native ICP outlier rejection scale.",
        "icp_pyramid_levels": "Number of native ICP pyramid levels.",
    }
    output_help = {
        "poses": "Accepted source-to-scene poses in ranked order.",
        "pose": "Highest-ranked accepted pose.",
        "count": "Number of distinct emitted poses.",
        "report": "Native vote counts and, in geometric mode, candidate and refinement details.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._detector, self._model_cloud, self._key = None, None, None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        p = self.parameters if parameters is None else parameters
        super().validate_parameters(p)
        if p["ranking"] == "refined_geometry":
            radius = float(p["correspondence_radius"])
            if radius > float(p["refinement_distance"]):
                raise ValidationError(
                    "Correspondence radius must fit inside the refinement distance",
                    "correspondence_radius",
                    "refinement_distance",
                )
            if float(p["maximum_inlier_rmse"]) > radius:
                raise ValidationError(
                    "Maximum inlier RMSE must be at most the correspondence radius",
                    "maximum_inlier_rmse",
                    "correspondence_radius",
                )

    def process(self, inputs):
        model, cloud, p = inputs["model"], inputs["scene"], self.parameters
        if not isinstance(model, dict) or model.get("model") != "ppf_surface_v1":
            raise ValidationError("Connect a trained 3D PPF surface model", "model")
        if cloud.units != model.get("units"):
            raise ValidationError("PPF scene/model units differ", "model", "scene")
        if len(cloud.points) > 20000:
            raise ValidationError("PPF scenes are limited to 20000 oriented points", "scene")
        report = {
            "ranking": p["ranking"],
            "units": cloud.units,
            "scene_points": len(cloud.points),
            "native_candidates": None,
            "eligible_candidates": None,
            "evaluated_candidates": [],
        }
        outputs = dict(poses=[], pose=NO_VALUE, count=0, report=report)

        def cancelled():
            return Result(
                outputs,
                passed=False,
                message="PPF matching cancelled",
                fail_reasons=["ppf_cancelled"],
            )

        if self._cancelled():
            return cancelled()
        key = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        if key != self._key:
            self._detector, self._model_cloud = trained_ppf(model)
            self._key = key
        if self._cancelled():
            return cancelled()
        found = self._detector.match(
            cloud_with_normals(cloud), float(p["scene_sample_step"]), float(p["scene_distance"])
        )
        selected = sorted(
            (pose for pose in found if pose.numVotes >= p["minimum_votes"]),
            key=lambda obj: -obj.numVotes,
        )
        report.update(native_candidates=len(found), eligible_candidates=len(selected))
        if self._cancelled():
            return cancelled()
        graphics, reasons = {}, []
        if p["ranking"] == "refined_geometry" and selected:
            icp = surface_module().ICP(
                int(p["icp_iterations"]),
                float(p["icp_tolerance"]),
                float(p["icp_rejection_scale"]),
                int(p["icp_pyramid_levels"]),
            )
            limit = int(p["candidate_limit"])
            candidates = [
                {"matrix": pose.pose, "votes": int(pose.numVotes), "candidate": i}
                for i, pose in enumerate(selected[:limit], start=1)
            ]
            ranked = rank_surface_candidates(
                candidates,
                self._model_cloud,
                cloud,
                p,
                icp,
                self._cancelled,
            )
            if ranked is None or self._cancelled():
                return cancelled()
            poses, report["evaluated_candidates"] = ranked
            if not poses:
                reasons = ["ppf_geometry_rejected"]
            elif self.capture_graphics:
                graphics["Ranked surface matches"] = render_matches(cloud, self._model_cloud, poses)
        else:
            poses = [
                pose_record(
                    pose.pose, model["frame"], cloud.frame, cloud.units, votes=int(pose.numVotes)
                )
                for pose in selected[: int(p["max_matches"])]
            ]
        if self._cancelled():
            return cancelled()
        outputs.update(poses=poses, pose=poses[0] if poses else NO_VALUE, count=len(poses))
        return Result(
            outputs,
            passed=bool(poses),
            graphics=graphics,
            fail_reasons=[] if poses else reasons or ["ppf_surface_match"],
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
