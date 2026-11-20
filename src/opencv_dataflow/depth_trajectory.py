"""Track a bounded depth segment with odometry and fused surface checks."""

from uuid import uuid4

import cv2
import numpy as np

from .calibration_revisions import REVISION, check_capture, is_full_grid
from .core import Port, Result
from .geometry3d_common import Tool3D, pose_record, require_pinhole, rigid
from .images import DepthArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, frame
from .ptcloud3d import FuseTSDF, RGBDOdometry, cpu_tsdf_volume, make_odometry
from .stereo3d import metric_depth

_MAX_BATCH_PIXELS = 16_777_216
_ODOMETRY_KEYS = (
    "algorithm",
    "input_units",
    "units_per_sample",
    "iteration_counts",
    "min_gradient_magnitudes",
    "min_depth",
    "max_depth",
    "max_depth_difference",
    "max_points_fraction",
    "max_translation_m",
    "max_rotation_degrees",
    "normal_window",
)
_VOLUME_KEYS = (
    "resolution",
    "voxel_size_m",
    "volume_origin_m",
    "truncation_m",
    "max_weight",
    "raycast_step_factor",
)


def _check_frames(group, model, size, maximum, camera_frame):
    if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= maximum:
        raise ValidationError(f"Batch an ordered group of 2–{maximum} depth frames")
    width, height = size
    if width * height * len(group.items) > _MAX_BATCH_PIXELS:
        raise ValidationError("Depth batch exceeds 16,777,216 input pixels")
    recording = []
    for packet in group.items:
        depth = packet.value
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype not in (np.uint16, np.float32, np.float64)
            or depth.shape != (height, width)
        ):
            raise ValidationError(
                "Depth frames must be scalar planes matching the camera dimensions"
            )
        payload = packet.payload
        check_capture(model.get(REVISION), payload, size)
        if "spatial" in payload and not is_full_grid(payload, size):
            raise ValidationError(
                "Depth trajectory requires full camera frames at calibrated scale"
            )
        if payload.get("optical_frame", camera_frame) != camera_frame:
            raise ValidationError("Depth optical frame differs from the camera model")
        recording.append(payload.get("recorded_rgbd"))
    if any(stamp is not None for stamp in recording):
        if any(
            not isinstance(stamp, dict)
            or not isinstance(stamp.get("clock"), str)
            or type(stamp.get("pair_index")) is not int
            or type(stamp.get("timestamp_ns")) is not int
            for stamp in recording
        ):
            raise ValidationError("Recorded frames need a clock, timestamp and pair index")
        if len({stamp["clock"] for stamp in recording}) != 1 or any(
            a["pair_index"] >= b["pair_index"] or a["timestamp_ns"] >= b["timestamp_ns"]
            for a, b in zip(recording, recording[1:], strict=False)
        ):
            raise ValidationError(
                "Recordings must use one clock and strictly increasing frame order"
            )
    return recording


def _raycast_depth(volume, pose):
    points, _ = volume.raycast(pose.astype(np.float32))
    return points[..., 2].copy()


def _surface_check(volume, pose, depth, valid, row, p):
    surface = _raycast_depth(volume, pose)
    residual = np.abs(surface - depth)
    supported = valid & (surface > 0) & (residual <= float(p["model_depth_tolerance_m"]))
    count = int(supported.sum())
    overlap = count / row["valid_pixels"]
    rms = float(np.sqrt(np.mean(residual[supported] ** 2))) if count else None
    row.update(model_support=overlap, model_rms_m=rms)
    return (
        count > 0
        and overlap >= float(p["minimum_model_overlap"])
        and rms <= float(p["maximum_model_rms_m"])
    )


def _track_frame(odometry, volume, last_depth, last_pose, depth, valid, row, p):
    if last_depth is None:
        pose = np.eye(4)
        volume.integrate(depth, pose.astype(np.float32))
        if _surface_check(volume, pose, depth, valid, row, p):
            row["status"] = "anchored"
            return pose
        volume.reset()
        row["status"] = "seed_surface_unavailable"
        return None

    # OpenCV returns current-to-reference when the current image is the source.
    converged, step = odometry.compute(depth, last_depth)
    if not converged:
        row["status"] = "odometry_failed"
        return None
    try:
        step = rigid(step)
    except ValueError:
        row["status"] = "invalid_odometry"
        return None
    distance = float(np.linalg.norm(step[:3, 3]))
    angle = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(step[:3, :3])[0])))
    row.update(step_translation_m=distance, step_rotation_deg=angle)
    if distance > float(p["max_translation_m"]) or angle > float(p["max_rotation_degrees"]):
        row["status"] = "step_limit"
        return None
    pose = last_pose @ step
    if not _surface_check(volume, pose, depth, valid, row, p):
        row["status"] = "surface_mismatch"
        return None
    volume.integrate(depth, pose.astype(np.float32))
    row["status"] = "tracked"
    return pose


class TrackDepthFusion(Tool3D):
    type_id, title, subtype = (
        "3d.track_depth_fusion",
        "Track depth trajectory with TSDF",
        "Volumetric reconstruction",
    )
    inputs = (Port("frames", "any"), Port("camera", "data"))
    outputs = (
        Port("frames", "any", required=False),
        Port("poses", "data", required=False, visual_type="pose"),
        Port("trajectory", "data"),
        Port("surface_depth", "depth", required=False),
    )
    flow_primary = ("frames", "poses", "surface_depth")
    payload_reference = "frames"
    defaults = {
        **{name: RGBDOdometry.defaults[name] for name in _ODOMETRY_KEYS},
        **{name: FuseTSDF.defaults[name] for name in _VOLUME_KEYS},
        "iteration_counts": [8, 8, 10],
        "min_gradient_magnitudes": [10.0, 10.0, 10.0],
        "max_points_fraction": 0.4,
        "world_frame": "depth_trajectory",
        "maximum_frames": 32,
        "minimum_tracked_frames": 2,
        "maximum_consecutive_losses": 2,
        "minimum_valid_fraction": 0.2,
        "minimum_model_overlap": 0.5,
        "model_depth_tolerance_m": 0.075,
        "maximum_model_rms_m": 0.045,
    }
    parameter_choices = {
        name: RGBDOdometry.parameter_choices[name]
        for name in ("algorithm", "normal_window", "input_units")
    }
    parameter_ranges = {
        "maximum_frames": NumericRange(2, 64, whole=True),
        "minimum_tracked_frames": NumericRange(2, 64, whole=True),
        "maximum_consecutive_losses": NumericRange(1, 64, whole=True),
        "minimum_valid_fraction": NumericRange(0.01, 1),
        "minimum_model_overlap": NumericRange(0.01, 1),
        "model_depth_tolerance_m": NumericRange(0.001, 1),
        "maximum_model_rms_m": NumericRange(0.001, 1),
        "min_depth": NumericRange(0.001),
        "max_depth": NumericRange(0.001, 1000),
        "max_depth_difference": NumericRange(0.0001, 10),
        "max_points_fraction": NumericRange(0.0001, 1),
        "max_translation_m": NumericRange(0.0001),
        "max_rotation_degrees": NumericRange(0.0001, 180),
        "voxel_size_m": NumericRange(0.00001),
        "truncation_m": NumericRange(0.00001),
        "max_weight": NumericRange(1, 255, whole=True),
        "raycast_step_factor": NumericRange(0.001),
        "units_per_sample": NumericRange(0.000000001),
    }
    parameter_help = {
        "world_frame": "Name prefix for a new world frame anchored at the first accepted exposure.",
        "maximum_frames": "Maximum replay frames per segment, including rejected frames.",
        "minimum_tracked_frames": "Include the anchor in this minimum accepted frame count.",
        "maximum_consecutive_losses": "Stop after this many consecutive rejected frames.",
        "minimum_valid_fraction": "Minimum fraction of pixels inside the tracking depth range.",
        "minimum_model_overlap": "Minimum fraction of valid depths supported by the fused surface.",
        "model_depth_tolerance_m": "Depth residual tolerance for a pixel to count as supported.",
        "maximum_model_rms_m": "Maximum RMS metric depth residual over supported pixels.",
        "resolution": "The fixed XYZ grid uses multiples of eight and up to 8 million voxels.",
        "volume_origin_m": "XYZ origin of the fixed grid in the segment reference frame.",
        "voxel_size_m": "Fixed TSDF voxel side length in meters.",
        "truncation_m": "Signed distance truncation in meters.",
        "max_weight": "Cap on the accumulated weight of each native voxel.",
        "raycast_step_factor": "Ray marching step as a fraction of the truncation distance.",
        "iteration_counts": "Iteration limits per odometry pyramid level.",
        "min_gradient_magnitudes": "One gradient setting per pyramid level, as required by OpenCV.",
        "min_depth": "Minimum usable camera-Z in meters.",
        "max_depth": "Maximum usable and integrated camera-Z in meters.",
        "max_depth_difference": "Native odometry depth correspondence distance in meters.",
        "max_points_fraction": "Fraction of samples used by OpenCV depth odometry.",
        "max_translation_m": "Largest accepted translation from the last tracked frame in meters.",
        "max_rotation_degrees": "Largest accepted rotation from the last tracked frame in degrees.",
        "input_units": "Units for untagged depth planes. Tagged depth retains its units.",
        "units_per_sample": "Scale of integer inputs in their declared depth units per sample.",
    }
    input_help = {
        "frames": "An ordered FlowGroup of depth planes, for example from Batch serial entries.",
        "camera": "Fixed undistorted pinhole intrinsics on the recording depth grid.",
    }
    preview_safe = False
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        name = p["world_frame"]
        if not isinstance(name, str) or not name.strip() or len(name) > 64:
            raise ValidationError("Use a nonempty frame name up to 64 characters", "world_frame")
        if int(p["minimum_tracked_frames"]) > int(p["maximum_frames"]):
            raise ValidationError(
                "Minimum tracked frames exceeds the frame limit", "minimum_tracked_frames"
            )
        if float(p["maximum_model_rms_m"]) > float(p["model_depth_tolerance_m"]):
            raise ValidationError(
                "Model RMS limit must fit inside the depth tolerance", "maximum_model_rms_m"
            )
        if float(p["min_depth"]) >= float(p["max_depth"]):
            raise ValidationError(
                "Maximum depth must exceed minimum depth", "min_depth", "max_depth"
            )

    def process(self, inputs):
        p, group, camera = self.parameters, inputs["frames"], inputs["camera"]
        k, distortion, size = require_pinhole(camera)
        if np.any(distortion):
            raise ValidationError("Use an undistorted depth camera")
        source_frame = camera.get("frame", "camera")
        timestamps = _check_frames(group, camera, size, int(p["maximum_frames"]), source_frame)
        world = f"{p['world_frame']}:{uuid4().hex}"
        maximum = float(p["max_depth"])
        poses, accepted, labels, rows = [], [], [], []
        last_pose = last_depth = None
        consecutive = 0
        distance = 0.0
        limit = int(p["maximum_consecutive_losses"])
        exhausted = False
        with cpu_tsdf_volume(p, k, size, maximum) as volume:
            odometry = make_odometry({**p, "mode": "depth"}, k, size)
            for index, (packet, label, stamp) in enumerate(
                zip(group.items, group.labels, timestamps, strict=True)
            ):
                row = {"index": index, "source": label}
                if "ticket" in packet.payload:
                    row["ticket"] = packet.payload["ticket"]
                if stamp is not None:
                    row.update(
                        recording_index=stamp["pair_index"], timestamp_ns=stamp["timestamp_ns"]
                    )
                rows.append(row)
                if self._cancelled():
                    row["status"] = "cancelled"
                    break
                if exhausted:
                    row["status"] = "loss_limit_reached"
                    continue
                depth = metric_depth(packet.value, p["units_per_sample"], p["input_units"])
                valid = (depth >= float(p["min_depth"])) & (depth <= maximum)
                row["valid_pixels"] = int(valid.sum())
                pose = None
                if row["valid_pixels"] < max(16, depth.size * float(p["minimum_valid_fraction"])):
                    row["status"] = "insufficient_depth"
                else:
                    depth = np.where(valid, depth, np.float32(0))
                    pose = _track_frame(
                        odometry, volume, last_depth, last_pose, depth, valid, row, p
                    )
                if pose is None:
                    consecutive += 1
                    exhausted = consecutive >= limit
                    continue
                row["recovered"] = bool(consecutive and last_pose is not None)
                record = pose_record(pose, source_frame, world, "m", source_index=index)
                row["pose"] = record
                poses.append(record)
                accepted.append(packet)
                labels.append(label)
                distance += row.get("step_translation_m", 0.0)
                last_pose, last_depth, consecutive = pose, depth, 0

            completed = (
                len(rows) == len(group.items)
                and not exhausted
                and "pose" in rows[-1]
                and len(poses) >= int(p["minimum_tracked_frames"])
            )
            surface = NO_VALUE
            if completed:
                depth = _raycast_depth(volume, last_pose)
                depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
                surface = DepthArray(depth, "m", 0.0, maximum)

        report = {
            "model": "depth_fusion_trajectory_v1",
            "source_frame": source_frame,
            "world_frame": world,
            "complete": completed,
            "accepted": len(poses),
            "rejected": len(rows) - len(poses),
            "path_length_m": distance,
            "tracking": rows,
        }
        payloads = {}
        if completed:
            payloads["surface_depth"] = {
                "spatial": frame(size, f"{world}:raycast"),
                "optical_frame": source_frame,
                "camera_to_world": poses[-1],
                **(
                    {"ticket": accepted[-1].payload["ticket"]}
                    if "ticket" in accepted[-1].payload
                    else {}
                ),
            }
        reasons = (
            []
            if completed
            else ["depth_trajectory_loss_limit" if exhausted else "depth_trajectory_incomplete"]
        )
        return Result(
            {
                "frames": FlowGroup(tuple(accepted), tuple(labels)) if completed else NO_VALUE,
                "poses": poses if completed else NO_VALUE,
                "trajectory": report,
                "surface_depth": surface,
            },
            payloads=payloads,
            passed=completed,
            score=len(poses) / len(group.items),
            fail_reasons=reasons,
            message=f"Tracked {len(poses)} of {len(group.items)} depth frames",
        )
