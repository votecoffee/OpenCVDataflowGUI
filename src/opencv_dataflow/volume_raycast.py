"""Render a native TSDF in a separately calibrated target camera."""

from dataclasses import dataclass

import numpy as np

from .cloud_projection import METERS
from .core import Port
from .geometry3d_common import require_pinhole, rigid
from .images import DepthArray, ImageArray
from .parameter_validation import ValidationError
from .payloads import frame

RAYCAST_INPUTS = (
    Port("target_camera", "data", required=False, required_when_connected=True),
    Port("target_pose", "data", required=False, required_when_connected=True, visual_type="pose"),
)
RAYCAST_INPUT_HELP = {
    "target_camera": "Undistorted pinhole intrinsics and raster size for the requested view.",
    "target_pose": "Rigid target-camera-to-world transform in meters or millimeters.",
}
RAYCAST_PLANES = (
    Port("normals", required=False),
    Port("valid_mask", required=False, visual_type="region"),
)
RAYCAST_OUTPUT_HELP = {
    "depth": "Metric Z in the rendered camera. Missing rays are NaN.",
    "normals": "Float XYZ unit normals in the target camera axes. Missing rays are NaN.",
    "valid_mask": "255 for valid rays and zero otherwise.",
}
MAX_RAYCAST_PIXELS = 2_097_152


@dataclass(frozen=True)
class RaycastTarget:
    camera_matrix: np.ndarray
    size: tuple[int, int]
    camera_to_world: np.ndarray
    optical_frame: str
    world_frame: str

    @property
    def pixels(self):
        return self.size[0] * self.size[1]

    def context(self):
        return {
            "optical_frame": self.optical_frame,
            "world_frame": self.world_frame,
            "camera_to_world_m": self.camera_to_world.tolist(),
            "camera_matrix": self.camera_matrix.tolist(),
            "size": list(self.size),
        }


def target_raycast(inputs, world_frame):
    if "target_camera" not in inputs and "target_pose" not in inputs:
        return None
    if "target_camera" not in inputs or "target_pose" not in inputs:
        raise ValidationError(
            "Connect both target camera and target pose", "target_camera", "target_pose"
        )
    camera, pose = inputs["target_camera"], inputs["target_pose"]
    k, distortion, size = require_pinhole(camera)
    if np.any(distortion) or min(size) < 2 or size[0] * size[1] > MAX_RAYCAST_PIXELS:
        raise ValidationError(
            "Target camera needs an undistorted pinhole grid with 2×2 to 2,097,152 pixels",
            "target_camera",
        )
    if np.max(np.abs(k)) > 1e8 or np.min(k.diagonal()[:2]) < 1e-6:
        raise ValidationError("Target intrinsics exceed the native raycast range", "target_camera")
    optical_frame = camera.get("frame", "camera")
    if (
        not isinstance(pose, dict)
        or pose.get("source_frame") != optical_frame
        or pose.get("target_frame") != world_frame
        or pose.get("units") not in METERS
    ):
        raise ValidationError(
            "Target pose must map the target optical frame to the volume world frame in m or mm",
            "target_pose",
        )
    transform = rigid(pose).copy()
    transform[:3, 3] *= METERS[pose["units"]]
    if np.max(np.abs(transform[:3, 3])) > 100_000:
        raise ValidationError("Target pose translation must be within 100 km", "target_pose")
    return RaycastTarget(
        k.astype(np.float32), size, transform.astype(np.float32), optical_frame, world_frame
    )


def render_raycast(volume, view, block_id, display_max, *, color=False, ticket=None):
    width, height = view.size
    args = view.camera_to_world, height, width, view.camera_matrix
    render = volume.raycastExColor(*args) if color else volume.raycastEx(*args)
    xyz, normal = (image[..., :3] for image in render[:2])
    lengths = np.linalg.norm(normal, axis=2)
    z = xyz[..., 2]
    valid = np.isfinite(xyz).all(axis=2) & (z > 0)
    valid &= np.isfinite(normal).all(axis=2) & np.isfinite(lengths) & (lengths > 1e-8)
    if color:
        rgb = render[2][..., :3]
        valid &= np.isfinite(rgb).all(axis=2)
    depth = np.full((height, width), np.nan, np.float32)
    depth[valid] = z[valid]
    normals = np.full((height, width, 3), np.nan, np.float32)
    normals[valid] = normal[valid] / lengths[valid, None]
    outputs = {
        "depth": DepthArray(depth, "m", 0, display_max),
        "normals": ImageArray(normals, "rgb"),
        "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
    }
    if color:
        pixels = np.zeros((height, width, 3), np.uint8)
        pixels[valid] = np.rint(np.clip(rgb[valid], 0, 255)).astype(np.uint8)
        outputs["color"] = ImageArray(pixels, "rgb")
    context = view.context()
    payload = {
        "spatial": frame(view.size, f"{block_id}:target_raycast"),
        "optical_frame": view.optical_frame,
        "tsdf_raycast": context,
    }
    if ticket is not None:
        payload["ticket"] = ticket
    payloads = dict.fromkeys(outputs, payload)
    payloads["normals"] = {**payload, "normal_components": ["x", "y", "z"]}
    count = int(np.count_nonzero(valid))
    report = dict(context, valid_pixels=count, coverage=count / view.pixels, has_color=color)
    return outputs, payloads, report
