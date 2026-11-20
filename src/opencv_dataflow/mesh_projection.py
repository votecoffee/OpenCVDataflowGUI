"""Project triangle meshes with calibrated pixel centers and metric depth."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, applied_payload
from .cloud_projection import MAX_PIXELS, METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, require_pinhole, rigid
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame

_MAX_TRIANGLES = 250_000


def _camera_vertices(mesh, camera_frame, transform):
    if mesh.units not in METERS:
        raise ValidationError("Mesh projection requires m or mm", "mesh")
    xyz = mesh.vertices.astype(np.float64) * METERS[mesh.units]
    if transform is None:
        if mesh.frame != camera_frame:
            raise ValidationError("Connect a transform from the mesh frame to the camera")
        return xyz
    if (
        not isinstance(transform, dict)
        or transform.get("source_frame") != mesh.frame
        or transform.get("target_frame") != camera_frame
        or transform.get("units") not in METERS
    ):
        raise ValidationError("Transform must map the mesh frame to the camera in metric units")
    pose = rigid(transform)
    xyz = xyz @ pose[:3, :3].T + pose[:3, 3] * METERS[transform["units"]]
    if not np.isfinite(xyz).all():
        raise ValidationError("Transformed mesh coordinates must be finite", "transform")
    return xyz


def _view_planes(k, size, near, far):
    width, height = size
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5
    # Nonnegative means inside. Pixel centers are integers in the calibrated grid.
    return (
        (np.array([0, 0, 1]), -near),
        (np.array([0, 0, -1]), far),
        (np.array([fx, 0, cx]), 0.0),
        (np.array([-fx, 0, width - cx]), 0.0),
        (np.array([0, fy, cy]), 0.0),
        (np.array([0, -fy, height - cy]), 0.0),
    )


def _clip_polygon(polygon, plane, bias):
    distances = polygon[:, :3] @ plane + bias
    clipped = []
    for i, (row, distance) in enumerate(zip(polygon, distances, strict=True)):
        previous, last = polygon[i - 1], distances[i - 1]
        if (last < 0 < distance) or (distance < 0 < last):
            # The same interpolation carries position and optional vertex color.
            scale = max(abs(last), abs(distance))
            left, right = abs(last) / scale, abs(distance) / scale
            factor = left / (left + right)
            clipped.append((1 - factor) * previous + factor * row)
        if distance >= 0:
            clipped.append(row)
    return np.asarray(clipped, dtype=np.float64)


def _clipped_geometry(xyz, faces, colors, shading, planes, limit, cancelled):
    codes = np.zeros(len(xyz), np.uint8)
    for bit, (plane, bias) in enumerate(planes):
        distance = np.einsum("ij,j->i", xyz, plane) + bias
        if not np.isfinite(distance).all():
            raise ValidationError("Mesh and camera yield nonfinite frustum distances")
        codes[distance < 0] |= 1 << bit
    corners = codes[faces]
    shared = np.bitwise_and.reduce(corners, axis=1)
    crossing = np.bitwise_or.reduce(corners, axis=1)
    inside = np.flatnonzero(crossing == 0)
    partial = np.flatnonzero((shared == 0) & (crossing != 0))
    rows, inverse = np.unique(faces[inside], return_inverse=True)
    base = xyz[rows]
    if colors is not None:
        base = np.column_stack((base, colors[rows]))
    vertex_groups = [base]
    face_groups = [inverse.reshape(-1, 3)]
    owners = [inside]
    next_vertex, total = len(rows), len(inside)
    clipped_count = 0
    for cursor, owner in enumerate(partial):
        if not cursor % 128 and cancelled():
            return None, "cancelled"
        triangle = faces[owner]
        polygon = xyz[triangle]
        if colors is not None:
            rgb = colors[triangle]
            if shading == "flat":
                rgb = np.broadcast_to(rgb[0], rgb.shape)
            polygon = np.column_stack((polygon, rgb))
        for bit, (plane, bias) in enumerate(planes):
            if crossing[owner] & (1 << bit):
                polygon = _clip_polygon(polygon, plane, bias)
                if len(polygon) < 3:
                    break
        if len(polygon) < 3:
            continue
        count = len(polygon) - 2
        total += count
        if total > limit:
            return None, "triangle_budget"
        fan = np.arange(next_vertex + 1, next_vertex + len(polygon) - 1)
        face_groups.append(np.column_stack((np.full(count, next_vertex), fan, fan + 1)))
        vertex_groups.append(polygon)
        owners.append(np.full(count, owner))
        next_vertex += len(polygon)
        clipped_count += 1

    values = np.concatenate(vertex_groups)
    indices = np.concatenate(face_groups)
    order = np.argsort(np.concatenate(owners), kind="stable")
    return (values, indices[order], clipped_count), ""


def _projected_vertices(xyz, k, size):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        uv = xyz[:, :2] / xyz[:, 2, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    if not np.isfinite(uv).all() or np.any(xyz[:, 2] <= 0):
        raise ValidationError("Clipped mesh coordinates exceed the rasterizer's numerical range")
    upper = np.subtract(size, 0.5)
    if np.any(uv < -0.5001) or np.any(uv > upper + 0.0001):
        raise ValidationError("Clipped mesh coordinates exceed the rasterizer's numerical range")
    return np.clip(uv, -0.5, upper)


def _pixel_work(uv, faces, size):
    if not len(faces):
        return 0
    projected = uv[faces]
    # Include a one-pixel margin around native scanline rounding.
    low = np.maximum(np.floor(projected.min(axis=1)) - 1, 0)
    high = np.minimum(np.ceil(projected.max(axis=1)) + 1, np.subtract(size, 1))
    spans = np.maximum(high - low + 1, 0).astype(np.int64)
    return int(np.sum(spans[:, 0] * spans[:, 1], dtype=np.int64))


def _native_raster(values, uv, faces, size, near, far, culling, shading, mode):
    width, height = size
    # The host has already clipped in metric space. The native inverse buffer
    # compares an affine function of 1/Z. Using virtual planes at far and 2*far
    # keeps it near 0 (or negative), avoiding depth precision loss near 1.
    native_near, native_far = far, 2 * far
    depth = None if mode == "color_only" else np.ones((height, width), np.float32)
    # A negative RGB sentinel identifies uncovered pixels in native color-only mode.
    color = None if mode == "depth_only" else np.full((height, width, 3), -1, np.float32)
    if len(faces):
        # Native uses half-pixel screen edges and has +Y up, -Z forward.
        z = values[:, 2]
        xy = (uv + 0.5 - [width / 2, height / 2]) * [1, -1] * (z / (height / 2))[:, None]
        native = np.column_stack((xy, -z)).astype(np.float32)[:, None, :]
        indices = np.ascontiguousarray(faces, np.int32)[:, None, :]
        settings = cv2.TriangleRasterizeSettings()
        settings.setCullingMode(
            {
                "none": cv2.RASTERIZE_CULLING_NONE,
                "clockwise": cv2.RASTERIZE_CULLING_CW,
                "counterclockwise": cv2.RASTERIZE_CULLING_CCW,
            }[culling]
        )
        settings.setShadingType(
            {
                "white": cv2.RASTERIZE_SHADING_WHITE,
                "flat": cv2.RASTERIZE_SHADING_FLAT,
                "interpolated": cv2.RASTERIZE_SHADING_SHADED,
            }[shading]
        )
        settings.setGlCompatibleMode(cv2.RASTERIZE_COMPAT_INVDEPTH)
        pose = np.eye(4)
        args = pose, np.pi / 2, native_near, native_far, settings
        if color is None:
            depth = cv2.triangleRasterizeDepth(native, indices, depth, *args)
        else:
            rgb = (
                None
                if shading == "white"
                else np.ascontiguousarray(values[:, 3:], np.float32)[:, None, :]
            )
            if depth is None:
                color = cv2.triangleRasterizeColor(native, indices, rgb, color, *args)
            else:
                color, depth = cv2.triangleRasterize(native, indices, rgb, color, depth, *args)
    if depth is None:
        return None, color, color[:, :, 0] >= 0
    hits = np.isfinite(depth) & (depth < 1)
    depth[~hits] = np.nan
    np.subtract(2, depth, out=depth)
    np.divide(native_far, depth, out=depth)
    np.clip(depth, near, far, out=depth)
    return depth, color, hits


class ProjectMesh(Tool3D):
    type_id = "3d.mesh_project"
    title, subtype = "Project mesh to color and depth", "Mesh geometry"
    description = "Rasterize a metric triangle mesh onto a rectified calibrated camera grid."
    payload_reference = "mesh"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("mesh", "mesh"),
        Port("camera", "data"),
        Port("transform", "data", required=False, visual_type="pose", required_when_connected=True),
    )
    outputs = (
        Port("color", required=False),
        Port("depth", "depth", required=False),
        Port("coverage_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_z_m": 0.05,
        "maximum_z_m": 10.0,
        "output_mode": "color_and_depth",
        "culling": "none",
        "shading": "interpolated",
        "max_triangles": 100_000,
        "max_pixel_tests": 25_000_000,
    }
    parameter_choices = {
        "output_mode": ("color_and_depth", "color_only", "depth_only"),
        "culling": ("none", "clockwise", "counterclockwise"),
        "shading": ("interpolated", "flat", "white"),
    }
    parameter_ranges = {
        "minimum_z_m": NumericRange(1e-6, 100_000),
        "maximum_z_m": NumericRange(1e-6, 100_000),
        "max_triangles": NumericRange(1, _MAX_TRIANGLES, whole=True),
        "max_pixel_tests": NumericRange(1, 250_000_000, whole=True),
    }
    parameter_help = {
        "minimum_z_m": "Positive near clip plane in camera meters.",
        "maximum_z_m": "Far clip plane in camera meters.",
        "output_mode": "Select color, depth or both. All modes use nearest-depth visibility.",
        "culling": "Discard the selected winding as seen on the image. Image Y points down.",
        "shading": "Use interpolated RGB, the first triangle vertex RGB, or white.",
        "max_triangles": "Maximum input and clipped output triangle count.",
        "max_pixel_tests": "Conservative limit on candidate pixels across clipped triangle bounds.",
    }
    input_help = {
        "mesh": "Triangle mesh in meters or millimeters with optional per-vertex RGB.",
        "camera": "Rectified pinhole intrinsics, image size and named optical frame.",
        "transform": "Optional mesh-to-camera rigid transform in metric units.",
    }
    output_help = {
        "color": "RGB render with black background.",
        "depth": "Camera Z in meters as float32, with NaN for uncovered pixels.",
        "coverage_mask": "255 where a visible triangle was rasterized, zero elsewhere.",
        "report": "Grid, clipping, triangle work and coverage counts.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["maximum_z_m"]) <= float(p["minimum_z_m"]):
            raise ValidationError("Maximum Z must exceed minimum Z", "maximum_z_m", "minimum_z_m")

    def process(self, inputs):
        mesh, camera, p = inputs["mesh"], inputs["camera"], self.parameters
        k, distortion, size = require_pinhole(camera)
        if np.any(distortion):
            raise ValidationError("Use rectified pinhole intrinsics with zero distortion", "camera")
        if size[0] * size[1] > MAX_PIXELS:
            raise ValidationError("Target camera grid exceeds the pixel limit", "camera")
        camera_frame = camera.get("frame", "camera")
        near, far = float(p["minimum_z_m"]), float(p["maximum_z_m"])
        mode = p["output_mode"]
        shading = p["shading"] if mesh.colors is not None else "white"
        with_color = mode != "depth_only"
        limit = int(p["max_triangles"])
        report = {
            "camera_frame": camera_frame,
            "mesh_frame": mesh.frame,
            "image_size": list(size),
            "minimum_z_m": near,
            "maximum_z_m": far,
            "output_mode": mode,
            "culling": p["culling"],
            "shading": shading,
            "input_triangles": len(mesh.faces),
            "raster_triangles": 0,
            "covered_pixels": 0,
        }

        def failure(code, message):
            return Result(
                {"color": NO_VALUE, "depth": NO_VALUE, "coverage_mask": NO_VALUE, "report": report},
                passed=False,
                fail_reasons=[f"mesh_projection_{code}"],
                message=message,
            )

        if len(mesh.faces) > limit:
            return failure("triangle_budget", "Mesh exceeds the configured triangle limit")
        if self._cancelled():
            return failure("cancelled", "Mesh projection cancelled")
        mesh.validate()
        xyz = _camera_vertices(mesh, camera_frame, inputs.get("transform"))
        colors = mesh.colors / 255.0 if with_color and shading != "white" else None
        geometry, reason = _clipped_geometry(
            xyz,
            mesh.faces,
            colors,
            shading,
            _view_planes(k, size, near, far),
            limit,
            self._cancelled,
        )
        if reason == "triangle_budget":
            return failure(reason, "Clipped mesh exceeds the configured triangle limit")
        if reason == "cancelled" or self._cancelled():
            return failure("cancelled", "Mesh projection cancelled")
        values, faces, partial = geometry
        uv = _projected_vertices(values[:, :3], k, size)
        work = _pixel_work(uv, faces, size)
        report.update(
            raster_triangles=len(faces), clipped_source_triangles=partial, pixel_tests=work
        )
        if work > p["max_pixel_tests"]:
            return failure("pixel_budget", "Mesh exceeds the configured pixel work limit")
        if self._cancelled():
            return failure("cancelled", "Mesh projection cancelled")
        depth, color, hits = _native_raster(
            values,
            uv,
            faces,
            size,
            near,
            far,
            p["culling"],
            shading,
            mode,
        )
        if self._cancelled():
            return failure("cancelled", "Mesh projection cancelled")
        output_color = NO_VALUE
        if color is not None:
            color[~hits] = 0
            output_color = ImageArray(np.rint(np.clip(color, 0, 1) * 255).astype(np.uint8), "rgb")
        mask = hits.astype(np.uint8) * 255
        output_depth = DepthArray(depth, "m", near, far) if mode != "color_only" else NO_VALUE
        count = int(np.count_nonzero(hits))
        report["covered_pixels"] = count
        target = applied_payload(
            {"spatial": frame(size, f"{self.id}:target")}, camera.get(REVISION)
        )
        outputs = {
            "color": output_color,
            "depth": output_depth,
            "coverage_mask": mask,
            "report": report,
        }
        return Result(
            outputs,
            payloads={
                name: target for name, value in outputs.items() if isinstance(value, np.ndarray)
            },
            passed=bool(count),
            score=count / hits.size,
            fail_reasons=[] if count else ["mesh_projection_empty"],
            message=f"{count} covered pixels from {len(faces)} raster triangles",
        )
