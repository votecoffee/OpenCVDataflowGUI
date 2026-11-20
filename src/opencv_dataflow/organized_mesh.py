"""Triangulate adjacent samples of a calibrated or organized depth scan."""

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration_revisions import REVISION
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image
from .images import ImageArray
from .meshes import MAX_FACES, MAX_VERTICES, TriangleMesh, render_wireframe
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import selection_mask
from .stereo3d import metric_depth, rgbd_function

# Each split covers the same four pixel corners (TL, TR, BL, BR).
_SPLITS = np.array([[[0, 2, 3], [0, 3, 1]], [[0, 2, 1], [1, 2, 3]]])
_DIAGONALS = np.array([[0, 3], [1, 2]])
_UV_CONVENTION = "pixel centers normalized to grid extent, V up"


def _check_grid(width, height):
    if min(width, height) < 2 or width * height > MAX_VERTICES:
        raise ValidationError("Source grid needs at least 2×2 and at most 1,000,000 samples")


def _depth_grid(depth, camera_model, inputs, p, payload):
    raw = np.asarray(depth)
    if raw.ndim != 2 or raw.dtype not in (np.uint16, np.float32, np.float64):
        raise ValidationError("Depth must be a uint16, float32 or float64 plane", "depth")
    height, width = depth.shape
    _check_grid(width, height)
    k, distortion, size = camera_for_image(camera_model, depth)
    if camera_model.get("lens", "pinhole") != "pinhole" or np.any(distortion):
        raise ValidationError("Use undistorted pinhole intrinsics", "camera")
    frame = camera_model.get("frame", "camera")
    if payload.get("optical_frame", frame) != frame:
        raise ValidationError("Depth optical frame differs from the camera model", "camera")
    with np.errstate(over="ignore", invalid="ignore"):
        z = metric_depth(depth, p["units_per_sample"], p["input_units"])
    if raw.dtype == np.uint16:
        z[raw == 65535] = np.nan
    mask = selection_mask(inputs, depth)
    if mask is not None:
        z[mask == 0] = np.nan
    xyz = rgbd_function("depthTo3d")(z, k)[..., :3].reshape(-1, 3)
    provenance = {"kind": "calibrated_depth", "camera_matrix": k.tolist()}
    if REVISION in camera_model:
        provenance[REVISION] = camera_model[REVISION]
    return xyz, z.ravel(), size, frame, np.zeros(3), provenance


def _cloud_grid(cloud, inputs):
    if cloud.pixels is None or cloud.source_size is None:
        raise ValidationError(
            "Cloud needs a source grid with one point per retained pixel", "cloud"
        )
    if cloud.units not in ("m", "mm"):
        raise ValidationError("Organized cloud requires physical units m or mm", "cloud")
    width, height = cloud.source_size
    _check_grid(width, height)
    x, y = cloud.pixels.T
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValidationError("An organized cloud pixel is outside its source grid", "cloud")
    indices = y * width + x
    if len(np.unique(indices)) != len(indices):
        raise ValidationError("Organized cloud contains multiple points at one pixel", "cloud")
    scale = 0.001 if cloud.units == "mm" else 1.0
    xyz = np.full((width * height, 3), np.nan, np.float64)
    xyz[indices] = cloud.points.astype(np.float64) * scale
    origin, forward = np.zeros(3), np.array([0.0, 0.0, 1.0])
    provenance = {"kind": "organized_cloud", "viewpoint": "cloud_optical_frame"}
    view = cloud.metadata.get("pcd_viewpoint")
    if view is not None:
        if not isinstance(view, dict) or (view.get("frame"), view.get("units")) != (
            cloud.frame,
            cloud.units,
        ):
            raise ValidationError("Cloud viewpoint frame or units differ from the cloud", "cloud")
        values = np.asarray(view.get("values"), np.float64)
        if values.shape != (7,) or not np.isfinite(values).all() or not np.any(values[3:]):
            raise ValidationError(
                "Cloud viewpoint needs finite XYZ and a nonzero quaternion", "cloud"
            )
        origin = values[:3] * scale
        if np.abs(origin).max() > np.finfo(np.float32).max:
            raise ValidationError("Cloud viewpoint is outside the supported metric range", "cloud")
        q = values[3:] / np.abs(values[3:]).max()
        forward = Rotation.from_quat(q, scalar_first=True).apply(forward)
        provenance.update(
            viewpoint="pcd_viewpoint",
            viewpoint_origin_m=origin.tolist(),
            viewpoint_wxyz=(q / np.linalg.norm(q)).tolist(),
        )
    z = (xyz - origin) @ forward
    if "mask" in inputs or "region" in inputs:
        grid = ImageArray(z.reshape(height, width), "gray", cloud.metadata)
        mask = selection_mask(inputs, grid)
        z[mask.ravel() == 0] = np.nan
    return xyz, z, indices, origin, provenance


def _face_gates(xyz, z, faces, jump, max_edge):
    za, zb, zc = np.moveaxis(z[faces], -1, 0)
    a, b, c = np.moveaxis(xyz[faces], -2, 0)
    ab, ac, bc = b - a, c - a, c - b
    ab2, ac2, bc2 = (np.einsum("...i,...i->...", d, d) for d in (ab, ac, bc))
    longest = np.maximum(np.maximum(ab2, ac2), bc2)
    cross = np.cross(ab, ac)
    area2 = np.einsum("...i,...i->...", cross, cross)
    keep = (
        np.isfinite(za)
        & np.isfinite(zb)
        & np.isfinite(zc)
        & (np.maximum(np.maximum(za, zb), zc) - np.minimum(np.minimum(za, zb), zc) <= jump)
        & (area2 > (1e-8 * longest) ** 2)
    )
    if max_edge:
        keep &= longest <= max_edge**2
    return keep


def _grid_faces(xyz, z, size, p, cancelled):
    width, height = size
    cells = (width - 1) * (height - 1)
    chunks, count = [], 0
    coverage = np.zeros(3, np.int64)
    for start in range(0, cells, 8192):
        if cancelled():
            return None, None, "cancelled"
        linear = np.arange(start, min(start + 8192, cells), dtype=np.int32)
        corner = linear + linear // (width - 1)
        quads = corner[:, None] + np.array([0, 1, width, width + 1], np.int32)
        faces = quads[:, _SPLITS]
        keep = _face_gates(
            xyz, z, faces, float(p["max_depth_jump_m"]), float(p["max_edge_length_m"])
        )
        totals = keep.sum(axis=2)
        diag = xyz[quads[:, _DIAGONALS]]
        vectors = diag[:, :, 0] - diag[:, :, 1]
        lengths = np.einsum("...i,...i->...", vectors, vectors)
        chosen = (
            (totals[:, 1] > totals[:, 0])
            | ((totals[:, 1] == totals[:, 0]) & (lengths[:, 1] < lengths[:, 0]))
        ).astype(np.intp)
        batch = np.arange(len(quads))
        accepted = faces[batch, chosen][keep[batch, chosen]]
        count += len(accepted)
        if count > p["max_faces"]:
            return None, None, "face_limit"
        chunks.append(accepted)
        coverage += np.bincount(totals[batch, chosen], minlength=3)
    return np.concatenate(chunks), coverage, None


def _mesh(xyz, grid_faces, size, units, frame, origin, rgb=None):
    width, height = size
    indices, inverse = np.unique(grid_faces.ravel(), return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    vertices = xyz[indices]
    a, b, c = np.moveaxis(vertices[faces], -2, 0)
    cross = np.cross(b - a, c - a)
    away = np.einsum("ij,ij->i", cross, a - origin) > 0
    faces[away] = faces[away][:, [0, 2, 1]]
    cross[away] *= -1
    normals = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(normals, corner, cross)
    lengths = np.linalg.norm(normals, axis=1)
    # In a pathological fold, opposed incident faces can cancel exactly.
    zero = lengths == 0
    if zero.any():
        fallback = np.empty_like(normals)
        for corner in faces.T:
            fallback[corner] = cross
        normals[zero] = fallback[zero]
        lengths[zero] = np.linalg.norm(normals[zero], axis=1)
    uv = np.column_stack(((indices % width + 0.5) / width, 1 - (indices // width + 0.5) / height))
    return TriangleMesh(
        vertices * (1000 if units == "mm" else 1),
        faces,
        units,
        frame,
        normals / lengths[:, None],
        rgb[indices] if rgb is not None else None,
        uv,
    )


class MeshOrganizedDepth(Tool3D):
    type_id = "3d.organized_mesh"
    title, subtype = "Mesh an organized depth grid", "Mesh geometry"
    description = (
        "Build a local surface from adjacent samples while excluding depth discontinuities."
    )
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = ""
    inputs = (
        Port("depth", "depth", required=False, required_when_connected=True),
        Port("camera", "data", required=False, required_when_connected=True),
        Port("cloud", "cloud", required=False, required_when_connected=True),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (Port("mesh", "mesh", required=False), Port("report", "data"))
    defaults = {
        "max_depth_jump_m": 0.02,
        "max_edge_length_m": 0.0,
        "max_faces": MAX_FACES,
        "input_units": "m",
        "units_per_sample": 0.001,
        "output_units": "m",
    }
    parameter_choices = {"input_units": ("m", "mm"), "output_units": ("m", "mm")}
    parameter_ranges = {
        "max_depth_jump_m": NumericRange(0, 1000),
        "max_edge_length_m": NumericRange(0, 1000),
        "max_faces": NumericRange(1, MAX_FACES, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "max_depth_jump_m": "Largest axial depth spread within one triangle, in meters.",
        "max_edge_length_m": "Longest XYZ edge allowed, in meters. Zero disables this test.",
        "max_faces": "Withhold the mesh if more than this many triangles pass the gates.",
        "input_units": "Physical units for an untagged depth plane.",
        "units_per_sample": "Length in input units per untagged uint16 sample.",
        "output_units": "Length units stored in the output mesh.",
    }
    input_help = {
        "depth": "Positive axial camera Z. Wire depth and camera, or wire an organized cloud.",
        "camera": "Matching undistorted pinhole intrinsics for depth input.",
        "cloud": "Metric organized XYZ with unique source pixels. PCD viewpoint is honored.",
        "mask": "Optional raster selection on the source grid.",
        "region": "Optional source-grid region intersected with the raster selection.",
    }
    output_help = {
        "mesh": "Faces, unit normals toward the camera, and source pixels encoded as UV.",
        "report": "Calibration, grid, triangle acceptance and units.",
    }

    def process(self, inputs):
        p = self.parameters
        has_cloud = "cloud" in inputs
        has_depth, has_camera = "depth" in inputs, "camera" in inputs
        if has_depth != has_camera or has_cloud == has_depth:
            raise ValidationError(
                "Connect depth and camera together, or an organized cloud",
                "depth",
                "camera",
                "cloud",
            )
        colors = None
        port = "cloud" if has_cloud else "depth"
        source_payload = self.input_payloads[port]
        if has_cloud:
            cloud = inputs["cloud"]
            xyz, z, indices, origin, provenance = _cloud_grid(cloud, inputs)
            size, frame = cloud.source_size, cloud.frame
            if cloud.colors is not None:
                colors = np.zeros((len(z), 3), np.uint8)
                colors[indices] = cloud.colors
        else:
            xyz, z, size, frame, origin, provenance = _depth_grid(
                inputs["depth"], inputs["camera"], inputs, p, source_payload
            )
        xyz = xyz.astype(np.float64, copy=False)
        scale = 1000 if p["output_units"] == "mm" else 1
        # Gate the geometry at the precision the mesh will actually store.
        with np.errstate(over="ignore", invalid="ignore"):
            xyz = (xyz * scale).astype(np.float32).astype(np.float64) / scale
        representable = np.isfinite(xyz).all(axis=1)
        z[~(representable & np.isfinite(z) & (z > 0))] = np.nan
        # Bound floating arithmetic even for invalid reference geometry.
        xyz[~np.isfinite(z)] = 0
        grid = {
            "source_grid": list(size),
            "frame": frame,
            "uv_convention": _UV_CONVENTION,
            **provenance,
        }
        report = {
            **grid,
            "units": p["output_units"],
            "valid_grid_samples": int(np.isfinite(z).sum()),
            "max_depth_jump_m": float(p["max_depth_jump_m"]),
            "max_edge_length_m": float(p["max_edge_length_m"]),
            "face_limit": int(p["max_faces"]),
            "vertex_count": 0,
            "face_count": 0,
        }
        payload = {key: source_payload[key] for key in ("ticket",) if key in source_payload}
        payloads = {"mesh": {**payload, "organized_mesh": grid}, "report": payload}
        faces, coverage, issue = _grid_faces(xyz, z, size, p, self._cancelled)
        if self._cancelled():
            issue = "cancelled"
        if issue is None:
            report.update(
                zip(("empty_cells", "partial_cells", "full_cells"), coverage.tolist(), strict=True)
            )
            if not len(faces):
                issue = "no_surface"
        if issue is not None:
            messages = {
                "cancelled": "Organized meshing cancelled",
                "face_limit": "Triangle budget exceeded. Use a smaller grid or selection.",
                "no_surface": "No adjacent triangles met the depth and edge limits",
            }
            return Result(
                {"mesh": NO_VALUE, "report": report},
                passed=False,
                fail_reasons=[f"organized_mesh_{issue}"],
                payloads=payloads,
                message=messages[issue],
            )
        mesh = _mesh(xyz, faces, size, p["output_units"], frame, origin, colors)
        report.update(vertex_count=len(mesh.vertices), face_count=len(mesh.faces))
        return Result(
            {"mesh": mesh, "report": report},
            payloads=payloads,
            graphics={"Mesh wireframe": render_wireframe(mesh)} if self.capture_graphics else {},
            message=f"{len(mesh.faces):,} triangles on the {size[0]}×{size[1]} source grid",
        )
