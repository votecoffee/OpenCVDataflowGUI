"""Bake visible samples from a calibrated image into a mesh's UV atlas."""

import numpy as np

from .calibration_revisions import REVISION, applied_payload
from .cloud_projection import MAX_PIXELS
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image
from .images import ImageArray, image_format
from .mesh_projection import (
    _MAX_TRIANGLES,
    _camera_vertices,
    _clipped_geometry,
    _native_raster,
    _pixel_work,
    _projected_vertices,
    _view_planes,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .selection import selection_mask

_MAX_ATLAS_PIXELS = 4_194_304
_CHUNK = 32768
_ENCODINGS = {"gray": 1, "rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4}


def _edge(a, b, xy):
    offset = a[..., 0] * b[..., 1] - b[..., 0] * a[..., 1]
    return (a[..., 1] - b[..., 1]) * xy[..., 0] + (b[..., 0] - a[..., 0]) * xy[..., 1] + offset


def _atlas_samples(uv, low, high, area):
    """Yield texel centers and barycentric weights with half-open shared edges."""
    span = high - low + 1
    count = int(span[0]) * int(span[1])
    for start in range(0, count, _CHUNK):
        rows = np.arange(start, min(start + _CHUNK, count))
        y, x = np.divmod(rows, span[0])
        xy = np.column_stack((x, y)) + low
        barycentric = []
        owned = np.ones(len(xy), bool)
        for a, b in ((uv[1], uv[2]), (uv[2], uv[0]), (uv[0], uv[1])):
            edge = _edge(a, b, xy)
            closed = b[1] < a[1] or (b[1] == a[1] and b[0] > a[0])
            owned &= (edge >= 0) if closed else (edge > 0)
            barycentric.append(edge)
        if owned.any():
            yield xy[owned], np.column_stack(barycentric)[owned] / area


def _donors(uv, sampling):
    if sampling == "nearest":
        yield np.floor(uv + 0.5).astype(np.int32), np.ones(len(uv))
        return
    # Suppress insignificant fractional weights caused by projection roundoff.
    rounded = np.rint(uv)
    uv = np.where(np.abs(uv - rounded) <= 1e-7, rounded, uv)
    low = np.floor(uv).astype(np.int32)
    dx, dy = (uv - low).T
    yield low, (1 - dx) * (1 - dy)
    yield low + [1, 0], dx * (1 - dy)
    yield low + [0, 1], (1 - dx) * dy
    yield low + [1, 1], dx * dy


def _sample_image(points, triangle, image, source_depth, selection, k, near, far, p):
    """Test each active donor against the depth of this face on that pixel ray."""
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    height, width = image.shape[:2]
    uv = np.full((len(points), 2), -1.0)
    valid = (points[:, 2] >= near) & (points[:, 2] <= far)
    uv[valid] = points[valid, :2] / points[valid, 2, None] * [fx, fy] + [cx, cy]
    valid &= (uv >= -0.5).all(axis=1) & (uv < [width - 0.5, height - 0.5]).all(axis=1)
    uv[~valid] = -1
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    plane = float(normal @ triangle[0])
    texels = np.zeros((len(uv),) + image.shape[2:], dtype=np.float64)
    for xy, weight in _donors(uv, p["sampling"]):
        active = np.flatnonzero(valid & (weight > 0))
        x, y = xy[active].T
        in_bounds = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        valid[active[~in_bounds]] = False
        active = active[in_bounds]
        if not len(active):
            continue
        x, y = xy[active].T
        denominator = (x - cx) * (normal[0] / fx) + (y - cy) * (normal[1] / fy) + normal[2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            expected = plane / denominator
            limit = p["visibility_tolerance_m"] + np.abs(expected) * (8 * np.finfo(np.float32).eps)
            visible = np.isfinite(expected) & (expected > 0)
            visible &= np.abs(expected - source_depth[y, x]) <= limit
        pixels = image[y, x]
        visible &= np.isfinite(pixels).reshape(len(active), -1).all(axis=1)
        if selection is not None:
            visible &= selection[y, x] != 0
        if image.ndim == 3 and image.shape[2] == 4:
            visible &= pixels[:, 3] > 0
        valid[active[~visible]] = False
        rows = active[visible]
        scale = weight[rows, None] if image.ndim == 3 else weight[rows]
        with np.errstate(over="ignore", invalid="ignore"):
            texels[rows] += pixels[visible] * scale
    valid &= np.isfinite(texels).reshape(len(uv), -1).all(axis=1)
    samples = texels[valid]
    if image.dtype.kind == "u":
        samples = np.rint(samples)
    return valid, samples.astype(image.dtype)


class BakeMeshTexture(Tool3D):
    type_id = "3d.mesh_texture"
    title, subtype = "Bake camera image to mesh UVs", "Mesh geometry"
    description = "Transfer visible samples from a calibrated image to an existing mesh UV atlas."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        Port("mesh", "mesh"),
        Port("camera", "data"),
        Port("transform", "data", required=False, visual_type="pose", required_when_connected=True),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (
        Port("texture", required=False),
        Port("texture_mask", required=False, visual_type="region"),
        Port("mesh", "mesh", required=False),
        Port("report", "data"),
    )
    defaults = {
        "texture_width": 512,
        "texture_height": 512,
        "sampling": "bilinear",
        "minimum_z_m": 0.05,
        "maximum_z_m": 10.0,
        "visibility_tolerance_m": 0.001,
        "minimum_coverage": 0.0,
        "max_triangles": 100_000,
        "max_pixel_tests": 25_000_000,
    }
    parameter_choices = {"sampling": ("nearest", "bilinear")}
    parameter_ranges = {
        "texture_width": NumericRange(1, 4096, whole=True),
        "texture_height": NumericRange(1, 4096, whole=True),
        "minimum_z_m": NumericRange(1e-6, 100_000),
        "maximum_z_m": NumericRange(1e-6, 100_000),
        "visibility_tolerance_m": NumericRange(0, 1000),
        "minimum_coverage": NumericRange(0, 1),
        "max_triangles": NumericRange(1, _MAX_TRIANGLES, whole=True),
        "max_pixel_tests": NumericRange(1, 250_000_000, whole=True),
    }
    parameter_help = {
        "texture_width": "Width of the destination atlas in texels.",
        "texture_height": "Height of the destination atlas in texels.",
        "sampling": "In bilinear mode, every contributing pixel must be visible and selected.",
        "minimum_z_m": "Positive near clip distance along camera Z, in meters.",
        "maximum_z_m": "Far clip distance along camera Z, in meters.",
        "visibility_tolerance_m": "Absolute depth agreement tolerance at each source pixel center.",
        "minimum_coverage": "Required fraction of rasterized UV texels with valid image support.",
        "max_triangles": "Maximum input triangle count and clipped visibility triangle count.",
        "max_pixel_tests": "Combined conservative camera and atlas triangle pixel-work limit.",
    }
    input_help = {
        "image": "Rectified gray or RGB-family image registered to the camera.",
        "mesh": "Metric triangle mesh with normalized non-overlapping UVs, V up.",
        "camera": "Rectified pinhole intrinsics and named frame for the source image.",
        "transform": "Mesh-to-camera rigid transform when the named frames differ.",
        "mask": "Optional uint8 selection on the source camera image.",
        "region": "Optional region on the source camera image, intersected with the mask.",
    }
    output_help = {
        "texture": "UV-space atlas in the source encoding and dtype. Unsupported texels are zero.",
        "texture_mask": "255 for observed texels and zero for gaps, occlusion or invalid donors.",
        "mesh": "The original geometry and UVs for use with the atlas after acceptance.",
        "report": "Source registration, coverage, geometry and budget diagnostics.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["maximum_z_m"] <= p["minimum_z_m"]:
            raise ValidationError("Maximum Z must exceed minimum Z", "minimum_z_m", "maximum_z_m")
        if p["texture_width"] * p["texture_height"] > _MAX_ATLAS_PIXELS:
            raise ValidationError(
                "Texture exceeds the atlas pixel limit", "texture_width", "texture_height"
            )

    def process(self, inputs):
        image, mesh, camera = inputs["image"], inputs["mesh"], inputs["camera"]
        raw, encoding = np.asarray(image), image_format(image)
        channels = 1 if raw.ndim == 2 else raw.shape[2]
        if (
            _ENCODINGS.get(encoding) != channels
            or (encoding == "gray" and raw.ndim != 2)
            or raw.dtype not in (np.uint8, np.uint16, np.float32, np.float64)
        ):
            raise ValidationError(
                "Use a gray, BGR, RGB, BGRA or RGBA image with a supported depth", "image"
            )
        k, distortion, size = camera_for_image(camera, image)
        if camera.get("lens", "pinhole") != "pinhole" or np.any(distortion):
            raise ValidationError("Use rectified pinhole intrinsics with zero distortion", "camera")
        if size[0] * size[1] > MAX_PIXELS:
            raise ValidationError("Source camera grid exceeds the pixel limit", "image")
        if mesh.texcoords is None or np.any(
            (mesh.texcoords[:, :2] < 0) | (mesh.texcoords[:, :2] > 1)
        ):
            raise ValidationError("Mesh needs UV coordinates in [0, 1]", "mesh")
        selected = selection_mask(inputs, image)
        p = self.parameters
        near, far = float(p["minimum_z_m"]), float(p["maximum_z_m"])
        limit, budget = int(p["max_triangles"]), int(p["max_pixel_tests"])
        atlas = (int(p["texture_width"]), int(p["texture_height"]))
        camera_frame = camera.get("frame", "camera")
        report = {
            "camera_frame": camera_frame,
            "mesh_frame": mesh.frame,
            "source_size": list(size),
            "texture_size": list(atlas),
            "uv_convention": "u=(x+0.5)/width, v=1-(y+0.5)/height",
            "input_triangles": len(mesh.faces),
            "sampling": p["sampling"],
            "atlas_texels": 0,
            "valid_texels": 0,
            "coverage": 0.0,
        }
        outputs = {
            "texture": NO_VALUE,
            "texture_mask": NO_VALUE,
            "mesh": NO_VALUE,
            "report": report,
        }

        def finish(status, score=0.0, payloads=None):
            report["status"] = status
            return Result(
                outputs,
                passed=status == "ok",
                score=score,
                payloads=payloads or {},
                fail_reasons=[] if status == "ok" else [f"mesh_texture_{status}"],
                message=f"{report['valid_texels']:,}/{report['atlas_texels']:,} texels · {status}",
            )

        if len(mesh.faces) > limit:
            return finish("triangle_budget")
        if self._cancelled():
            return finish("cancelled")
        mesh.validate()
        xyz = _camera_vertices(mesh, camera_frame, inputs.get("transform"))
        geometry, issue = _clipped_geometry(
            xyz, mesh.faces, None, "white", _view_planes(k, size, near, far), limit, self._cancelled
        )
        if issue:
            return finish(issue)
        values, faces, _ = geometry
        projected = _projected_vertices(values[:, :3], k, size)
        source_work = _pixel_work(projected, faces, size)
        uv = (mesh.texcoords[:, :2].astype(np.float64) * [1, -1] + [0, 1]) * atlas - 0.5
        corners = uv[mesh.faces]
        area = _edge(corners[:, 0], corners[:, 1], corners[:, 2])
        low = np.ceil(corners.min(axis=1)).astype(np.int32)
        high = np.floor(corners.max(axis=1)).astype(np.int32)
        spans = np.maximum(high - low + 1, 0)
        nondegenerate = np.abs(area) > 1e-12
        atlas_work = int(np.prod(spans[nondegenerate], axis=1, dtype=np.int64).sum())
        report.update(
            raster_triangles=len(faces),
            camera_pixel_tests=source_work,
            atlas_texel_tests=atlas_work,
            degenerate_uv_faces=int((~nondegenerate).sum()),
        )
        if source_work + atlas_work > budget:
            return finish("pixel_budget")
        if self._cancelled():
            return finish("cancelled")
        depth, _, _ = _native_raster(
            values, projected, faces, size, near, far, "none", "white", "depth_only"
        )
        occupied = np.zeros(atlas[::-1], bool)
        mask = np.zeros(atlas[::-1], np.uint8)
        texture = np.zeros(atlas[::-1] + raw.shape[2:], raw.dtype)
        for index in np.flatnonzero(nondegenerate & (spans > 0).all(axis=1)):
            triangle, uv_triangle = xyz[mesh.faces[index]], corners[index]
            if area[index] < 0:
                triangle = triangle[::-1]
                uv_triangle = uv_triangle[::-1]
            for xy, weights in _atlas_samples(
                uv_triangle, low[index], high[index], abs(area[index])
            ):
                if self._cancelled():
                    return finish("cancelled")
                x, y = xy.T
                if occupied[y, x].any():
                    return finish("overlapping_uv")
                occupied[y, x] = True
                valid, samples = _sample_image(
                    weights @ triangle, triangle, raw, depth, selected, k, near, far, p
                )
                x, y = xy[valid].T
                texture[y, x] = samples
                mask[y, x] = 255
        if self._cancelled():
            return finish("cancelled")
        used, valid = int(occupied.sum()), int(np.count_nonzero(mask))
        fraction = valid / used if used else 0.0
        report.update(atlas_texels=used, valid_texels=valid, coverage=fraction)
        outputs.update(texture=ImageArray(texture, encoding), texture_mask=mask)
        payload = applied_payload({"spatial": frame(atlas, f"{self.id}:uv")}, camera.get(REVISION))
        payloads = {"texture": payload, "texture_mask": payload}
        if not valid:
            return finish("no_visible_texels", fraction, payloads)
        if fraction < p["minimum_coverage"]:
            return finish("insufficient_coverage", fraction, payloads)
        outputs["mesh"] = mesh
        return finish("ok", fraction, payloads)
