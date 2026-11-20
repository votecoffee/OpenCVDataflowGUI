"""Fuse posed camera Z into a bounded TSDF and extract only observed surface cells."""

from itertools import product

import numpy as np

from .cloud_io import cloud_path
from .core import Block, Port, Result
from .geometry3d_common import Tool3D
from .mesh_io import LoadMesh, write_mesh
from .meshes import MAX_FACES, TriangleMesh, render_wireframe
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import integer
from .ptcloud3d import tsdf_views
from .stereo3d import metric_depth

MAX_VOXELS = 2_097_152
MAX_ACTIVE_CELLS = 250_000
VOLUME_CHUNK = 65_536


def integrate_view(field, counts, grid, origin, voxel, depth, transform, k, p):
    """Integrate camera-Z signed distances in bounded chunks of world samples."""
    rotation, shift = transform[:3, :3], transform[:3, 3]
    width, height = depth.shape[1], depth.shape[0]
    truncation, maximum = float(p["truncation_m"]), float(p["maximum_z_m"])
    for start in range(0, field.size, VOLUME_CHUNK):
        end = min(start + VOLUME_CHUNK, field.size)
        index = np.column_stack(np.unravel_index(np.arange(start, end), grid))
        xyz = (index * voxel + (origin - shift)) @ rotation
        sample = np.flatnonzero((xyz[:, 2] > 0) & (xyz[:, 2] <= maximum + truncation))
        xyz = xyz[sample]
        uv = (xyz[:, :2] / xyz[:, 2, None]) * [k[0, 0], k[1, 1]] + k[:2, 2]
        inside = (uv >= -0.5).all(axis=1) & (uv < [width - 0.5, height - 0.5]).all(axis=1)
        x, y = np.floor(uv[inside] + 0.5).astype(np.int32).T
        observed = depth[y, x]
        distance = observed - xyz[inside, 2]
        valid = np.isfinite(observed) & (observed <= maximum) & (distance >= -truncation)
        target = start + sample[inside][valid]
        tsdf = np.minimum(distance[valid] / truncation, 1)
        weight = counts[target].astype(np.float32)
        field[target] += (tsdf - field[target]) / (weight + 1)
        counts[target] = np.minimum(counts[target] + 1, int(p["max_weight"]))


def surface_cell_mask(values, observed):
    """Select zero crossings in cells with valid observations at all eight corners."""
    shape = tuple(size - 1 for size in values.shape)
    eligible = np.ones(shape, bool)
    positive = np.zeros(shape, bool)
    nonpositive = np.zeros(shape, bool)
    for corner in product((0, 1), repeat=3):
        slices = tuple(slice(a, a + n) for a, n in zip(corner, shape, strict=True))
        eligible &= observed[slices]
        positive |= values[slices] > 0
        nonpositive |= values[slices] <= 0
    mask = np.zeros(values.shape, bool)
    # Lewiner scans each cell using its far corner as the mask index.
    mask[1:, 1:, 1:] = eligible & positive & nonpositive
    return mask


def extract_surface(values, mask, origin, voxel, target, limit):
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError("TSDF mesh extraction requires the app's skimage extra") from error
    try:
        vertices, faces, _, _ = marching_cubes(
            values,
            level=0,
            spacing=(voxel,) * 3,
            mask=mask,
            gradient_direction="descent",
            allow_degenerate=False,
        )
    except RuntimeError as error:
        if str(error) == "No surface found at the given iso value.":
            return NO_VALUE
        raise
    if not len(faces):
        return NO_VALUE
    if len(faces) > limit:
        raise ValidationError(
            "Mesh exceeds maximum faces. Use coarser spacing or a smaller volume."
        )
    return TriangleMesh(vertices + origin, faces, "m", target)


class FuseDepthMesh(Tool3D):
    type_id, title, subtype = (
        "3d.tsdf_mesh",
        "Fuse posed depth into a mesh",
        "Volumetric reconstruction",
    )
    process_inputs_readonly = True
    inputs = (
        Port("depth", "depth", required=False, required_when_connected=True),
        Port("frames", "any", required=False, required_when_connected=True),
        Port("poses", "data", visual_type="pose"),
        Port("camera", "data"),
    )
    outputs = (Port("mesh", "mesh", required=False), Port("report", "data"))
    defaults = {
        "resolution": [64, 64, 64],
        "volume_origin_m": [-0.64, -0.64, 0.5],
        "voxel_size_m": 0.02,
        "truncation_m": 0.06,
        "maximum_z_m": 5.0,
        "minimum_observations": 1,
        "max_weight": 32,
        "max_faces": 250000,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {"input_units": ("m", "mm")}
    parameter_ranges = {
        "voxel_size_m": NumericRange(1e-5, 100),
        "truncation_m": NumericRange(1e-5, 100),
        "maximum_z_m": NumericRange(1e-4, 10000),
        "minimum_observations": NumericRange(1, 32, whole=True),
        "max_weight": NumericRange(1, 32, whole=True),
        "max_faces": NumericRange(1, MAX_FACES, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "resolution": "XYZ sample counts. At most 2,097,152 samples once multiplied.",
        "volume_origin_m": "World XYZ of the first lattice sample, in meters.",
        "voxel_size_m": "World lattice spacing in meters.",
        "truncation_m": "Signed-distance band behind the surface, in meters.",
        "maximum_z_m": "Farthest usable observed camera Z, in meters.",
        "minimum_observations": "Required observations at each of a surface cell's eight corners.",
        "max_weight": "Saturation of each voxel's running-average weight.",
        "max_faces": "Withhold meshes over this triangle count.",
        "input_units": "Physical units for depth without an explicit encoding.",
        "units_per_sample": "Physical units per raw uint16 count.",
    }
    input_help = {
        "depth": "Connect either one depth plane or a group of depth planes.",
        "poses": "One camera-to-world pose per frame in the same order, using one world frame.",
        "camera": "Undistorted pinhole intrinsics for all depth frames.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_observations"]) > int(p["max_weight"]):
            raise ValidationError(
                "Minimum observations exceed the maximum voxel weight",
                "minimum_observations",
                "max_weight",
            )
        if not isinstance(p["resolution"], (list, tuple)) or len(p["resolution"]) != 3:
            raise ValidationError("Resolution needs three XYZ sample counts", "resolution")
        try:
            grid = tuple(integer(v, "Volume dimension", 2, 256) for v in p["resolution"])
        except ValueError as error:
            raise ValidationError(str(error), "resolution") from error
        if np.prod(grid) > MAX_VOXELS:
            raise ValidationError("Resolution exceeds the 2,097,152 sample limit", "resolution")
        origin = np.asarray(p["volume_origin_m"], dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValidationError(
                "Origin must be three finite world coordinates", "volume_origin_m"
            )

    def process(self, inputs):
        p = self.parameters
        depths, transforms, target, k, _ = tsdf_views(inputs, inputs["camera"])
        if len(depths) > 32:
            raise ValidationError("A TSDF mesh accepts at most 32 posed depth views")
        grid = tuple(map(int, p["resolution"]))
        origin = np.asarray(p["volume_origin_m"], dtype=float)
        voxel = float(p["voxel_size_m"])
        values = np.zeros(grid, np.float32)
        counts = np.zeros(grid, np.uint8)
        for depth, pose in zip(depths, transforms, strict=True):
            if self._cancelled():
                raise InterruptedError("TSDF mesh fusion cancelled")
            samples = metric_depth(depth, p["units_per_sample"], p["input_units"])
            integrate_view(values.ravel(), counts.ravel(), grid, origin, voxel, samples, pose, k, p)
        observed = counts >= int(p["minimum_observations"])
        active = surface_cell_mask(values, observed)
        cells = int(active.sum())
        if cells > MAX_ACTIVE_CELLS:
            raise ValidationError(
                "Too many surface cells. Use coarser spacing or a smaller volume."
            )
        mesh = (
            extract_surface(values, active, origin, voxel, target, int(p["max_faces"]))
            if cells
            else NO_VALUE
        )
        ok = mesh is not NO_VALUE
        report = {
            "frames": len(depths),
            "observed_voxels": int(observed.sum()),
            "surface_cells": cells,
            "resolution": list(grid),
            "voxel_size_m": voxel,
            "world_frame": target,
            "units": "m",
            "face_count": len(mesh.faces) if ok else 0,
        }
        graphics = (
            {"Mesh wireframe": render_wireframe(mesh)} if ok and self.capture_graphics else {}
        )
        return Result(
            {"mesh": mesh, "report": report},
            passed=ok,
            fail_reasons=[] if ok else ["no_supported_tsdf_surface"],
            graphics=graphics,
            message=f"{report['face_count']} triangles from {len(depths)} views",
        )


class SaveMeshPLY(Block):
    type_id, title, category = "output.mesh_ply", "Save triangle mesh as PLY", "Output"
    block_type, subtype = "Output", "3D geometry"
    process_inputs_readonly = True
    inputs = (Port("mesh", "mesh"),)
    outputs = (Port("path", "data"), Port("face_count", "number"))
    defaults = {"path": "surface.ply", "max_faces": 250000}
    parameter_ranges = {"max_faces": NumericRange(1, MAX_FACES, whole=True)}
    parameter_help = {
        "path": "Write a new OBJ or binary PLY. Resolve relative paths from the project folder."
    }

    def process(self, inputs):
        path = cloud_path(self)
        mesh = inputs["mesh"]
        write_mesh(path, mesh, int(self.parameters["max_faces"]))
        return Result(
            {"path": str(path), "face_count": len(mesh.faces)},
            message=f"Saved mesh as {path.suffix[1:].upper()}",
        )


TSDF_MESH_BLOCKS = (FuseDepthMesh, LoadMesh, SaveMeshPLY)
