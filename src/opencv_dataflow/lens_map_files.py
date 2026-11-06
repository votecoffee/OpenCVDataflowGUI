"""Save and load complete, versioned NumPy lens map pairs."""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from .calibration import validate_profile
from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .payloads import NO_VALUE
from .remap_maps import MAX_MAP_PIXELS, lens_map_pair, lens_map_payloads, valid_lens_size

_MODEL = "opencv_dataflow_lens_maps_v1"
_PLANES = ("map_x", "map_y")
_MEMBERS = ("metadata.npy", "map_x.npy", "map_y.npy")
_METADATA_BYTES = 8_192
_MAX_FILE_BYTES = 8 * MAX_MAP_PIXELS + 65_536
_F32 = np.dtype("<f4")


def _path(block):
    value = block.parameters["path"]
    if not isinstance(value, str) or not value.strip() or Path(value).suffix.lower() != ".npz":
        raise ValidationError("Choose a .npz lens map path", "path")
    path = Path(value)
    if block.parameters["path_base"] == "project" and not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError("Save or open the project before using a relative lens map path")
        path = block.project_dir / path
    return path.parent.resolve() / path.name


def _validate_info(info):
    if (
        not isinstance(info, dict)
        or set(info) != {"id", "source_size", "output_size", "rectified_calibration", "valid_roi"}
        or not isinstance(info["id"], str)
        or not re.fullmatch("[0-9a-f]{64}", info["id"])
        or not valid_lens_size(info["source_size"])
        or not valid_lens_size(info["output_size"])
        or math.prod(info["output_size"]) > MAX_MAP_PIXELS
    ):
        raise ValidationError("Invalid lens map metadata")
    profile = info["rectified_calibration"]
    if (
        not isinstance(profile, dict)
        or profile.get("model") != "opencv_pinhole_v1"
        or profile.get("image_size") != info["output_size"]
        or profile.get("distortion") != [0.0] * 5
    ):
        raise ValidationError("Lens maps require a matching zero-distortion profile")
    validate_profile(profile)
    rotation = np.asarray(profile.get("rectification_rotation"), dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3))
        or not np.isclose(np.linalg.det(rotation), 1)
    ):
        raise ValidationError("Lens maps contain an invalid rectification rotation")
    roi = info["valid_roi"]
    if roi is not None:
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) is not int or v < 0 for v in roi)
            or roi[0] + roi[2] > info["output_size"][0]
            or roi[1] + roi[3] > info["output_size"][1]
        ):
            raise ValidationError("Lens maps contain an invalid output ROI")


def _digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _read_array(archive, name, dtype, expected_shape=None):
    entry = archive.getinfo(name)
    with archive.open(entry) as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValidationError("Lens map arrays require NPY version 1")
        shape, fortran, stored_type = np.lib.format.read_array_header_1_0(stream)
        if expected_shape is None:
            shape_ok = len(shape) == 1 and 0 < shape[0] <= _METADATA_BYTES
        else:
            shape_ok = shape == expected_shape
        if fortran or stored_type != dtype or not shape_ok:
            raise ValidationError(f"Unexpected shape or type for {name}")
        size = math.prod(shape) * dtype.itemsize
        if entry.file_size != stream.tell() + size:
            raise ValidationError(f"Unexpected byte count for {name}")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValidationError(f"Truncated lens map entry {name}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class SaveLensMaps(Block):
    type_id, title, category = "output.lens_maps", "Save lens correction maps", "Output"
    block_type, subtype = "Output", "Calibration profiles"
    inputs = (Port("map_x"), Port("map_y"))
    defaults = {"path": "lens_maps.npz", "path_base": "working_directory"}
    parameter_choices = {"path_base": ("working_directory", "project")}
    preview_safe = False
    process_inputs_readonly = True

    def process(self, inputs):
        path = _path(self)
        if path.exists():
            raise FileExistsError("Choose a new path. Existing lens map files are preserved.")
        x, y = (np.asarray(inputs[name]) for name in _PLANES)
        if (
            x.ndim != 2
            or x.shape != y.shape
            or x.size > MAX_MAP_PIXELS
            or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (x, y))
        ):
            raise ValidationError("Use matching, finite float32 lens map planes")
        tagged = lens_map_pair(self, list(x.shape[::-1]))
        if tagged is None:
            raise ValidationError("Connect X and Y maps from Build or Load lens correction maps")
        info = {key: value for key, value in tagged.items() if key != "axis"}
        _validate_info(info)
        arrays = {name: np.ascontiguousarray(a, dtype=_F32) for name, a in zip(_PLANES, (x, y))}
        manifest = {
            "model": _MODEL,
            "info": info,
            "sha256": {name: _digest(a) for name, a in arrays.items()},
        }
        metadata = json.dumps(manifest, allow_nan=False, sort_keys=True).encode("utf-8")
        if len(metadata) > _METADATA_BYTES:
            raise ValidationError("Lens map metadata exceeds 8 KiB")
        stream = NamedTemporaryFile(
            prefix=".lens-maps-", suffix=".tmp", dir=path.parent, delete=False
        )
        temporary = Path(stream.name)
        try:
            with stream:
                np.savez(stream, metadata=np.frombuffer(metadata, dtype=np.uint8), **arrays)
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return Result(message=f"Saved lens map pair to {path.name}")


class LoadLensMaps(Block):
    type_id, title, category = "input.lens_maps", "Load lens correction maps", "Input"
    block_type, subtype = "Input", "Calibration profiles"
    outputs = (
        Port("map_x"),
        Port("map_y"),
        Port("rectified_calibration", "data"),
        Port("valid_roi", "data", required=False),
    )
    defaults = {"path": "", "path_base": "working_directory"}
    parameter_choices = SaveLensMaps.parameter_choices

    def process(self, inputs):
        path = _path(self)
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > _MAX_FILE_BYTES:
                raise ValidationError("Lens map file exceeds the archive size limit")
            with ZipFile(stream) as archive:
                if sorted(archive.namelist()) != sorted(_MEMBERS) or any(
                    entry.compress_type != ZIP_STORED for entry in archive.infolist()
                ):
                    raise ValidationError("Expected an uncompressed lens map NPZ archive")
                manifest = json.loads(
                    _read_array(archive, "metadata.npy", np.dtype(np.uint8)).tobytes()
                )
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("model") != _MODEL
                    or not isinstance(manifest.get("sha256"), dict)
                    or set(manifest["sha256"]) != set(_PLANES)
                ):
                    raise ValidationError("Invalid lens map file manifest")
                info = manifest.get("info")
                _validate_info(info)
                shape = tuple(info["output_size"][::-1])
                outputs = {}
                for name in _PLANES:
                    array = _read_array(archive, f"{name}.npy", _F32, shape)
                    if not np.isfinite(array).all() or _digest(array) != manifest["sha256"][name]:
                        raise ValidationError(
                            f"Lens map checksum or pixel validation failed for {name}"
                        )
                    outputs[name] = array.astype(np.float32, copy=False)
        outputs["rectified_calibration"] = info["rectified_calibration"]
        roi = info["valid_roi"]
        outputs["valid_roi"] = NO_VALUE if roi is None else roi
        return Result(outputs, payloads=lens_map_payloads(info), message=f"Loaded {path.name}")
