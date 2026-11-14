"""Atomic, checksummed depth/RGB recipes; bounded NPY and strict JSON, no pickle."""

import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from .specs import MAX_PIXELS, Camera, Recipe, SuctionSettings

MAX_BYTES = 64 * 1024 * 1024


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON number")
    def floating(value):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else invalid(value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=floating)


def array_bytes(value):
    data = io.BytesIO()
    np.save(data, value, allow_pickle=False)
    return data.getvalue()


def read_array(raw, *, rgb=False, labels=False):
    data = io.BytesIO(raw)
    version = np.lib.format.read_magic(data)
    if version not in ((1, 0), (2, 0)):
        raise ValueError("Use NPY version 1 or 2")
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, _, dtype = reader(data)
    if (len(shape) != (3 if rgb else 2) or any(type(v) is not int or v < 2 for v in shape[:2])
            or shape[0] * shape[1] > MAX_PIXELS or (rgb and shape[2] != 3)
            or dtype.str not in (("|u1",) if rgb else ("|u1", "<u2", "<i4") if labels
                                 else ("<u2", "<f4", "<f8"))
            or int(np.prod(shape)) * dtype.itemsize != len(raw) - data.tell()):
        raise ValueError("Malformed/oversized depth or RGB NPY array")
    return np.load(io.BytesIO(raw), allow_pickle=False)


def save_recipe(path, recipe):
    if not isinstance(recipe, Recipe):
        raise ValueError("Save a validated bin-picking Recipe")
    entries = {"depth.npy": array_bytes(recipe.reference_depth)}
    if recipe.reference_rgb is not None:
        entries["rgb.npy"] = array_bytes(recipe.reference_rgb)
    if recipe.reference_instances is not None:
        entries["labels.npy"] = array_bytes(recipe.reference_instances)
    manifest = {"version": 1, **recipe.configuration(),
                "arrays": {name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()}}
    entries["recipe.json"] = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
    if sum(map(len, entries.values())) > MAX_BYTES:
        raise ValueError("Recipe content exceeds 64 MiB")
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, prefix=".bin-picking.", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as archive:
            for name, raw in entries.items():
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 0
                archive.writestr(info, raw)
        if pending.stat().st_size > MAX_BYTES:
            raise ValueError("Recipe archive exceeds 64 MiB")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recipe(path, *, sha256=None):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Recipe archive exceeds 64 MiB")
    raw = path.read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Recipe SHA-256 mismatch")
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if (not {"depth.npy", "recipe.json"}.issubset(names) or
                    set(names) - {"depth.npy", "recipe.json", "rgb.npy", "labels.npy"}
                    or len(names) != len(set(names)) or
                    sum(v.file_size for v in archive.infolist()) > MAX_BYTES or
                    archive.getinfo("recipe.json").file_size > 1024 * 1024):
                raise ValueError("Unexpected/duplicate/oversized recipe entries")
            manifest = strict_json(archive.read("recipe.json"))
            if type(manifest["version"]) is not int or manifest.pop("version") != 1:
                raise ValueError("Unsupported bin-picking recipe version")
            hashes = manifest.pop("arrays")
            if not isinstance(hashes, dict) or set(hashes) != set(names) - {"recipe.json"}:
                raise ValueError("Missing source array hashes")
            arrays = {}
            for name, digest in hashes.items():
                value = archive.read(name)
                if hashlib.sha256(value).hexdigest() != digest:
                    raise ValueError("Reference array checksum mismatch")
                arrays[name] = read_array(value, rgb=name == "rgb.npy", labels=name == "labels.npy")
        return Recipe(reference_depth=arrays["depth.npy"], reference_rgb=arrays.get("rgb.npy"),
                      reference_instances=arrays.get("labels.npy"),
                      **{**manifest, "camera": Camera(**manifest["camera"]),
                         "settings": SuctionSettings(**manifest["settings"])})
    except (BadZipFile, KeyError, TypeError, OSError) as error:
        raise ValueError(f"Invalid bin-picking recipe: {error}") from error
