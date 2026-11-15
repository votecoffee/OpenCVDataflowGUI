"""Frozen multi-capture camera references with bounded, pickle-free persistence."""

import copy
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import cv2
import numpy as np

from ..images import DepthArray, image_format
from ..parameter_validation import ValidationError
from ..payloads import ValuePacket, metadata
from ..processing import image_value
from ..selection import rasterize

MAX_PIXELS = 2_097_152
MAX_SAMPLES = 32_000_000
MAX_BYTES = 192 * 1024 * 1024
TRAIN_DEFAULTS = {
    "focus_method": "tenengrad",
    "focus_aperture": "3",
    "normalize_focus": True,
    "float_min": 0.0,
    "float_max": 1.0,
    "stability_threshold": 0.08,
}
ROLES = ("lighting", "focus", "fiducials", "ignore", "position")


def settings_value(parameters):
    settings = {**TRAIN_DEFAULTS, **{k: v for k, v in parameters.items() if k in TRAIN_DEFAULTS}}
    if (
        settings["focus_method"] not in ("tenengrad", "laplacian_variance")
        or settings["focus_aperture"] not in ("1", "3", "5", "7")
        or type(settings["normalize_focus"]) is not bool
    ):
        raise ValidationError("Invalid focus training settings")
    for key in ("float_min", "float_max", "stability_threshold"):
        if isinstance(settings[key], (bool, str)) or not math.isfinite(settings[key]):
            raise ValidationError("Use finite numeric training settings", key)
    if not -1e12 <= settings["float_min"] < settings["float_max"] <= 1e12:
        raise ValidationError("Choose ordered Float min/max within +/-1e12")
    if not 0 <= settings["stability_threshold"] <= 1:
        raise ValidationError("Stability threshold must be in [0,1]")
    return settings


@dataclass(frozen=True)
class Reference:
    image: np.ndarray
    stable: np.ndarray
    info: dict
    history: np.ndarray | None = None


def normalized(image, settings):
    raw = image_value(image, (np.uint8, np.uint16, np.int16, np.float32, np.float64))
    fmt = image_format(image)
    if isinstance(image, DepthArray) or fmt not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Use gray or straight-alpha RGB/BGR intensity images")
    if raw.size == 0 or np.prod(raw.shape[:2]) > MAX_PIXELS or max(raw.shape[:2]) > 16384:
        raise ValidationError(
            "Camera health images are limited to 2,097,152 pixels and 16384 per axis"
        )
    if metadata(image).get("alpha_representation") == "premultiplied":
        raise ValidationError("Unassociate premultiplied alpha before camera health checks")
    value = raw[..., None] if raw.ndim == 2 else raw[..., :3]
    if raw.dtype.kind in "ui":
        bounds = np.iinfo(raw.dtype)
        lo, hi = bounds.min, bounds.max
    else:
        lo, hi = settings["float_min"], settings["float_max"]
        if np.any(value < lo) or np.any(value > hi):
            raise ValidationError("Floating samples exceed configured Float min/max")
    return ((value.astype(np.float64) - lo) / (hi - lo)).astype(np.float32)


def gray(image, encoding):
    if image.shape[2] == 1:
        return image[..., 0]
    weights = [0.299, 0.587, 0.114] if encoding.startswith("rgb") else [0.114, 0.587, 0.299]
    return (image @ np.asarray(weights, np.float32)).astype(np.float32)


def focus(gray_image, mask, settings):
    if np.count_nonzero(mask) < 8:
        return None
    image = gray_image.astype(np.float64)
    k = int(settings["focus_aperture"])
    if settings["focus_method"] == "tenengrad":
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
        value = float(np.mean((dx * dx + dy * dy)[mask]))
    else:
        response = cv2.Laplacian(image, cv2.CV_64F, ksize=k)
        value = float(response[mask].var())
    if settings["normalize_focus"]:
        variance = float(image[mask].var())
        if variance < 1e-8:
            return None
        value /= variance
    return value if math.isfinite(value) else None


def full_region(size):
    return {
        "model": "image_region_v1",
        "image_size": list(size),
        "shape": "rectangle",
        "rectangle": [0, 0, *size],
    }


def regions_value(regions, size):
    result = {}
    for role in ROLES:
        rows = regions.get(
            role,
            [{"name": role.title(), "region": full_region(size)}]
            if role in ("lighting", "focus")
            else [],
        )
        if not isinstance(rows, list) or len(rows) > 16:
            raise ValidationError("Each region category accepts up to sixteen named regions")
        names = set()
        result[role] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise ValidationError("Regions need a name and an image_region_v1 definition")
            name = row["name"].strip()
            if not name or len(name) > 80 or name.casefold() in names:
                raise ValidationError("Use unique nonempty region names, up to eighty characters")
            names.add(name.casefold())
            region = copy.deepcopy(row.get("region"))
            mask = rasterize(region, (size[1], size[0]))
            if np.count_nonzero(mask) < 8:
                raise ValidationError(f"Region {name} needs at least eight pixels")
            result[role].append({"name": name, "region": region})
    return result


def union(regions, role, shape, *, whole_if_empty=False):
    rows = regions[role]
    mask = np.ones(shape, bool) if not rows and whole_if_empty else np.zeros(shape, bool)
    for row in rows:
        mask |= rasterize(row["region"], shape) != 0
    return mask


def _summary(values):
    data = np.asarray(values, float)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "minimum": data.min(axis=0).tolist(),
        "maximum": data.max(axis=0).tolist(),
    }


def train_reference(images, regions=None, parameters=None):
    settings = settings_value(parameters or {})
    if not 1 <= len(images) <= 32:
        raise ValidationError("Train with one to thirty-two independent reference captures")
    frames, captures = [], []
    source = images[0].value if isinstance(images[0], ValuePacket) else images[0]
    fmt, dtype, shape = image_format(source), source.dtype.name, list(source.shape)
    for entry in images:
        image = entry.value if isinstance(entry, ValuePacket) else entry
        if image_format(image) != fmt or image.dtype.name != dtype or list(image.shape) != shape:
            raise ValidationError(
                "Training captures must share pixel depth, encoding and dimensions"
            )
        frames.append(normalized(image, settings))
        if sum(a.size for a in frames) > MAX_SAMPLES:
            raise ValidationError("Training history exceeds 32 million intensity samples")
        context = metadata(entry)
        previous = captures[0].get("spatial") if captures else None
        spatial = context.get("spatial")
        if (
            previous
            and spatial
            and (
                previous["root_size"] != spatial["root_size"]
                or not np.allclose(previous["to_root"], spatial["to_root"], atol=1e-9, rtol=0)
            )
        ):
            raise ValidationError("Training captures use different crop/scale pixel grids")
        captures.append(
            {k: context[k] for k in ("ticket", "processing_time", "spatial") if k in context}
        )
    history = np.stack(frames)
    baseline = np.median(history, axis=0).astype(np.float32)
    spread = np.ptp(history, axis=0).max(axis=2)
    stable = spread <= settings["stability_threshold"]
    size = [shape[1], shape[0]]
    regions = regions_value(regions or {}, size)
    stable &= ~union(regions, "ignore", stable.shape)
    baselines = {}
    for role in ("lighting", "focus"):
        baselines[role] = []
        for row in regions[role]:
            mask = (rasterize(row["region"], stable.shape) != 0) & stable
            values = []
            for image in frames:
                value = (
                    image[mask].mean(axis=0).tolist()
                    if role == "lighting" and mask.any()
                    else (focus(gray(image, fmt), mask, settings) if role == "focus" else None)
                )
                values.append(value)
            valid = [value for value in values if value is not None]
            baselines[role].append(
                {
                    "name": row["name"],
                    "values": values,
                    "statistics": _summary(valid) if valid else None,
                }
            )
    info = {
        "model": "camera_health_reference_v1",
        "image_size": size,
        "input_shape": shape,
        "encoding": fmt,
        "dtype": dtype,
        "capture_count": len(frames),
        "settings": settings,
        "regions": regions,
        "baselines": baselines,
        "captures": captures,
        "stable_pixels": int(stable.sum()),
    }
    return Reference(baseline, stable.astype(np.uint8) * 255, info, history)


def save_reference(reference, path):
    if reference.history is None:
        raise ValidationError("Training history is required when saving a new reference")
    out = io.BytesIO()
    raw = json.dumps(reference.info, sort_keys=True, allow_nan=False).encode()
    if len(raw) > 2_000_000:
        raise ValidationError("Camera reference metadata exceeds two million bytes")
    np.savez_compressed(
        out,
        image=reference.image,
        stable=reference.stable,
        metadata=np.frombuffer(raw, np.uint8),
        history=reference.history,
    )
    data = out.getvalue()
    if len(data) > MAX_BYTES:
        raise ValidationError("Reference archive exceeds 192 MiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValidationError("Reference path already contains a different trained baseline")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return hashlib.sha256(data).hexdigest()


def stage_reference(reference, project_dir):
    if project_dir is None:
        raise ValidationError("Save the project before staging a camera reference")
    relative = f"models/camera_health/{uuid4().hex}/reference.npz"
    digest = save_reference(reference, Path(project_dir) / relative)
    return {"reference_path": relative, "reference_sha256": digest}


def load_reference(path, digest, *, include_history=False):
    path = Path(path)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValidationError("Use a trained, SHA-256-pinned camera reference")
    if path.stat().st_size > MAX_BYTES:
        raise ValidationError("Reference archive exceeds 192 MiB")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValidationError("Camera reference SHA-256 does not match its saved training")
    with ZipFile(io.BytesIO(data)) as archive:
        if sorted(archive.namelist()) != ["history.npy", "image.npy", "metadata.npy", "stable.npy"]:
            raise ValidationError("Unexpected camera reference archive members")
        headers = {}
        for name in archive.namelist():
            entry = archive.getinfo(name)
            with archive.open(entry) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValidationError("Unsupported camera reference array header")
                shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                if (
                    order
                    or dtype.hasobject
                    or entry.file_size != stream.tell() + math.prod(shape) * dtype.itemsize
                ):
                    raise ValidationError("Invalid reference array layout")
                headers[name] = (shape, dtype)
        shape, kind = headers["image.npy"]
        if (
            len(shape) != 3
            or shape[2] not in (1, 3)
            or kind != np.float32
            or math.prod(shape[:2]) > MAX_PIXELS
            or min(shape) < 1
        ):
            raise ValidationError("Invalid reference image dimensions/type")
        if headers["stable.npy"] != (shape[:2], np.dtype("uint8")):
            raise ValidationError("Invalid stable-pixel mask")
        hs, ht = headers["history.npy"]
        if (
            len(hs) != 4
            or hs[1:] != shape
            or not 1 <= hs[0] <= 32
            or math.prod(hs) > MAX_SAMPLES
            or ht != np.float32
        ):
            raise ValidationError("Invalid bounded capture history")
        ms, mt = headers["metadata.npy"]
        if len(ms) != 1 or not 1 <= ms[0] <= 2_000_000 or mt != np.uint8:
            raise ValidationError("Invalid camera reference metadata")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        info = json.loads(arrays["metadata"].tobytes())
        image, stable = arrays["image"], arrays["stable"]
        history = arrays["history"] if include_history else None
    if (
        info.get("model") != "camera_health_reference_v1"
        or info.get("image_size") != [shape[1], shape[0]]
        or info.get("capture_count") != hs[0]
        or not np.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
        or not np.isin(stable, [0, 255]).all()
    ):
        raise ValidationError("Invalid trained camera reference")
    regions_value(info["regions"], info["image_size"])
    if info.get("dtype") not in ("uint8", "uint16", "int16", "float32", "float64"):
        raise ValidationError("Invalid trained input pixel depth")
    encoding = info.get("encoding")
    if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Invalid trained color encoding")
    expected_shape = list(shape[:2]) + (
        [] if encoding == "gray" else [4 if encoding.endswith("a") else 3]
    )
    if info.get("input_shape") != expected_shape or shape[2] != (1 if encoding == "gray" else 3):
        raise ValidationError("Trained input layout differs from reference intensities")
    if max(shape[:2]) > 16384 or set(info["settings"]) != set(TRAIN_DEFAULTS):
        raise ValidationError("Invalid training geometry/settings")
    settings_value(info["settings"])
    if len(info.get("captures", [])) != hs[0]:
        raise ValidationError("Reference history metadata count does not match saved captures")
    if history is not None and (
        not np.isfinite(history).all() or history.min() < 0 or history.max() > 1
    ):
        raise ValidationError("Reference history contains invalid normalized samples")
    for array in (image, stable, history):
        if array is not None:
            array.flags.writeable = False
    return Reference(image, stable, info, history)
