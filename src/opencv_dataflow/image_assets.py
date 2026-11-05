"""Bounded, lossless, non-pickle trained image assets."""

import base64
import copy
import hashlib
import io
import json

import numpy as np

from .images import ImageArray, image_format
from .processing import image_value

MAX_IMAGE_BYTES = 8 * 1024 * 1024


def seal(asset):
    body = {k: v for k, v in asset.items() if k != "id"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def encode_image(image):
    pixels = image_value(image)
    if pixels.nbytes > MAX_IMAGE_BYTES:
        raise ValueError("Trained image exceeds 8 MiB of pixel data")
    stream = io.BytesIO()
    np.save(stream, pixels, allow_pickle=False)
    context = {
        k: copy.deepcopy(v)
        for k, v in getattr(image, "metadata", {}).items()
        if k in ("spatial", "pyramid", "recipe_ref")
    }
    asset = {
        "model": "trained_image_v1",
        "shape": list(pixels.shape),
        "dtype": str(pixels.dtype),
        "color_format": image_format(image),
        "context": context,
        "data": base64.b64encode(stream.getvalue()).decode("ascii"),
    }
    asset["id"] = seal(asset)
    return asset


def decode_image(asset, load=True):
    if (
        not isinstance(asset, dict)
        or set(asset) != {"model", "shape", "dtype", "color_format", "context", "data", "id"}
        or asset.get("model") != "trained_image_v1"
        or not isinstance(asset.get("data"), str)
        or len(asset["data"]) > (MAX_IMAGE_BYTES + 65536) * 4 // 3 + 4
        or not isinstance(asset.get("context"), dict)
    ):
        raise ValueError("Invalid or oversized trained_image_v1 asset")
    if seal(asset) != asset["id"]:
        raise ValueError("Trained image checksum mismatch")
    from .payloads import validate_payload

    validate_payload(asset["context"])
    raw = base64.b64decode(asset["data"], validate=True)
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported trained image NPY version")
    size = int(np.prod(shape, dtype=object)) * dtype.itemsize
    if (
        dtype.hasobject
        or dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
        or len(shape) not in (2, 3)
        or any(n <= 0 for n in shape)
        or size > MAX_IMAGE_BYTES
        or len(raw) - stream.tell() != size
        or list(shape) != asset["shape"]
        or str(dtype) != asset["dtype"]
        or asset["color_format"] not in ("gray", "bgr", "rgb", "bgra", "rgba", "hsv", "lab")
    ):
        raise ValueError("Invalid trained image shape, depth, size or encoding")
    channels = 1 if len(shape) == 2 else shape[2]
    if (
        channels
        != {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4, "hsv": 3, "lab": 3}[
            asset["color_format"]
        ]
    ):
        raise ValueError("Trained image channel count differs from encoding")
    if not load:
        return None
    pixels = np.load(io.BytesIO(raw), allow_pickle=False)
    image_value(pixels)
    return ImageArray(pixels, asset["color_format"], metadata=asset["context"])
