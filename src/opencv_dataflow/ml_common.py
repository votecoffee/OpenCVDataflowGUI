"""Portable ML data contracts and local artifact resolution."""

import hashlib
from pathlib import Path

import numpy as np


def local_path(block, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Choose a local model/artifact path")
    path = Path(value)
    if not path.is_absolute() and block.project_dir is not None:
        path = block.project_dir / path
    return path.resolve()


def file_key(path):
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return str(path), stat.st_size, stat.st_mtime_ns


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array(value, name, max_elements=2_000_000):
    result = np.asarray(value, dtype=np.float32)
    if not result.size or result.size > max_elements or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain 1..{max_elements} finite numeric values")
    return result


def tensor_record(value, max_elements):
    value = array(value, "Network output", max_elements)
    return {
        "model": "tensor_v1",
        "shape": list(value.shape),
        "dtype": "float32",
        "values": value.reshape(-1).tolist(),
    }


def selected_tensor(tensors, name=""):
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Connect the DNN tensors output")
    value = tensors.get(name) if name else next(iter(tensors.values()))
    if not isinstance(value, dict) or value.get("model") != "tensor_v1":
        raise ValueError("Select an existing tensor output name")
    values = array(value.get("values"), "Tensor")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > 8
        or any(type(n) is not int or n < 1 for n in shape)
        or int(np.prod(shape, dtype=object)) != values.size
    ):
        raise ValueError("Invalid tensor shape")
    return values.reshape(shape)


def normalized_vectors(values, mode):
    values = array(values, "Vectors")
    if mode == "l2":
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("Cannot L2 normalize a zero or overflowing vector")
        values = values / norms
    elif mode != "none":
        raise ValueError("Unknown vector normalization")
    return values
