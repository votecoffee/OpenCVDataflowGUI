"""Bounded numeric/JSON messages between the GUI and its isolated Python worker."""

import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

LIMIT = 128 * 1024 * 1024


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def bad(value):
        raise ValueError("Nonfinite JSON number")

    def number(value):
        result = float(value)
        return result if math.isfinite(result) else bad(value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=number)


def write_packet(directory, metadata, arrays):
    directory = Path(directory)
    for name, array in arrays.items():
        if not name.replace("_", "").isalnum() or np.asarray(array).dtype.hasobject:
            raise ValueError("Only named numeric arrays may cross the worker boundary")
    if sum(np.asarray(a).nbytes for a in arrays.values()) > LIMIT:
        raise ValueError("Worker arrays exceed 128 MiB")
    raw = (json.dumps(metadata, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Worker metadata exceeds 4 MiB")
    np.savez_compressed(directory / "arrays.npz", **arrays)
    (directory / "metadata.json").write_bytes(raw)


def read_packet(directory):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    array_path = directory / "arrays.npz"
    if metadata_path.stat().st_size > 4 * 1024 * 1024 or array_path.stat().st_size > LIMIT:
        raise ValueError("Oversized worker response")
    metadata = strict_json(metadata_path.read_bytes())
    arrays = {}
    with ZipFile(array_path) as archive:
        names = archive.namelist()
        if (
            len(names) > 16
            or len(names) != len(set(names))
            or sum(i.file_size for i in archive.infolist()) > LIMIT
        ):
            raise ValueError("Invalid worker array members")
        for name in names:
            stem = name.removesuffix(".npy")
            if name != stem + ".npy" or not stem.replace("_", "").isalnum():
                raise ValueError("Invalid worker array name")
            raw = archive.read(name)
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            if version not in ((1, 0), (2, 0)):
                raise ValueError("Unsupported numeric array format")
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, _, dtype = reader(stream)
            if (
                not 1 <= len(shape) <= 3
                or any(type(v) is not int or not 0 <= v <= 2_000_000 for v in shape)
                or dtype.str not in ("|u1", "<i4", "<f4", "<f8", "|b1")
                or math.prod(shape) * dtype.itemsize != len(raw) - stream.tell()
            ):
                raise ValueError("Invalid numeric array header")
            arrays[stem] = np.load(io.BytesIO(raw), allow_pickle=False)
    return metadata, arrays
