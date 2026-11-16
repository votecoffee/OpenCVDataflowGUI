"""Bounded numeric/JSON messages between the GUI and its isolated Python worker."""

import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

LIMIT = 128 * 1024 * 1024
DTYPES = ("|u1", "<i4", "<f4", "<f8", "|b1")
CHUNK_BYTES = 4 * 1024 * 1024


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


def write_packet(directory, metadata, arrays, *, compression="stored"):
    directory = Path(directory)
    if compression not in ("stored", "deflated_fast"):
        raise ValueError("Choose stored or deflated_fast packet encoding")
    if len(arrays) > 16:
        raise ValueError("Invalid worker array members")
    normalized = {}
    for name, array in arrays.items():
        array = np.asarray(array)
        if (
            not isinstance(name, str)
            or not name.replace("_", "").isalnum()
            or array.dtype.hasobject
        ):
            raise ValueError("Only named numeric arrays may cross the worker boundary")
        if (
            array.dtype.str not in DTYPES
            or not 1 <= array.ndim <= 3
            or any(v > 2_000_000 for v in array.shape)
        ):
            raise ValueError("Invalid numeric array header")
        normalized[name] = array
    if sum(a.nbytes for a in normalized.values()) > LIMIT:
        raise ValueError("Worker arrays exceed 128 MiB")
    raw = (json.dumps(metadata, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Worker metadata exceeds 4 MiB")

    def encode(mode):
        with ZipFile(
            directory / "arrays.npz",
            "w",
            compression=ZIP_STORED if mode == "stored" else ZIP_DEFLATED,
            compresslevel=None if mode == "stored" else 1,
        ) as archive:
            for name, array in normalized.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
            expanded = sum(info.file_size for info in archive.infolist())
        if expanded > LIMIT:
            raise ValueError("Worker arrays including headers exceed 128 MiB")

    encode(compression)
    if (directory / "arrays.npz").stat().st_size > LIMIT and compression == "stored":
        # ZIP framing can exceed the file cap near the exact legacy payload bound.
        encode("deflated_fast")
    if (directory / "arrays.npz").stat().st_size > LIMIT:
        raise ValueError("Oversized worker response")
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
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported numeric array format")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream)
                size = math.prod(shape) * dtype.itemsize
                if (
                    not 1 <= len(shape) <= 3
                    or any(type(v) is not int or not 0 <= v <= 2_000_000 for v in shape)
                    or dtype.str not in DTYPES
                    or size != archive.getinfo(name).file_size - stream.tell()
                ):
                    raise ValueError("Invalid numeric array header")
                array = np.empty(shape, dtype=dtype, order="F" if fortran else "C")
                buffer = memoryview(array.ravel(order="K")).cast("B")
                offset = 0
                while offset < size:
                    count = stream.readinto(buffer[offset : offset + CHUNK_BYTES])
                    if not count:
                        raise ValueError("Truncated numeric array data")
                    offset += count
                if stream.read(1):
                    raise ValueError("Trailing numeric array data")
                arrays[stem] = array
    return metadata, arrays
