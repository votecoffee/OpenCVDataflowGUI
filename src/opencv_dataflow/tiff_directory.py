"""Read bounded TIFF directory declarations without decoding pixel data."""

import struct
from dataclasses import dataclass

from .parameter_validation import ValidationError

MAX_BYTES = 128 * 1024**2
MAX_PIXELS = 16_777_216
MAX_PAGES = 64
MAX_CONTAINER_PAGES = 1024
TYPE_BYTES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
    16: 8,
    17: 8,
    18: 8,
}
INTEGER_TYPES = {1: "B", 3: "H", 4: "I", 16: "Q"}
LOSSLESS_COMPRESSION = {1, 5, 8, 32946, 32773}


@dataclass(frozen=True)
class TiffPage:
    size: tuple[int, int]
    dtype: str
    channels: int
    compression: int

    @property
    def pixels(self):
        return self.size[0] * self.size[1]

    @property
    def nbytes(self):
        return (
            self.pixels
            * self.channels
            * {"uint8": 1, "uint16": 2, "float32": 4, "float64": 8}[self.dtype]
        )

    def describe(self):
        return {
            "size": list(self.size),
            "dtype": self.dtype,
            "channels": self.channels,
            "compression": {1: "none", 5: "lzw", 8: "deflate", 32946: "deflate", 32773: "packbits"}[
                self.compression
            ],
            "compression_code": self.compression,
        }


def directory(data):
    """Accept baseline top-left contiguous gray/RGB(A), integer8/16 or float32/64.

    Both TIFF byte orders and classic/BigTIFF directory layouts are inspected.
    This bounds declared allocations and offsets, not the native codec's process.
    """

    def fail(message):
        raise ValidationError(f"Invalid or unsupported multipage TIFF: {message}")

    if not isinstance(data, bytes) or not 8 <= len(data) <= MAX_BYTES:
        fail("buffer must contain 8 bytes..128 MiB")
    if data[:2] not in (b"II", b"MM"):
        fail("missing byte-order signature")
    endian = "<" if data[:2] == b"II" else ">"

    def unpack(code, offset):
        length = struct.calcsize(endian + code)
        if offset < 0 or offset + length > len(data):
            fail("truncated directory or field")
        return struct.unpack_from(endian + code, data, offset)

    magic = unpack("H", 2)[0]
    if magic == 42:
        count_code, offset_code, entry_code, inline, first = "H", "I", "HHII", 4, 4
    elif magic == 43 and unpack("HH", 4) == (8, 0):
        count_code, offset_code, entry_code, inline, first = "Q", "Q", "HHQQ", 8, 8
    else:
        fail("expected classic TIFF or 64-bit BigTIFF")
    offset = unpack(offset_code, first)[0]
    count_bytes = struct.calcsize(count_code)
    entry_bytes = struct.calcsize(endian + entry_code)
    seen, pages, metadata_bytes = set(), [], 0
    while offset:
        if offset in seen or len(pages) >= MAX_CONTAINER_PAGES:
            fail("cyclic directories or more than 1024 pages")
        seen.add(offset)
        count = unpack(count_code, offset)[0]
        if not 1 <= count <= 256:
            fail("directory must contain 1..256 fields")
        table = offset + count_bytes
        end = table + count * entry_bytes
        next_offset = unpack(offset_code, end)[0]
        metadata_bytes += count_bytes + count * entry_bytes + inline
        tags = {}
        for index in range(count):
            position = table + index * entry_bytes
            tag, kind, length, pointer = unpack(entry_code, position)
            if tag in tags or kind not in TYPE_BYTES or not 0 < length <= 131072:
                fail("duplicate, empty, excessive or unknown field")
            byte_count = length * TYPE_BYTES[kind]
            start = position + entry_bytes - inline if byte_count <= inline else pointer
            if start < 0 or start + byte_count > len(data):
                fail("field data lies outside the buffer")
            metadata_bytes += byte_count
            if metadata_bytes > 8 * 1024**2:
                fail("directory metadata exceeds 8 MiB")
            tags[tag] = (kind, length, start)

        def numbers(tag, default=None):
            if tag not in tags:
                if default is None:
                    fail(f"missing field {tag}")
                return (default,)
            kind, length, start = tags[tag]
            if kind not in INTEGER_TYPES:
                fail(f"field {tag} must use unsigned integers")
            return unpack(str(length) + INTEGER_TYPES[kind], start)

        def scalar(tag, default=None):
            values = numbers(tag, default)
            if len(values) != 1:
                fail(f"field {tag} must be scalar")
            return values[0]

        width, height = scalar(256), scalar(257)
        channels, compression = scalar(277, 1), scalar(259, 1)
        bits, sample = numbers(258), numbers(339, 1)
        if not 0 < width <= 16384 or not 0 < height <= 16384 or width * height > MAX_PIXELS:
            fail("page exceeds 16384 per axis or 16777216 pixels")
        if (
            channels not in (1, 3, 4)
            or scalar(262) != (1 if channels == 1 else 2)
            or scalar(284, 1) != 1
            or scalar(274, 1) != 1
        ):
            fail("use top-left contiguous grayscale or RGB/RGBA pages")
        if (
            len(bits) not in (1, channels)
            or len(set(bits)) != 1
            or len(sample) not in (1, channels)
            or len(set(sample)) != 1
        ):
            fail("sample types must be uniform within each page")
        dtype = {(1, 8): "uint8", (1, 16): "uint16", (3, 32): "float32", (3, 64): "float64"}.get(
            (sample[0], bits[0])
        )
        if dtype is None or compression not in LOSSLESS_COMPRESSION:
            fail("use uint8/16 or float32/64 with none/LZW/Deflate/PackBits compression")
        page = TiffPage((width, height), dtype, channels, compression)
        if page.nbytes > MAX_BYTES:
            fail("declared page samples exceed 128 MiB")
        if 324 in tags or 325 in tags:
            tile_width, tile_height = scalar(322), scalar(323)
            if (
                not 0 < tile_width <= 16384
                or not 0 < tile_height <= 16384
                or tile_width * tile_height * channels * bits[0] // 8 > MAX_BYTES
            ):
                fail("tile allocation exceeds the supported budget")
            offsets, counts = numbers(324), numbers(325)
            expected = (
                (width + tile_width - 1) // tile_width * ((height + tile_height - 1) // tile_height)
            )
        else:
            rows = scalar(278, height)
            if rows < 1:
                fail("rows per strip must be positive")
            offsets, counts = numbers(273), numbers(279)
            expected = (height + rows - 1) // rows
        if len(offsets) != len(counts) or len(offsets) != expected:
            fail("strip/tile counts differ from page dimensions")
        if any(
            size < 1 or start + size > len(data)
            for start, size in zip(offsets, counts, strict=True)
        ):
            fail("pixel strip/tile lies outside the buffer")
        pages.append(page)
        offset = next_offset
    if not pages:
        fail("no pages")
    return tuple(pages)
