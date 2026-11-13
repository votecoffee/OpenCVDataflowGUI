"""Immutable bounded native metadata chunks and verified codec writes."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from .image_metadata import METADATA_NAMES, summarize_exif
from .parameter_validation import ValidationError

MAX_CHUNK_BYTES = 1_048_576
MAX_METADATA_BYTES = 4 * MAX_CHUNK_BYTES
XMP_JPEG_PREFIX = b"http://ns.adobe.com/xap/1.0/\0"
ORIENTATIONS = {
    "normal": 1,
    "mirror_horizontal": 2,
    "rotate_180": 3,
    "mirror_vertical": 4,
    "transpose": 5,
    "rotate_clockwise": 6,
    "transverse": 7,
    "rotate_counterclockwise": 8,
}
CODEC_TYPES = {".png": {0, 1, 2, 3}, ".jpg": {0, 1, 2}, ".jpeg": {0, 1, 2}}


def canonical_chunk(kind, raw):
    if kind == cv2.IMAGE_METADATA_EXIF:
        raw = raw[6:] if raw.startswith(b"Exif\0\0") else raw
        order = "<" if raw[:2] == b"II" else ">" if raw[:2] == b"MM" else None
        if order is None or len(raw) < 8 or struct.unpack_from(order + "H", raw, 2)[0] != 42:
            raise ValidationError("EXIF must contain a classic TIFF header")
        offset = struct.unpack_from(order + "I", raw, 4)[0]
        if not 8 <= offset <= len(raw) - 2:
            raise ValidationError("EXIF has an invalid root directory")
        count = struct.unpack_from(order + "H", raw, offset)[0]
        if count > 512 or offset + 2 + count * 12 + 4 > len(raw):
            raise ValidationError("EXIF root directory is truncated or oversized")
    elif kind == cv2.IMAGE_METADATA_XMP:
        if raw.startswith(XMP_JPEG_PREFIX):
            raw = raw[len(XMP_JPEG_PREFIX) :]
        # Native PNG returns a C-string terminator, which is not XML content.
        raw = raw.rstrip(b"\0")
        try:
            ET.fromstring(raw.decode("utf8"))
        except (ET.ParseError, UnicodeError) as exc:
            raise ValidationError("XMP must be well-formed UTF-8 XML") from exc
    elif kind == cv2.IMAGE_METADATA_ICCP:
        if len(raw) < 128 or raw[36:40] != b"acsp" or int.from_bytes(raw[:4], "big") != len(raw):
            raise ValidationError("ICC profile header or declared byte size is invalid")
    elif kind == cv2.IMAGE_METADATA_CICP and (len(raw) != 4 or raw[3] not in (0, 1)):
        raise ValidationError("CICP requires four bytes and a 0/1 full-range flag")
    return raw


@dataclass(frozen=True)
class ImageMetadata:
    chunks: tuple[tuple[int, bytes], ...]
    image_size: tuple[int, int] | None = None
    pixels_oriented: bool = False

    def __post_init__(self):
        if not isinstance(self.chunks, tuple) or len(self.chunks) > 4:
            raise ValidationError("Use at most four unique metadata chunks")
        normalized, seen = [], set()
        for chunk in self.chunks:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise ValidationError("Metadata chunks must be (native type, bytes) pairs")
            kind, raw = chunk
            if type(kind) is not int or kind not in METADATA_NAMES or kind in seen:
                raise ValidationError("Unsupported or duplicate metadata type")
            if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_CHUNK_BYTES:
                raise ValidationError("Each metadata chunk requires 1 byte..1 MiB")
            normalized.append((kind, canonical_chunk(kind, raw)))
            seen.add(kind)
        if sum(len(raw) for _, raw in normalized) > MAX_METADATA_BYTES:
            raise ValidationError("Metadata exceeds the 4 MiB limit")
        if self.image_size is not None and (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(type(n) is not int or n < 1 for n in self.image_size)
            or self.image_size[0] * self.image_size[1] > 16_777_216
        ):
            raise ValidationError("Metadata image_size is invalid")
        if type(self.pixels_oriented) is not bool:
            raise ValidationError("Metadata orientation state must be boolean")
        object.__setattr__(self, "chunks", tuple(normalized))

    def describe(self):
        exif = next((summarize_exif(raw) for kind, raw in self.chunks if kind == 0), {})
        return {
            "model": "image_metadata_v1",
            "image_size": list(self.image_size) if self.image_size else None,
            "pixels_oriented": self.pixels_oriented,
            "exif": exif,
            "chunks": [
                {
                    "type": METADATA_NAMES[kind],
                    "byte_count": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                for kind, raw in self.chunks
            ],
        }


def native_chunks(packet, extension, image):
    if not isinstance(packet, ImageMetadata):
        raise ValidationError("Metadata requires an ImageMetadata packet, not a summary dictionary")
    if extension not in CODEC_TYPES:
        raise ValidationError("Metadata writes are qualified for PNG and JPEG only")
    if packet.pixels_oriented:
        raise ValidationError(
            "Oriented pixels need fresh metadata; create a new normal-orientation packet"
        )
    if packet.image_size is not None and packet.image_size != (image.shape[1], image.shape[0]):
        raise ValidationError("Metadata belongs to a different stored image size")
    for kind, raw in packet.chunks:
        if kind not in CODEC_TYPES[extension]:
            raise ValidationError(f"{METADATA_NAMES[kind]} metadata is unsupported by {extension}")
        if kind == cv2.IMAGE_METADATA_ICCP:
            space = b"GRAY" if image.ndim == 2 else b"RGB "
            if raw[16:20] != space:
                raise ValidationError("ICC color space must match the stored gray/RGB image")
        if extension in (".jpg", ".jpeg") and kind in (0, 1):
            limit = 65527 if kind == 0 else 65533 - len(XMP_JPEG_PREFIX) - 1
            if len(raw) > limit:
                raise ValidationError("JPEG EXIF/XMP exceeds its single-marker byte limit")
    return [kind for kind, _ in packet.chunks], [
        # The native PNG writer passes XMP to a C-string API, so own its terminator.
        np.frombuffer(
            ((XMP_JPEG_PREFIX if extension in (".jpg", ".jpeg") else b"") + raw + b"\0")
            if kind == cv2.IMAGE_METADATA_XMP
            else raw,
            np.uint8,
        )
        for kind, raw in packet.chunks
    ]


def verify_metadata(buffer, requested, size):
    image, types, chunks = cv2.imdecodeWithMetadata(
        np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED
    )
    if image is None or image.shape[:2] != size:
        raise ValidationError("Metadata codec verification could not recover the stored image size")
    actual = ImageMetadata(
        tuple((int(k), chunk.tobytes()) for k, chunk in zip(types, chunks, strict=True))
    )
    expected, observed = dict(requested.chunks), dict(actual.chunks)
    missing = [METADATA_NAMES[k] for k, raw in expected.items() if observed.get(k) != raw]
    if missing:
        raise ValidationError(
            "Native codec did not preserve requested metadata: " + ", ".join(missing)
        )


def write_metadata_file(path, image, options, packet, maximum_bytes):
    """Native file API stages privately; verified bytes publish with exclusive creation."""
    path = Path(path)
    types, chunks = native_chunks(packet, path.suffix.lower(), image)
    if path.exists():
        raise FileExistsError("Choose a new path; existing images are preserved")
    stream = NamedTemporaryFile(
        prefix=".metadata-", suffix=path.suffix, dir=path.parent, delete=False
    )
    staged = Path(stream.name)
    stream.close()
    try:
        if not cv2.imwriteWithMetadata(str(staged), image, types, chunks, options):
            raise ValidationError("OpenCV could not write the image and metadata")
        if not 0 < staged.stat().st_size <= maximum_bytes:
            raise ValidationError("Metadata image exceeds the encoded byte limit")
        data = staged.read_bytes()
        verify_metadata(data, packet, image.shape[:2])
        with path.open("xb") as output:
            output.write(data)
    finally:
        staged.unlink(missing_ok=True)


def make_exif(orientation, software, description):
    fields = [(0x0112, 3, 1, struct.pack("<H", orientation) + b"\0\0")]
    entries, tail = [], bytearray()
    text = [(0x010E, description, "description"), (0x0131, software, "software")]
    count = 1 + sum(bool(value) for _, value, _ in text)
    offset = 8 + 2 + 12 * count + 4
    for tag, value, field in text:
        if not value:
            continue
        try:
            raw = value.encode("ascii") + b"\0"
        except UnicodeError as exc:
            raise ValidationError("EXIF text requires ASCII; use XMP for Unicode", field) from exc
        if b"\0" in raw[:-1] or len(raw) > 4096:
            raise ValidationError(
                "EXIF text must be at most 4095 ASCII characters without NUL", field
            )
        inline = raw.ljust(4, b"\0") if len(raw) <= 4 else struct.pack("<I", offset + len(tail))
        fields.append((tag, 2, len(raw), inline))
        if len(raw) > 4:
            tail.extend(raw)
    for tag, kind, length, raw in sorted(fields):
        entries.append(struct.pack("<HHI", tag, kind, length) + raw)
    return b"II" + struct.pack("<HIH", 42, 8, count) + b"".join(entries) + b"\0" * 4 + tail
