"""Bounded summaries of image metadata reported by OpenCV."""

import struct

import cv2
import numpy as np

METADATA_NAMES = {
    cv2.IMAGE_METADATA_EXIF: "exif",
    cv2.IMAGE_METADATA_XMP: "xmp",
    cv2.IMAGE_METADATA_ICCP: "icc",
    cv2.IMAGE_METADATA_CICP: "cicp",
}
EXIF_TEXT_FIELDS = {
    0x010F: "make",
    0x0110: "model",
    0x0131: "software",
    0x0132: "date_time",
}


def ifd_entries(data, offset, order):
    if not 8 <= offset <= len(data) - 2:
        return ()
    count = struct.unpack_from(f"{order}H", data, offset)[0]
    start = offset + 2
    end = start + count * 12
    if count > 512 or end > len(data):
        return ()
    return struct.iter_unpack(f"{order}HHI4s", data[start:end])


def exif_text(data, count, value, byte_order):
    if not 1 <= count <= 512:
        return ""
    if count <= 4:
        content = value[:count]
    else:
        offset = int.from_bytes(value, byte_order)
        if offset < 8 or offset + count > len(data):
            return ""
        content = data[offset : offset + count]
    return content.split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


def summarize_exif(data):
    if data.startswith(b"Exif\0\0"):
        data = data[6:]
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return {}
    order, byte_order = ("<", "little") if data[:2] == b"II" else (">", "big")
    magic, root = struct.unpack_from(f"{order}HI", data, 2)
    if magic != 42:
        return {}
    fields = {}
    capture_ifd = 0
    for tag, kind, count, value in ifd_entries(data, root, order):
        if tag == 0x0112 and kind == 3 and count == 1:
            orientation = int.from_bytes(value[:2], byte_order)
            if 1 <= orientation <= 8:
                fields["orientation"] = orientation
        elif tag == 0x8769 and kind == 4 and count == 1:
            capture_ifd = int.from_bytes(value, byte_order)
        elif kind == 2 and tag in EXIF_TEXT_FIELDS:
            text = exif_text(data, count, value, byte_order)
            if text:
                fields[EXIF_TEXT_FIELDS[tag]] = text
    for tag, kind, count, value in ifd_entries(data, capture_ifd, order):
        if tag == 0x9003 and kind == 2:
            text = exif_text(data, count, value, byte_order)
            if text:
                fields["date_time_original"] = text
            break
    return fields


def orient_image(image, orientation):
    if orientation in (2, 3, 4):
        return cv2.flip(image, {2: 1, 3: -1, 4: 0}[orientation])
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def decode_image_metadata(buffer, apply_orientation):
    image, types, chunks = cv2.imdecodeWithMetadata(
        np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED
    )
    summary = {"chunks": [], "exif": {}, "orientation_applied": False}
    if image is None or not image.size:
        return image, summary
    for kind, chunk in zip(types, chunks, strict=True):
        name = METADATA_NAMES.get(int(kind), "unknown")
        summary["chunks"].append({"type": name, "byte_count": int(chunk.nbytes)})
        if name == "exif" and chunk.nbytes <= 1_048_576:
            for key, value in summarize_exif(chunk.tobytes()).items():
                summary["exif"].setdefault(key, value)
    summary["decoded_size"] = [image.shape[1], image.shape[0]]
    orientation = summary["exif"].get("orientation", 1)
    if (
        apply_orientation
        and orientation != 1
        and buffer.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"))
    ):
        image = orient_image(image, orientation)
        summary["orientation_applied"] = True
    return image, summary
