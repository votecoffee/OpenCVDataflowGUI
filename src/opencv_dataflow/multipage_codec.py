"""Immutable multipage TIFF packets, per-page context, and checked native codecs."""

import copy
import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np

from .float_codecs import float_codec_options, validate_float_codec
from .images import format_image, image_format, opencv_image
from .parameter_validation import ValidationError
from .payloads import FlowGroup, ValuePacket, check_spatial, frame, metadata, validate_payload
from .processing import image_value
from .tiff_directory import MAX_BYTES, MAX_PAGES, MAX_PIXELS, directory

MAX_CONTEXT_BYTES = 2 * 1024**2
FORMATS = {1: ("gray",), 3: ("bgr", "rgb"), 4: ("bgra", "rgba")}


@dataclass(frozen=True)
class EncodedPages:
    buffer: bytes
    metadata_json: bytes

    def __post_init__(self):
        pages = directory(self.buffer)
        if (
            not isinstance(self.metadata_json, bytes)
            or not 0 < len(self.metadata_json) <= MAX_CONTEXT_BYTES
        ):
            raise ValidationError("Multipage context must contain 1 byte..2 MiB of UTF-8 JSON")
        try:
            sidecar = json.loads(self.metadata_json)
            if (
                not isinstance(sidecar, dict)
                or sidecar.get("version") != 1
                or not isinstance(sidecar.get("pages"), list)
                or len(sidecar["pages"]) != len(pages)
                or len(pages) > MAX_PAGES
            ):
                raise ValueError("context must describe exactly 1..64 encoded pages")
            for entry, page in zip(sidecar["pages"], pages, strict=True):
                if (
                    not isinstance(entry, dict)
                    or entry.get("size") != list(page.size)
                    or entry.get("dtype") != page.dtype
                    or entry.get("color_format") not in FORMATS[page.channels]
                    or not isinstance(entry.get("label"), str)
                    or len(entry["label"]) > 256
                ):
                    raise ValueError("page size, depth, encoding or label does not match the TIFF")
                validate_payload(entry["payload"])
                check_spatial(entry["payload"], page.size)
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise ValidationError(f"Invalid multipage context: {exc}") from exc

    @property
    def contexts(self):
        # Stored JSON bytes are immutable; callers always receive private dictionaries.
        return json.loads(self.metadata_json)["pages"]

    def describe(self):
        return {
            "model": "encoded_pages_v1",
            "codec": "tiff",
            "byte_count": len(self.buffer),
            "sha256": hashlib.sha256(self.buffer).hexdigest(),
            "pages": [
                {**page.describe(), **{k: v for k, v in entry.items() if k != "payload"}}
                for entry, page in zip(self.contexts, directory(self.buffer), strict=True)
            ],
        }


def buffer_bytes(value):
    if isinstance(value, EncodedPages):
        return value.buffer
    if isinstance(value, np.ndarray):
        if (
            value.dtype != np.uint8
            or value.ndim not in (1, 2)
            or value.ndim == 2
            and min(value.shape) != 1
            or value.nbytes > MAX_BYTES
        ):
            raise ValidationError("Encoded TIFF arrays must be uint8 vectors up to 128 MiB")
        return value.tobytes()
    if not isinstance(value, bytes):
        raise ValidationError("Connect an EncodedPages packet, TIFF bytes or a uint8 vector")
    return value


def read_page(data, index, page):
    try:
        ok, images = cv2.imdecodemulti(
            np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED, range=(index, index + 1)
        )
    except cv2.error as exc:
        raise ValidationError(f"Cannot decode TIFF page {index}: {exc}") from exc
    shape = tuple(page.size[::-1]) + (() if page.channels == 1 else (page.channels,))
    if (
        not ok
        or len(images) != 1
        or images[0].shape != shape
        or images[0].dtype != np.dtype(page.dtype)
        or not np.isfinite(images[0]).all()
    ):
        raise ValidationError(
            f"Decoded TIFF page {index} differs from its declared layout or has nonfinite samples"
        )
    return images[0]


def encode(group, p):
    if not isinstance(group, FlowGroup):
        raise ValidationError("Connect an ordered image group to Pages")
    if not 1 <= len(group.items) <= MAX_PAGES or len(group.items) != len(group.labels):
        raise ValidationError("Connect an ordered group of 1..64 labeled images")
    images, entries, total = [], [], 0
    options = {
        "tiff_compression": p["compression"],
        "tiff_predictor": p["predictor"],
        "tiff_rows_per_strip": int(p["rows_per_strip"]),
    }
    for packet, label in zip(group.items, group.labels, strict=True):
        page = packet.value
        if not isinstance(page, np.ndarray):
            raise ValidationError("Every multipage entry must be an image")
        values = image_value(page, (np.uint8, np.uint16, np.float32, np.float64))
        color = image_format(page)
        if (
            values.size == 0
            or color not in FORMATS[1 if values.ndim == 2 else values.shape[2]]
            or not isinstance(label, str)
            or len(label) > 256
        ):
            raise ValidationError(
                "Pages need gray/RGB(A)/BGR(A) pixels and labels up to 256 characters"
            )
        width, height = values.shape[1], values.shape[0]
        if max(width, height) > 16384 or width * height > MAX_PIXELS:
            raise ValidationError("Each page is limited to 16384 per axis and 16777216 pixels")
        total += values.nbytes
        if total > MAX_BYTES:
            raise ValidationError("Uncompressed TIFF pages exceed the 128 MiB batch limit")
        validate_float_codec(values, ".tiff", options, predictor_field="predictor")
        context = metadata(packet)
        validate_payload(context)
        check_spatial(context, [width, height])
        entries.append(
            {
                "size": [width, height],
                "dtype": str(values.dtype),
                "color_format": color,
                "label": label,
                "payload": context,
            }
        )
        images.append(page)
    sidecar = json.dumps(
        {"version": 1, "pages": entries}, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(sidecar) > MAX_CONTEXT_BYTES:
        raise ValidationError("Multipage context exceeds 2 MiB")
    native_options = float_codec_options(".tiff", options)
    if p["x_dpi"]:
        native_options.extend(
            [
                cv2.IMWRITE_TIFF_RESUNIT,
                cv2.IMWRITE_TIFF_RESOLUTION_UNIT_INCH,
                cv2.IMWRITE_TIFF_XDPI,
                int(p["x_dpi"]),
                cv2.IMWRITE_TIFF_YDPI,
                int(p["y_dpi"]),
            ]
        )
    try:
        ok, encoded = cv2.imencodemulti(".tiff", [opencv_image(i) for i in images], native_options)
    except cv2.error as exc:
        raise ValidationError(f"Cannot encode multipage TIFF: {exc}") from exc
    if not ok or encoded.size == 0 or encoded.nbytes > MAX_BYTES:
        raise ValidationError("Native multipage encoding failed or exceeds 128 MiB")
    data = encoded.tobytes()
    pages = directory(data)
    if len(pages) != len(images):
        raise ValidationError("Native encoder changed the page count")
    for index, (image, page) in enumerate(zip(images, pages, strict=True)):
        decoded = read_page(data, index, page)
        expected = opencv_image(image)
        if (
            decoded.shape != expected.shape
            or decoded.dtype != expected.dtype
            or decoded.tobytes() != np.ascontiguousarray(expected).tobytes()
        ):
            raise ValidationError(f"Native TIFF encoding changed samples on page {index}")
    return EncodedPages(data, sidecar)


def decode(value, p, context, root_id):
    data = buffer_bytes(value)
    pages = directory(data)
    start, count, stride = (int(p[k]) for k in ("start_page", "page_count", "stride"))
    if start >= len(pages):
        raise ValidationError("Start page is outside the TIFF", "start_page")
    indexes = list(range(start, len(pages), stride))
    if count:
        indexes = indexes[:count]
    if len(indexes) > MAX_PAGES:
        raise ValidationError("Select at most 64 pages using Page count and Stride", "page_count")
    if int(p["preview_page"]) >= len(indexes):
        raise ValidationError("Preview page is outside the selected subset", "preview_page")
    sidecar = value.contexts if isinstance(value, EncodedPages) else None
    pixels, sample_bytes, work_bytes = 0, 0, 0
    for index in indexes:
        page = pages[index]
        color = p["color_format"]
        channels = (
            page.channels
            if color == "unchanged"
            else {"gray": 1, "bgra": 4, "rgba": 4}.get(color, 3)
        )
        depth = page.dtype if p["pixel_depth"] == "unchanged" else p["pixel_depth"]
        if color in ("hsv", "lab") and depth not in ("uint8", "float32"):
            raise ValidationError(
                "HSV/Lab requires uint8 or float32 depth", "pixel_depth", "color_format"
            )
        pixels += page.pixels
        output_bytes = page.pixels * channels * np.dtype(depth).itemsize
        sample_bytes += page.nbytes + output_bytes
        # Conservative allowance for per-page float64 conversion and color intermediates.
        work_bytes = max(work_bytes, page.pixels * 8 * (page.channels + channels + 4))
    estimated = sample_bytes + work_bytes
    if pixels > int(p["maximum_pixels"]) or estimated > int(p["maximum_decoded_bytes"]):
        raise ValidationError(
            "Selected TIFF pages exceed the pixel or decoded workspace budget",
            "maximum_pixels",
            "maximum_decoded_bytes",
            "page_count",
        )
    digest = hashlib.sha256(data).hexdigest()
    packets, labels = [], []
    for index in indexes:
        page = pages[index]
        image = read_page(data, index, page)
        color = p["color_format"]
        if color == "unchanged" and sidecar:
            color = sidecar[index]["color_format"]
        image = format_image(image, color, p["pixel_depth"])
        if sidecar:
            payload = copy.deepcopy(sidecar[index]["payload"])
            label = sidecar[index]["label"]
        else:
            payload = copy.deepcopy(context)
            for key in ("spatial", "pyramid", "raw_sensor", "alpha"):
                payload.pop(key, None)
            label = f"page_{index}"
        payload.setdefault("spatial", frame(page.size, f"{root_id}:{digest[:16]}:page:{index}"))
        payload["multipage"] = {
            "codec": "tiff",
            "page_index": index,
            "total_pages": len(pages),
            "container_sha256": digest,
        }
        image.metadata = copy.deepcopy(payload)
        packets.append(ValuePacket(image, payload))
        labels.append(label)
    group = FlowGroup(tuple(packets), tuple(labels))
    info = {
        "codec": "tiff",
        "total_pages": len(pages),
        "selected_indexes": indexes,
        "page_count": len(indexes),
        "byte_count": len(data),
        "sha256": digest,
        "restored_page_context": sidecar is not None,
        "selected_pixels": pixels,
        "workspace_upper_bound_bytes": estimated,
        "pages": [pages[i].describe() for i in indexes],
    }
    return group, info
