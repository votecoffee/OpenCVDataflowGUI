"""Bounded animation buffers and verified native frame/timing transport."""

import copy
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, frame, validate_payload

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_777_216
MAX_FRAMES = 64
EXTENSIONS = {"apng": ".png", "gif": ".gif", "webp": ".webp"}
GIF_QUALITY = {"fast_no_dither": 1, "fast_floyd": 2, **{f"palette_{2**i}": i for i in range(3, 9)}}
GIF_DITHER = {"most": -1, "normal": 0, "less": 1, "minimal": 2, "none": 3}
PNG_STRATEGY = {"default": 0, "filtered": 1, "huffman": 2, "rle": 3, "fixed": 4}


def whole(value, name, low, high):
    return NumericRange(low, high, whole=True).check(value, name, name)


def header(data):
    """Inspect the container before native allocation; Pillow does not decode pixels here."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
        raise ValidationError("Animation buffers require 1 byte..32 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                codec = {"PNG": "apng", "GIF": "gif", "WEBP": "webp"}.get(image.format)
                if codec is None:
                    raise ValidationError("Use a supported APNG, GIF or WebP animation")
                width, height = image.size
                if not 0 < width <= 8192 or not 0 < height <= 8192 or width * height > MAX_PIXELS:
                    raise ValidationError("Animation canvas exceeds the pixel or 8192-axis limit")
                poster = bool(image.info.get("default_image", False))
                count = image.n_frames - int(poster)
                if not 1 <= count <= 32767:
                    raise ValidationError("Animation container has an unsupported frame count")
                background = image.info.get("background")
                if codec == "webp" and isinstance(background, tuple):
                    r, g, b, a = background
                    background = [b, g, r, a]
                else:
                    background = None
                return {
                    "codec": codec,
                    "size": [width, height],
                    "total_frames": count,
                    "has_still_image": poster,
                    "background_bgra": background,
                    "container_loop": image.info.get("loop"),
                }
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValidationError(f"Invalid or oversized animation container: {exc}") from exc


@dataclass(frozen=True)
class EncodedAnimation:
    codec: str
    buffer: bytes
    contexts: tuple = ()
    color_formats: tuple = ()
    labels: tuple = ()

    def __post_init__(self):
        details = header(self.buffer)
        if self.codec != details["codec"] or not isinstance(self.contexts, tuple):
            raise ValidationError("Animation packet codec or frame contexts are invalid")
        if self.contexts and len(self.contexts) != details["total_frames"]:
            raise ValidationError("Animation frame contexts do not match the encoded frame count")
        if len(self.contexts) > MAX_FRAMES:
            raise ValidationError("Animation packet contexts exceed 64 frames")
        if (
            not isinstance(self.color_formats, tuple)
            or self.color_formats
            and (
                len(self.color_formats) != details["total_frames"]
                or any(c not in ("gray", "bgr", "rgb", "bgra", "rgba") for c in self.color_formats)
            )
        ):
            raise ValidationError("Animation logical color formats do not match the encoded frames")
        for context in self.contexts:
            validate_payload(context)
        if (
            not isinstance(self.labels, tuple)
            or self.labels
            and (
                len(self.labels) != details["total_frames"]
                or any(not isinstance(label, str) for label in self.labels)
            )
        ):
            raise ValidationError("Animation labels do not match the encoded frames")
        object.__setattr__(self, "contexts", copy.deepcopy(self.contexts))

    def describe(self):
        return {
            "model": "encoded_animation_v1",
            "codec": self.codec,
            "color_formats": list(self.color_formats),
            "labels": list(self.labels),
            "byte_count": len(self.buffer),
            "sha256": hashlib.sha256(self.buffer).hexdigest(),
        }


def png_still_image(data):
    # Native animation readers in this wheel misread the separate APNG poster.
    # Retain its original PNG chunks and decode IDAT as a standalone native PNG.
    chunks, offset = [data[:8]], 8
    for _ in range(65536):
        if offset + 12 > len(data):
            break
        size = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + size
        if end > len(data):
            break
        kind = data[offset + 4 : offset + 8]
        if kind not in (b"acTL", b"fcTL", b"fdAT"):
            chunks.append(data[offset:end])
        offset = end
        if kind == b"IEND":
            return cv2.imdecode(np.frombuffer(b"".join(chunks), np.uint8), cv2.IMREAD_UNCHANGED)
    raise ValidationError("APNG poster has truncated or excessive PNG chunks")


def decode(data, start=0, count=0, maximum_pixels=MAX_PIXELS, *, file_api=False):
    details = header(data)
    start = whole(start, "start_frame", 0, 32766)
    count = whole(count, "frame_count", 0, MAX_FRAMES)
    maximum_pixels = whole(maximum_pixels, "maximum_pixels", 1, MAX_PIXELS)
    available = details["total_frames"] - start
    if available <= 0:
        raise ValidationError("Start frame is outside the animation", "start_frame")
    selected = min(count, available) if count else available
    pixel_count = (selected + int(details["has_still_image"])) * np.prod(details["size"])
    if selected > MAX_FRAMES or pixel_count > maximum_pixels:
        raise ValidationError(
            "Selected animation exceeds 64 frames or the total pixel budget",
            "frame_count",
            "maximum_pixels",
        )
    try:
        if file_api:
            # Snapshot bytes first, including Unicode source paths, then use native file I/O.
            with TemporaryDirectory(prefix="ocv-animation-") as temporary:
                path = Path(temporary) / ("input" + EXTENSIONS[details["codec"]])
                path.write_bytes(data)
                ok, animation = cv2.imreadanimation(str(path), start, selected)
        else:
            ok, animation = cv2.imdecodeanimation(np.frombuffer(data, np.uint8), start, selected)
    except cv2.error as exc:
        raise ValidationError(f"Native animation decoding failed: {exc}") from exc
    if not ok or len(animation.frames) != selected or len(animation.durations) != selected:
        raise ValidationError("Native animation decoder did not return the selected frame sequence")
    if details["has_still_image"]:
        still = png_still_image(data)
        if (
            still is None
            or still.dtype != np.uint8
            or still.shape[:2] != tuple(details["size"][::-1])
        ):
            raise ValidationError("APNG poster differs from the frame canvas or uint8 depth")
        animation.still_image = still
    for image in animation.frames:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape[:2] != tuple(details["size"][::-1])
            or image_format(image) not in ("gray", "bgr", "bgra")
        ):
            raise ValidationError("Decoded animation frame depth, layout or canvas differs")
    durations = [whole(int(d), "duration_ms", 1, 655350) for d in animation.durations]
    background = details["background_bgra"]
    if background is None:
        background = [int(v) for v in animation.bgcolor]
    details.update(
        start_frame=start,
        frame_count=selected,
        durations_ms=durations,
        loop_count=int(animation.loop_count),
        background_bgra=background,
        elapsed_ms=sum(durations),
        byte_count=len(data),
    )
    return animation, details


def pixels(value):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.size == 0
        or image_format(value) not in ("gray", "bgr", "rgb", "bgra", "rgba")
    ):
        raise ValidationError(
            "Animation frames require uint8 grayscale, RGB/BGR or RGBA/BGRA images"
        )
    return np.ascontiguousarray(opencv_image(value))


def options(p):
    if p["codec"] == "webp":
        return [
            cv2.IMWRITE_WEBP_QUALITY,
            101 if p["webp_mode"] == "lossless" else int(p["webp_quality"]),
        ]
    if p["codec"] == "gif":
        return [
            cv2.IMWRITE_GIF_DITHER,
            GIF_DITHER[p["gif_dither"]],
            cv2.IMWRITE_GIF_QUALITY,
            GIF_QUALITY[p["gif_quality"]],
            cv2.IMWRITE_GIF_COLORTABLE,
            int(p["gif_color_table"] == "local"),
            cv2.IMWRITE_GIF_TRANSPARENCY,
            int(p["gif_transparency_threshold"]),
        ]
    result = []
    if p["png_compression"] >= 0:
        result += [cv2.IMWRITE_PNG_COMPRESSION, int(p["png_compression"])]
    if p["png_strategy"] != "native":
        result += [cv2.IMWRITE_PNG_STRATEGY, PNG_STRATEGY[p["png_strategy"]]]
    return result


def prepare(group, p, durations=None, still=None):
    if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= MAX_FRAMES:
        raise ValidationError("Connect an ordered group of 2..64 image frames to Frames")
    values = [item.value for item in group.items]
    if any(not isinstance(value, np.ndarray) or value.ndim not in (2, 3) for value in values):
        raise ValidationError("Animation frames must be images")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValidationError("Animation frames must share the same canvas and channel layout")
    height, width = shape[:2]
    if (
        max(height, width) > 8192
        or (len(values) + int(still is not None)) * height * width > p["maximum_pixels"]
    ):
        raise ValidationError(
            "Animation frames exceed the canvas or total pixel budget", "maximum_pixels"
        )
    images = [pixels(value) for value in values]
    if p["codec"] in ("gif", "webp") and images[0].ndim == 2:
        images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in images]
    mode = p["duration_mode"]
    if durations is None:
        durations = (
            p["durations_ms"]
            if mode == "explicit"
            else [
                item.payload.get("animation_frame", {}).get("duration_ms", p["default_duration_ms"])
                if mode == "payload_or_default"
                else p["default_duration_ms"]
                for item in group.items
            ]
        )
    if not isinstance(durations, (tuple, list)) or len(durations) != len(images):
        raise ValidationError("Provide one duration in milliseconds for each frame", "durations_ms")
    requested = [whole(d, "duration_ms", 1, 655350) for d in durations]
    actual = list(requested)
    if p["codec"] == "gif":
        if p["gif_timing"] == "exact" and any(d % 10 for d in actual):
            raise ValidationError(
                "GIF durations must be multiples of 10 ms; choose floor_10ms to quantize",
                "gif_timing",
                "durations_ms",
            )
        actual = [d // 10 * 10 for d in actual]
        if any(d == 0 for d in actual):
            raise ValidationError("GIF frames need at least 10 ms", "durations_ms")
    animation = cv2.Animation()
    animation.frames, animation.durations = images, actual
    first = group.items[0].payload.get("animation_frame", {})
    loop = (
        first.get("loop_count", p["loop_count"])
        if p["loop_mode"] == "payload_or_default"
        else p["loop_count"]
    )
    animation.loop_count = whole(loop, "loop_count", 0, 65535)
    bg = (
        first.get("background_bgra", p["background_bgra"])
        if p["background_mode"] == "payload_or_default"
        else p["background_bgra"]
    )
    if not isinstance(bg, list) or len(bg) != 4:
        raise ValidationError("Background BGRA requires four bytes", "background_bgra")
    bg = [whole(c, "background_bgra", 0, 255) for c in bg]
    if p["codec"] == "gif" and any(bg):
        raise ValidationError(
            "Native GIF background selection is unavailable; use zero BGRA", "background_bgra"
        )
    if p["codec"] == "apng" and bg[3] != 0:
        raise ValidationError("APNG background stores BGR only; use zero alpha", "background_bgra")
    if p["codec"] == "apng" and images[0].ndim == 2 and any(bg):
        raise ValidationError("Native APNG grayscale background selection is unavailable")
    # Native WebP currently packs Scalar as ARGB, unlike the public BGRA contract.
    animation.bgcolor = bg[::-1] if p["codec"] == "webp" else bg
    if still is not None:
        if p["codec"] != "apng":
            raise ValidationError("A separate still image is supported only for APNG")
        poster = pixels(still)
        if poster.shape != images[0].shape:
            raise ValidationError("APNG still image must share the frame canvas and layout")
        if poster.ndim == 3 and poster.shape[2] == 3:
            raise ValidationError(
                "Native APNG RGB posters are unreliable; use grayscale or BGRA frames and poster"
            )
        animation.still_image = poster
    return animation, requested


def encode(animation, p, requested, *, file_api=False):
    extension, native_options = EXTENSIONS[p["codec"]], options(p)
    if not cv2.haveImageWriter(extension):
        raise ValidationError(f"This OpenCV build has no {p['codec']} writer")
    try:
        if file_api:
            with TemporaryDirectory(prefix="ocv-animation-") as temporary:
                path = Path(temporary) / ("output" + extension)
                ok = cv2.imwriteanimation(str(path), animation, native_options)
                with path.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
        else:
            ok, encoded = cv2.imencodeanimation(extension, animation, native_options)
            data = encoded.tobytes()
    except (cv2.error, OSError) as exc:
        raise ValidationError(f"Native animation encoding failed: {exc}") from exc
    if not ok:
        raise ValidationError("Native animation encoding failed")
    decoded, details = decode(data, maximum_pixels=p["maximum_pixels"])
    if (
        len(decoded.frames) != len(animation.frames)
        or details["durations_ms"] != list(animation.durations)
        or details["loop_count"] != animation.loop_count
    ):
        raise ValidationError(
            "Native encoder changed frame count, timing or loop count (identical frames may merge)"
        )
    expected_bg = [int(v) for v in animation.bgcolor]
    if p["codec"] == "webp" and details["background_bgra"] != expected_bg[::-1]:
        raise ValidationError("Native WebP writer did not preserve the background color")
    poster = animation.still_image
    if poster is not None and (
        decoded.still_image is None or not np.array_equal(poster, decoded.still_image)
    ):
        raise ValidationError("Native APNG writer did not preserve the still image")
    exact = p["codec"] == "apng" or (p["codec"] == "webp" and p["webp_mode"] == "lossless")
    errors = []
    for before, after in zip(animation.frames, decoded.frames, strict=True):
        if before.ndim == 3 and before.shape[2] == 4:
            if after.ndim != 3 or after.shape[2] != 4:
                if exact:
                    raise ValidationError("Native encoder changed the alpha layout")
                before, after = before[..., :3], after[..., :3]
            elif exact:
                if not np.array_equal(before[..., 3], after[..., 3]):
                    raise ValidationError("Native lossless encoder changed alpha samples")
                before, after = before[before[..., 3] != 0, :3], after[before[..., 3] != 0, :3]
        elif after.ndim == 3 and after.shape[2] == 4:
            after = after[..., :3]
        error = int(np.max(np.abs(before.astype(np.int16) - after.astype(np.int16)), initial=0))
        if exact and error:
            raise ValidationError("Native lossless encoder changed visible image samples")
        errors.append(error)
    details.update(
        requested_durations_ms=requested,
        maximum_sample_errors=errors,
        exact_visible_samples=exact,
        hidden_transparent_rgb_qualified=False,
    )
    return data, details


def frame_group(animation, details, owner, context=None, contexts=(), labels=()):
    packets = []
    offset = 0
    for local, (image, duration) in enumerate(
        zip(animation.frames, details["durations_ms"], strict=True)
    ):
        index = details["start_frame"] + local
        payload = copy.deepcopy(contexts[index] if contexts else (context or {}))
        if not contexts:
            for key in ("spatial", "pyramid", "raw_sensor", "alpha_representation"):
                payload.pop(key, None)
            payload["spatial"] = frame(details["size"], f"{owner}:animation:{index}")
        payload["animation_frame"] = {
            "index": index,
            "duration_ms": duration,
            "selection_offset_ms": offset,
            "loop_count": details["loop_count"],
            "codec": details["codec"],
            "background_bgra": list(details["background_bgra"]),
        }
        image = ImageArray(image, image_format(image), payload)
        packets.append(ValuePacket(image, payload))
        offset += duration
    return FlowGroup(
        tuple(packets),
        tuple(
            labels[details["start_frame"] + i] if labels else f"frame-{details['start_frame'] + i}"
            for i in range(len(packets))
        ),
    )
