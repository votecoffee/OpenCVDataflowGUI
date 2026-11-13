"""Native APNG/GIF/WebP file and buffer blocks sharing one animation contract."""

import copy
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .animation_codec import (
    EXTENSIONS,
    GIF_DITHER,
    GIF_QUALITY,
    MAX_BYTES,
    MAX_PIXELS,
    PNG_STRATEGY,
    EncodedAnimation,
    decode,
    encode,
    frame_group,
    prepare,
    whole,
)
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame

DECODE_DEFAULTS = {
    "start_frame": 0,
    "frame_count": 0,
    "preview_frame": 0,
    "color_format": "unchanged",
    "pixel_depth": "unchanged",
    "maximum_pixels": MAX_PIXELS,
}
DECODE_CHOICES = {"color_format": COLOR_FORMATS, "pixel_depth": PIXEL_DEPTHS}
DECODE_RANGES = {
    "start_frame": NumericRange(0, 32766, whole=True),
    "frame_count": NumericRange(0, 64, whole=True),
    "preview_frame": NumericRange(0, 63, whole=True),
    "maximum_pixels": NumericRange(1, MAX_PIXELS, whole=True),
}
DECODE_HELP = {
    "frame_count": "Zero reads the remaining frames, subject to the 64-frame budget. "
    "A positive value explicitly selects a bounded subset.",
    "preview_frame": "Zero-based frame within the selected subset shown by Image.",
    "maximum_pixels": "Maximum total selected pixels before native decoding; not per frame.",
}


def local_path(block):
    value = block.parameters["path"]
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Choose an animation file path", "path")
    path = Path(value)
    if block.parameters["path_base"] == "project" and not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError("Save or open the project before using a project-relative path")
        path = block.project_dir / path
    return path.resolve()


def drawings(block, details, index):
    if not block.generate_graphics:
        return {}
    text = f"Frame {details['start_frame'] + index}: {details['durations_ms'][index]} ms"
    return {
        "animation frame": GraphicDrawing(
            (graphic_object("text", [(4, 18)], (60, 220, 160), text=text),),
            source="image",
            from_output=True,
        )
    }


class _DecodeAnimation(Block):
    outputs = (
        Port("frames", "any"),
        Port("image"),
        Port("durations", "data"),
        Port("frame_count", "integer"),
        Port("info", "data"),
        Port("still_image", required=False),
    )
    defaults, parameter_choices = DECODE_DEFAULTS, DECODE_CHOICES
    parameter_ranges, parameter_help = DECODE_RANGES, DECODE_HELP
    process_inputs_readonly = parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [p.name for p in self.outputs if p.name != "still_image"]

    def decoded_result(
        self, data, contexts=(), context=None, file_api=False, formats=(), labels=()
    ):
        p = self.parameters
        animation, details = decode(
            data, p["start_frame"], p["frame_count"], p["maximum_pixels"], file_api=file_api
        )
        index = int(p["preview_frame"])
        if index >= details["frame_count"]:
            raise ValidationError("Preview frame is outside the selected subset", "preview_frame")
        group = frame_group(
            animation,
            details,
            f"{self.id}:{hashlib.sha256(data).hexdigest()[:12]}",
            context,
            contexts,
            labels,
        )
        for local, packet in enumerate(group.items):
            color = p["color_format"]
            if color == "unchanged" and formats:
                color = formats[details["start_frame"] + local]
            packet.value = format_image(packet.value, color, p["pixel_depth"])
        image = group.items[index]
        still = animation.still_image
        if still is not None:
            still = format_image(still, p["color_format"], p["pixel_depth"])
        payloads = {"image": image.payload}
        if still is not None:
            payloads["still_image"] = {"spatial": frame(details["size"], f"{self.id}:poster")}
        result = Result(
            {
                "frames": group,
                "image": image.value,
                "durations": details["durations_ms"],
                "frame_count": len(group.items),
                "info": details,
                "still_image": still if still is not None else NO_VALUE,
            },
            payloads=payloads,
            drawings=drawings(self, details, index),
            message=f"Decoded {len(group.items)} {details['codec'].upper()} frames",
        )
        return result


class AnimationFile(_DecodeAnimation):
    type_id, title, category = "input.animation_file", "Load animation frames", "Input"
    block_type, subtype = "Input", "Files"
    defaults = {"path": "", "path_base": "project", **DECODE_DEFAULTS}
    parameter_choices = {"path_base": ("project", "working_directory"), **DECODE_CHOICES}

    def issue_source_ticket(self):
        ticket = super().issue_source_ticket()
        if self.ticket_settings["mode"] == "auto_number":
            name = Path(self.parameters["path"]).name
            if name:
                ticket.update(
                    kind="string",
                    value=name,
                    session=f"{ticket['session']}:animation:{ticket['sequence']}",
                )
        return ticket

    def process(self, inputs):
        path = local_path(self)
        with path.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
        context = {"ticket": self._source_ticket} if self._source_ticket else {}
        result = self.decoded_result(data, context=context, file_api=True)
        result.outputs["info"].update(filename=path.name)
        return result


class DecodeAnimation(_DecodeAnimation):
    type_id, title, category = (
        "cv.decode_animation",
        "Decode animation from memory",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Codecs"
    inputs = (Port("encoded", "any"),)
    input_help = {
        "encoded": "EncodedAnimation packet, bytes, or a uint8 vector. "
        "Frame contexts from Encode animation are preserved."
    }

    def process(self, inputs):
        value = inputs["encoded"]
        contexts = value.contexts if isinstance(value, EncodedAnimation) else ()
        data = value.buffer if isinstance(value, EncodedAnimation) else value
        if isinstance(data, np.ndarray):
            if (
                data.dtype != np.uint8
                or data.ndim not in (1, 2)
                or min(data.shape) != 1
                and data.ndim == 2
            ):
                raise ValidationError("Encoded arrays must be uint8 vectors")
            if data.nbytes > MAX_BYTES:
                raise ValidationError("Animation buffers exceed 32 MiB")
            data = data.tobytes()
        return self.decoded_result(
            data,
            contexts,
            self.input_payloads.get("encoded", {}),
            formats=value.color_formats if isinstance(value, EncodedAnimation) else (),
            labels=value.labels if isinstance(value, EncodedAnimation) else (),
        )


class _EncodeAnimation(Block):
    inputs = (
        Port("frames", "any"),
        Port("durations", "data", required=False, required_when_connected=True),
        Port("still_image", required=False, required_when_connected=True),
    )
    defaults = {
        "codec": "apng",
        "duration_mode": "payload_or_default",
        "default_duration_ms": 100,
        "durations_ms": [],
        "loop_mode": "payload_or_default",
        "loop_count": 0,
        "background_mode": "payload_or_default",
        "background_bgra": [0, 0, 0, 0],
        "maximum_pixels": MAX_PIXELS,
        "gif_timing": "exact",
        "gif_quality": "palette_256",
        "gif_dither": "none",
        "gif_color_table": "global",
        "gif_transparency_threshold": 1,
        "webp_mode": "lossless",
        "webp_quality": 90,
        "png_compression": -1,
        "png_strategy": "native",
    }
    parameter_choices = {
        "codec": tuple(EXTENSIONS),
        "duration_mode": ("payload_or_default", "uniform", "explicit"),
        "loop_mode": ("payload_or_default", "override"),
        "background_mode": ("payload_or_default", "override"),
        "gif_timing": ("exact", "floor_10ms"),
        "gif_quality": tuple(GIF_QUALITY),
        "gif_dither": tuple(GIF_DITHER),
        "gif_color_table": ("global", "local"),
        "webp_mode": ("lossless", "lossy"),
        "png_strategy": ("native", *PNG_STRATEGY),
    }
    parameter_ranges = {
        "default_duration_ms": NumericRange(1, 655350, whole=True),
        "loop_count": NumericRange(0, 65535, whole=True),
        "maximum_pixels": NumericRange(1, MAX_PIXELS, whole=True),
        "gif_transparency_threshold": NumericRange(0, 255, whole=True),
        "webp_quality": NumericRange(1, 100, whole=True),
        "png_compression": NumericRange(-1, 9, whole=True),
    }
    parameter_help = {
        "duration_mode": "Use frame payload timing with a default, one uniform duration, "
        "or the explicit list. Wired Durations overrides this choice.",
        "loop_count": "Zero is infinite; GIF viewers may count repeats differently.",
        "loop_mode": "Use the first frame's loop metadata when available, or override it.",
        "background_mode": "Use the first frame's background metadata, or override it.",
        "background_bgra": "Four background bytes. GIF requires zero; APNG alpha must be zero. "
        "WebP packing is normalized and independently checked.",
        "gif_timing": "Exact requires 10-ms units; floor_10ms quantizes and reports both lists.",
        "gif_quality": "Native fast modes or palette size. GIF has binary alpha and reduced color.",
        "gif_dither": "Palette-mode dither; fast modes select their own native dither algorithm.",
        "gif_transparency_threshold": "Alpha below this becomes transparent; zero disables alpha.",
        "webp_mode": "Lossless uses animation QUALITY=101. The still-image LOSSLESS_MODE flag "
        "is ignored by this backend.",
        "webp_quality": "Lossy WebP only, 1..100.",
        "png_compression": "APNG only; -1 keeps the native compression default, otherwise 0..9.",
        "png_strategy": "APNG zlib strategy. Native APNG ignores PNG filter and buffer-size flags.",
        "maximum_pixels": "Total pixel budget. Readback verifies frame count, timing and loops.",
    }
    input_help = {
        "frames": "FlowGroup of 2..64 matching uint8 images; each retains its own context.",
        "durations": "Optional millisecond list, one entry per frame.",
        "still_image": "APNG poster, separate from timed frames. Use grayscale or BGRA to avoid "
        "a native RGB poster defect.",
    }
    process_inputs_readonly = parallel_safe = True
    payload_reference = "frames"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["frames"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        bg = p["background_bgra"]
        if not isinstance(bg, list) or len(bg) != 4:
            raise ValidationError("Background BGRA requires four bytes", "background_bgra")
        for channel in bg:
            whole(channel, "background_bgra", 0, 255)
        durations = p["durations_ms"]
        if not isinstance(durations, list) or len(durations) > 64:
            raise ValidationError(
                "Durations must be a list with at most 64 entries", "durations_ms"
            )
        for duration in durations:
            whole(duration, "durations_ms", 1, 655350)

    def encoded_result(self, inputs, *, file_api=False):
        animation, requested = prepare(
            inputs["frames"], self.parameters, inputs.get("durations"), inputs.get("still_image")
        )
        data, details = encode(animation, self.parameters, requested, file_api=file_api)
        contexts = tuple(copy.deepcopy(item.payload) for item in inputs["frames"].items)
        packet = EncodedAnimation(
            self.parameters["codec"],
            data,
            contexts,
            tuple(image_format(item.value) for item in inputs["frames"].items),
            inputs["frames"].labels,
        )
        return packet, details


class EncodeAnimation(_EncodeAnimation):
    type_id, title, category = (
        "cv.encode_animation",
        "Encode animation in memory",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Codecs"
    outputs = (Port("encoded", "any"), Port("byte_count", "integer"), Port("info", "data"))

    def process(self, inputs):
        packet, details = self.encoded_result(inputs)
        return Result(
            {"encoded": packet, "byte_count": len(packet.buffer), "info": details},
            message=f"Encoded and verified {details['frame_count']} animation frames",
        )


class WriteAnimationFile(_EncodeAnimation):
    type_id, title, category = "output.animation_file", "Write animation file", "Output"
    block_type, subtype = "Output", "Images"
    defaults = {"path": "output.png", "path_base": "project", **_EncodeAnimation.defaults}
    parameter_choices = {
        "path_base": ("project", "working_directory"),
        **_EncodeAnimation.parameter_choices,
    }
    outputs = (Port("info", "data"), Port("frame_count", "integer"), Port("byte_count", "integer"))
    preview_safe = False
    parallel_safe = False

    def process(self, inputs):
        path = local_path(self)
        allowed = (
            (".png", ".apng")
            if self.parameters["codec"] == "apng"
            else (EXTENSIONS[self.parameters["codec"]],)
        )
        if path.suffix.lower() not in allowed:
            raise ValidationError(
                "Output extension must match the selected animation codec", "path", "codec"
            )
        if path.exists():
            raise FileExistsError("Choose a new path; existing animation files are preserved")
        packet, details = self.encoded_result(inputs, file_api=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(packet.buffer)
            # Exclusive publication preserves a destination created while encoding was in progress.
            import os

            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        details.update(filename=path.name)
        return Result(
            {
                "info": details,
                "frame_count": details["frame_count"],
                "byte_count": len(packet.buffer),
            },
            message=f"Wrote verified animation {path.name}",
        )


ANIMATION_BLOCKS = (AnimationFile, DecodeAnimation, EncodeAnimation, WriteAnimationFile)
