"""Native codec options and typed in-memory image packets."""

from dataclasses import dataclass

import cv2
import numpy as np

from .core import Block, Port, Result
from .float_codecs import (
    FLOAT_CODEC_CHOICES,
    FLOAT_CODEC_DEFAULTS,
    FLOAT_CODEC_HELP,
    FLOAT_CODEC_RANGES,
    float_codec_options,
    validate_float_codec,
)
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value, integer

CODEC_DEFAULTS = {
    "jpeg_quality": -1,
    "jpeg_progressive": False,
    "jpeg_optimize": False,
    "jpeg_restart_interval": 0,
    "jpeg_luma_quality": -1,
    "jpeg_chroma_quality": -1,
    "png_compression": -1,
    "png_strategy": "native",
    "png_bilevel": False,
    "webp_quality": -1,
    **FLOAT_CODEC_DEFAULTS,
}
CODEC_CHOICES = {
    "png_strategy": ("native", "default", "filtered", "huffman", "rle", "fixed"),
    **FLOAT_CODEC_CHOICES,
}
CODEC_HELP = {
    "jpeg_quality": "-1 preserves the codec default; otherwise 0..100. JPEG only.",
    "png_compression": "-1 preserves the codec default; otherwise 0..9. PNG only.",
    "webp_quality": "-1 preserves the codec default; otherwise 1..100. WebP only.",
    **FLOAT_CODEC_HELP,
}


def codec_options(ext, parameters):
    options = []
    if ext in (".jpg", ".jpeg"):
        for key, flag in (
            ("jpeg_quality", cv2.IMWRITE_JPEG_QUALITY),
            ("jpeg_luma_quality", cv2.IMWRITE_JPEG_LUMA_QUALITY),
            ("jpeg_chroma_quality", cv2.IMWRITE_JPEG_CHROMA_QUALITY),
        ):
            value = integer(parameters[key], key, -1, 100)
            if value >= 0:
                options.extend([flag, value])
        options.extend(
            [
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                int(parameters["jpeg_progressive"]),
                cv2.IMWRITE_JPEG_OPTIMIZE,
                int(parameters["jpeg_optimize"]),
                cv2.IMWRITE_JPEG_RST_INTERVAL,
                integer(parameters["jpeg_restart_interval"], "Restart interval", 0, 65535),
            ]
        )
    elif ext == ".png":
        compression = integer(parameters["png_compression"], "PNG compression", -1, 9)
        if compression >= 0:
            options.extend([cv2.IMWRITE_PNG_COMPRESSION, compression])
        strategy = parameters["png_strategy"]
        if strategy != "native":
            value = {
                "default": cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
                "filtered": cv2.IMWRITE_PNG_STRATEGY_FILTERED,
                "huffman": cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY,
                "rle": cv2.IMWRITE_PNG_STRATEGY_RLE,
                "fixed": cv2.IMWRITE_PNG_STRATEGY_FIXED,
            }[strategy]
            options.extend([cv2.IMWRITE_PNG_STRATEGY, value])
        if parameters["png_bilevel"]:
            options.extend([cv2.IMWRITE_PNG_BILEVEL, 1])
    elif ext == ".webp":
        quality = integer(parameters["webp_quality"], "WebP quality", -1, 100)
        if quality >= 0:
            if quality == 0:
                raise ValueError("WebP quality must be 1..100 or -1 for the native default")
            options.extend([cv2.IMWRITE_WEBP_QUALITY, quality])
    options.extend(float_codec_options(ext, parameters))
    return options


FORMATS = {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff", "hdr": ".hdr", "exr": ".exr"}
MAGIC_BYTES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "tiff": (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"),
    "hdr": (b"#?RADIANCE", b"#?RGBE"),
    "exr": b"\x76\x2f\x31\x01",
}
MAX_ENCODED_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216
ENCODED_FORMATS = ("gray", "bgr", "rgb", "bgra", "rgba")


def encode_image_buffer(source, extension, p, *, allow_signed_tiff=False):
    """Use one depth/availability policy for file and in-memory image output."""
    depths = (np.uint8, np.uint16, np.float32, np.float64)
    if allow_signed_tiff and extension in (".tif", ".tiff"):
        depths += (np.int16,)
    image = image_value(source, depths, canonical=True)
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValidationError("Image exceeds the 16,777,216-pixel codec limit")
    if not np.isfinite(image).all():
        raise ValidationError("Color conversion produced nonfinite samples")
    if extension in (".jpg", ".jpeg") and (image.dtype != np.uint8 or image.shape[2:] == (4,)):
        raise ValidationError("JPEG requires uint8 grayscale or RGB/BGR without alpha")
    if extension == ".png":
        if image.dtype not in (np.uint8, np.uint16):
            raise ValidationError("PNG requires uint8 or uint16; use TIFF for floating samples")
        if p["png_bilevel"] and (
            image.ndim != 2 or image.dtype != np.uint8 or np.any((image != 0) & (image != 255))
        ):
            raise ValidationError("Bilevel PNG needs a uint8 monochrome mask containing 0 and 255")
    if extension in (".webp", ".bmp") and image.dtype != np.uint8:
        raise ValidationError("This image codec requires explicit conversion to uint8")
    validate_float_codec(image, extension, p)
    try:
        available = cv2.haveImageWriter(extension)
    except cv2.error as exc:
        raise ValidationError(
            f"Image codec {extension} is unavailable or disabled in this OpenCV build"
        ) from exc
    if not available:
        raise ValidationError(f"Image codec {extension} is unavailable in this OpenCV build")
    try:
        ok, encoded = cv2.imencode(extension, image, codec_options(extension, p))
    except cv2.error as exc:
        raise ValidationError(f"OpenCV could not encode {extension}: {exc}") from exc
    if not ok or encoded.size == 0:
        raise ValidationError("OpenCV could not encode the image")
    if encoded.nbytes > MAX_ENCODED_BYTES:
        raise ValidationError("Encoded image exceeds the 128 MiB codec limit")
    return encoded


@dataclass(frozen=True)
class EncodedImage:
    codec: str
    buffer: bytes
    image_size: tuple[int, int]
    color_format: str

    def __post_init__(self):
        if (
            not isinstance(self.codec, str)
            or self.codec not in FORMATS
            or not isinstance(self.color_format, str)
            or self.color_format not in ENCODED_FORMATS
            or not isinstance(self.buffer, bytes)
            or not 0 < len(self.buffer) <= MAX_ENCODED_BYTES
            or not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(type(size) is not int or size <= 0 for size in self.image_size)
            or self.image_size[0] * self.image_size[1] > MAX_IMAGE_PIXELS
        ):
            raise ValidationError("Invalid encoded image packet or size limit exceeded")
        if not self.buffer.startswith(MAGIC_BYTES[self.codec]):
            raise ValidationError("Encoded bytes do not match the declared image codec")

    def describe(self):
        return {
            "model": "encoded_image_v1",
            "codec": self.codec,
            "image_size": list(self.image_size),
            "color_format": self.color_format,
            "byte_count": len(self.buffer),
        }


class EncodeImage(Block):
    type_id, title, category = "cv.encode_image", "Encode image in memory", "Image processing"
    block_type, subtype = "Image processing", "Codecs"
    inputs = (Port("image"),)
    outputs = (
        Port("encoded", "encoded_image", visual_type="data"),
        Port("byte_count", "number"),
    )
    defaults = {
        "codec": "png",
        **{key: value for key, value in CODEC_DEFAULTS.items() if key != "webp_quality"},
    }
    parameter_choices = {"codec": tuple(FORMATS), **CODEC_CHOICES}
    parameter_ranges = {
        "jpeg_quality": NumericRange(-1, 100, whole=True),
        "jpeg_luma_quality": NumericRange(-1, 100, whole=True),
        "jpeg_chroma_quality": NumericRange(-1, 100, whole=True),
        "jpeg_restart_interval": NumericRange(0, 65535, whole=True),
        "png_compression": NumericRange(-1, 9, whole=True),
        **FLOAT_CODEC_RANGES,
    }
    parameter_help = {
        key: description for key, description in CODEC_HELP.items() if key != "webp_quality"
    }

    def process(self, inputs):
        source = inputs["image"]
        codec = self.parameters["codec"]
        color = image_format(source)
        if color not in ENCODED_FORMATS:
            raise ValidationError("Encode a grayscale, RGB/BGR, or RGBA/BGRA image")
        height, width = source.shape[:2]
        extension = FORMATS[codec]
        encoded = encode_image_buffer(source, extension, self.parameters)
        packet = EncodedImage(codec, encoded.tobytes(), (width, height), color)
        return Result(
            {"encoded": packet, "byte_count": len(packet.buffer)},
            message=f"{codec.upper()} image encoded to {len(packet.buffer):,} bytes",
        )


class DecodeImage(ImageTool):
    type_id, title, subtype = "cv.decode_image", "Decode image from memory", "Codecs"
    inputs = EncodeImage.outputs[:1]
    defaults = {"color_format": "unchanged", "pixel_depth": "unchanged"}
    parameter_choices = {"color_format": COLOR_FORMATS, "pixel_depth": PIXEL_DEPTHS}
    parameter_help = {
        "color_format": "Unchanged restores the encoder's logical channel order.",
        "pixel_depth": "Unchanged preserves the bit depth stored by the codec.",
    }

    def process(self, inputs):
        packet = inputs["encoded"]
        decoded = cv2.imdecode(np.frombuffer(packet.buffer, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            raise ValidationError("OpenCV could not decode the image packet")
        if decoded.shape[:2] != packet.image_size[::-1]:
            raise ValidationError("Decoded image size differs from the packet")
        canonical = {"rgb": "bgr", "rgba": "bgra"}.get(packet.color_format, packet.color_format)
        if image_format(decoded) != canonical:
            raise ValidationError("Decoded channel layout differs from the packet")
        p = self.parameters
        color = packet.color_format if p["color_format"] == "unchanged" else p["color_format"]
        return Result({"image": format_image(decoded, color, p["pixel_depth"])})
