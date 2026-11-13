"""Precision-aware TIFF/HDR/EXR options shared by memory and file encoders."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError

TIFF_COMPRESSIONS = {
    "none": cv2.IMWRITE_TIFF_COMPRESSION_NONE,
    "lzw": cv2.IMWRITE_TIFF_COMPRESSION_LZW,
    "deflate": cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE,
    "packbits": cv2.IMWRITE_TIFF_COMPRESSION_PACKBITS,
}
TIFF_PREDICTORS = {
    "none": cv2.IMWRITE_TIFF_PREDICTOR_NONE,
    "horizontal": cv2.IMWRITE_TIFF_PREDICTOR_HORIZONTAL,
    "floating": cv2.IMWRITE_TIFF_PREDICTOR_FLOATINGPOINT,
}
EXR_COMPRESSIONS = {
    name.lower(): getattr(cv2, f"IMWRITE_EXR_COMPRESSION_{name}")
    for name in ("NO", "RLE", "ZIPS", "ZIP", "PIZ", "PXR24", "B44", "B44A", "DWAA", "DWAB")
}
FLOAT_CODEC_DEFAULTS = {
    "tiff_compression": "none",
    "tiff_predictor": "native",
    "tiff_rows_per_strip": 0,
    "hdr_compression": "rle",
    "exr_storage": "float32",
    "exr_compression": "zip",
    "exr_dwa_level": 45,
}
FLOAT_CODEC_CHOICES = {
    "tiff_compression": tuple(TIFF_COMPRESSIONS),
    "tiff_predictor": ("native", *TIFF_PREDICTORS),
    "hdr_compression": ("rle", "none"),
    "exr_storage": ("float32", "float16"),
    "exr_compression": tuple(EXR_COMPRESSIONS),
}
FLOAT_CODEC_RANGES = {
    "tiff_rows_per_strip": NumericRange(0, 16_777_216, whole=True),
    "exr_dwa_level": NumericRange(0, 1000, whole=True),
}
FLOAT_CODEC_HELP = {
    "tiff_compression": (
        "TIFF only. Explicit lossless compression avoids the native float32-color LogLuv default."
    ),
    "tiff_predictor": (
        "TIFF LZW/deflate only. Floating prediction requires floating samples; "
        "horizontal requires integer samples."
    ),
    "tiff_rows_per_strip": "TIFF strip height; zero keeps the native default.",
    "hdr_compression": (
        "Radiance HDR RLE or no compression. Both use lossy shared-exponent RGBE samples."
    ),
    "exr_storage": (
        "EXR needs a capable OpenCV build. Float32 retains sample precision; float16 "
        "quantizes and requires values within Â±65504."
    ),
    "exr_compression": (
        "EXR codec. PXR24, B44/B44A and DWA variants can be lossy; ZIP is the default."
    ),
    "exr_dwa_level": "EXR DWAA/DWAB compression level; unused by other compression modes.",
}


def float_codec_options(extension, p):
    if extension in (".tif", ".tiff"):
        options = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_COMPRESSIONS[p["tiff_compression"]]]
        if p["tiff_predictor"] != "native" and p["tiff_compression"] in ("lzw", "deflate"):
            options.extend([cv2.IMWRITE_TIFF_PREDICTOR, TIFF_PREDICTORS[p["tiff_predictor"]]])
        if p["tiff_rows_per_strip"]:
            options.extend([cv2.IMWRITE_TIFF_ROWSPERSTRIP, int(p["tiff_rows_per_strip"])])
        return options
    if extension == ".hdr":
        return [
            cv2.IMWRITE_HDR_COMPRESSION,
            cv2.IMWRITE_HDR_COMPRESSION_RLE
            if p["hdr_compression"] == "rle"
            else cv2.IMWRITE_HDR_COMPRESSION_NONE,
        ]
    if extension == ".exr":
        options = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_FLOAT
            if p["exr_storage"] == "float32"
            else cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            EXR_COMPRESSIONS[p["exr_compression"]],
        ]
        if p["exr_compression"] in ("dwaa", "dwab"):
            options.extend([cv2.IMWRITE_EXR_DWA_COMPRESSION_LEVEL, int(p["exr_dwa_level"])])
        return options
    return []


def validate_float_codec(image, extension, p, *, predictor_field="tiff_predictor"):
    if extension in (".tif", ".tiff") and p["tiff_compression"] in ("lzw", "deflate"):
        predictor = p["tiff_predictor"]
        if (predictor == "floating" and image.dtype.kind != "f") or (
            predictor == "horizontal" and image.dtype.kind == "f"
        ):
            raise ValidationError(
                "TIFF predictor does not match the image sample type", predictor_field
            )
    if extension == ".hdr":
        if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
            raise ValidationError("HDR requires float32 RGB/BGR without alpha")
        if image.min() < 0 or image.max() >= 2.0**127:
            raise ValidationError(
                "HDR requires nonnegative samples below 2**127; use TIFF for signed values"
            )
    if extension == ".exr":
        if image.dtype != np.float32:
            raise ValidationError(
                "EXR requires float32 input; choose float16 storage explicitly if needed"
            )
        if p["exr_storage"] == "float16" and np.max(np.abs(image)) > 65504:
            raise ValidationError("EXR float16 storage is limited to Â±65504", "exr_storage")
