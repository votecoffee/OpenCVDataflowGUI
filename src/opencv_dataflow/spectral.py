"""Spectral coefficient packets and reversible image transforms."""

from dataclasses import dataclass

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, convert_depth
from .parameter_validation import ValidationError
from .processing import image_value

DCT_FLAGS = {"2d": 0, "rows": cv2.DCT_ROWS}
DCT_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)


def dct_shape(shape, mode):
    height, width = shape
    return height + (height % 2 if mode == "2d" and height > 1 else 0), (
        width + (width % 2 if width > 1 else 0)
    )


@dataclass(frozen=True)
class DCTCoefficients:
    values: np.ndarray
    mode: str
    image_shape: tuple[int, int]
    image_depth: str

    def describe(self):
        return {
            "model": "image_dct_v1",
            "mode": self.mode,
            "image_shape": list(self.image_shape),
            "image_depth": self.image_depth,
            "values": self.values,
        }


class DiscreteCosineTransform(Block):
    type_id, title = "cv.dct", "Discrete cosine transform (DCT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = (Port("image"),)
    outputs = (Port("coefficients", "dct", visual_type="data"),)
    defaults = {"mode": "2d", "odd_size": "pad_even"}
    parameter_choices = {"mode": tuple(DCT_FLAGS), "odd_size": ("pad_even", "reject")}
    parameter_help = {
        "mode": "Transform the entire image or transform each row independently.",
        "odd_size": "Extend odd transformed axes by repeating the last pixel, or reject them.",
    }

    def process(self, inputs):
        image = image_value(inputs["image"], DCT_DEPTHS)
        if image.ndim != 2 or image.size == 0:
            raise ValidationError("DCT needs a nonempty monochrome image")
        mode = self.parameters["mode"]
        height, width = image.shape
        rows, columns = dct_shape(image.shape, mode)
        if (rows, columns) != image.shape and self.parameters["odd_size"] == "reject":
            raise ValidationError("DCT needs even transformed dimensions or Pad even", "odd_size")
        working = convert_depth(image, "float32") if image.dtype.kind == "u" else image
        if (rows, columns) != image.shape:
            working = np.pad(working, ((0, rows - height), (0, columns - width)), mode="edge")
        coefficients = cv2.dct(working, flags=DCT_FLAGS[mode])
        if not np.isfinite(coefficients).all():
            raise ValidationError("DCT produced nonfinite coefficients. Check the image range")
        graphics = {}
        if self.generate_graphics:
            graphics["magnitude"] = cv2.normalize(
                np.log1p(np.abs(coefficients)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
        return Result(
            {"coefficients": DCTCoefficients(coefficients, mode, image.shape, image.dtype.name)},
            graphics=graphics,
        )


class InverseDiscreteCosineTransform(Block):
    type_id, title = "cv.idct", "Inverse discrete cosine transform (IDCT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = DiscreteCosineTransform.outputs
    outputs = (Port("image"),)

    def process(self, inputs):
        packet = inputs["coefficients"]
        mode, shape, depth = packet.mode, packet.image_shape, packet.image_depth
        coefficients = packet.values
        if (
            not isinstance(mode, str)
            or mode not in DCT_FLAGS
            or depth not in ("uint8", "uint16", "float32", "float64")
            or not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)
            or not isinstance(coefficients, np.ndarray)
            or coefficients.shape != dct_shape(shape, mode)
            or coefficients.dtype not in (np.float32, np.float64)
            or not np.isfinite(coefficients).all()
        ):
            raise ValidationError("Invalid DCT coefficient values or metadata")
        height, width = shape
        output = cv2.idct(coefficients, flags=DCT_FLAGS[mode])[:height, :width]
        if not np.isfinite(output).all():
            raise ValidationError("IDCT produced nonfinite pixels. Check the coefficient range")
        if depth in ("uint8", "uint16"):
            output = convert_depth(output, depth)
        else:
            output = output.astype(depth, copy=False)
            if not np.isfinite(output).all():
                raise ValidationError("IDCT pixels exceed the requested floating-point depth")
        return Result({"image": ImageArray(output, "gray")})


SPECTRAL_BLOCKS = (DiscreteCosineTransform, InverseDiscreteCosineTransform)
