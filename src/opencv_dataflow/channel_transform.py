"""Apply a configurable linear transform to each selected pixel."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask


def _coefficients(value):
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "Enter a rectangular matrix of numeric coefficients", "matrix"
        ) from exc
    if matrix.shape not in ((1, 1), (1, 2), (1, 3), (1, 4), (3, 3), (3, 4)):
        raise ValidationError("Use one or three rows with weights and an optional offset", "matrix")
    if not np.isfinite(matrix).all():
        raise ValidationError("Matrix coefficients must be finite", "matrix")
    return matrix


class ChannelTransform(ImageTool):
    type_id, title, subtype = "cv.channel_transform", "Linear channel transform", "Color and depth"
    inputs = (*SELECTION_INPUTS, Port("matrix", "data", required=False))
    defaults = {"matrix": []}
    parameter_help = {
        "matrix": (
            "One row for gray or three for color, using stored channel order without alpha. "
            "Append an optional offset. Empty uses identity. A wired matrix takes precedence."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _coefficients(p["matrix"])

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or not image.size:
            raise ValidationError("Provide a nonempty grayscale, RGB, BGR, RGBA, or BGRA image")
        mask = selection_mask(inputs, image)
        matrix = _coefficients(inputs.get("matrix", self.parameters["matrix"]))
        if matrix is None:
            return Result({"image": source})
        channels = 1 if image.ndim == 2 else 3
        if matrix.shape[1] not in (channels, channels + 1):
            raise ValidationError(
                f"Each row needs {channels} input weight(s) and an optional offset", "matrix"
            )

        has_alpha = encoding in ("rgba", "bgra")
        signal = image[..., :3] if has_alpha else image
        grayscale = matrix.shape[0] == 1
        if mask is None:
            output = self._transform(signal, matrix)
        else:
            selected = mask != 0
            output = (
                np.zeros(image.shape[:2], dtype=image.dtype)
                if grayscale and channels > 1
                else signal.copy()
            )
            if selected.any():
                samples = signal[selected].reshape(-1, 1, channels)
                transformed = self._transform(samples, matrix)
                output[selected] = transformed.ravel() if grayscale else transformed[:, 0, :]
        if grayscale:
            encoding = "gray"
        elif has_alpha:
            output = np.dstack((output, image[..., 3]))
        return Result({"image": ImageArray(output, encoding)})

    @staticmethod
    def _transform(signal, matrix):
        output = cv2.transform(signal.astype(np.float64, copy=False), matrix)
        if not np.isfinite(output).all():
            raise ValidationError("Channel transform produced nonfinite values", "matrix")
        if signal.dtype.kind in "iu":
            limits = np.iinfo(signal.dtype)
            output = np.rint(output).clip(limits.min, limits.max)
        elif np.max(np.abs(output)) > np.finfo(signal.dtype).max:
            raise ValidationError("Channel transform exceeded the source pixel depth", "matrix")
        return output.astype(signal.dtype, copy=False)
