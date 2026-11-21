"""Validate scalar disparity rasters and their missing-sample encodings."""

import numpy as np

from .images import DepthArray
from .parameter_validation import ValidationError
from .raster_evaluation import MAX_EVALUATION_PIXELS

_DISPARITY_DTYPES = (np.uint8, np.int16, np.uint16, np.float32, np.float64)


def disparity_image(image):
    values = np.asarray(image)
    if (
        isinstance(image, DepthArray)
        or values.ndim != 2
        or values.dtype not in _DISPARITY_DTYPES
        or not 0 < values.size <= MAX_EVALUATION_PIXELS
    ):
        raise ValidationError(
            "Disparity needs a monochrome uint8, int16, uint16 or float grid "
            "of up to 16,777,216 pixels"
        )
    return values


def valid_samples(values, invalid_code, field):
    valid = np.isfinite(values)
    if invalid_code is not None:
        if values.dtype.kind == "f":
            representable = abs(invalid_code) <= float(np.finfo(values.dtype).max)
        else:
            limits = np.iinfo(values.dtype)
            representable = invalid_code.is_integer() and limits.min <= invalid_code <= limits.max
        if not representable:
            raise ValidationError(
                "Invalid disparity code must be representable in the source dtype", field
            )
        valid &= values != invalid_code
    return valid
