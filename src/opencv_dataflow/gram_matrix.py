"""Compute dot products between rows or columns of a numeric table."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_PRECISION = {"float64": (np.float64, cv2.CV_64F), "float32": (np.float32, cv2.CV_32F)}


def _configured(value):
    return not isinstance(value, (list, tuple)) or bool(value)


def _matrix(value, field):
    try:
        return finite_features(value, max_columns=100_000)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid {field}: {exc}", field) from exc


class GramMatrix(Block):
    type_id, title = "cv.gram_matrix", "Gram matrix"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (
        Port("samples", "data", required=False, required_when_connected=True),
        Port("offset", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("matrix", "data"), Port("offset", "data"))
    process_inputs_readonly = True
    defaults = {
        "samples": [],
        "offset": [],
        "axis": "columns",
        "centering": "none",
        "normalization": "none",
        "scale": 1.0,
        "precision": "float64",
    }
    parameter_choices = {
        "axis": ("columns", "rows"),
        "centering": ("none", "column_mean", "row_mean"),
        "normalization": ("none", "population", "sample"),
        "precision": tuple(_PRECISION),
    }
    parameter_ranges = {"scale": NumericRange(-1e100, 1e100)}
    parameter_help = {
        "samples": "Rectangular table. A wired table takes precedence.",
        "offset": "A scalar table, row, column, or full table to subtract before multiplication.",
        "axis": "Columns computes AᵀA. Rows computes AAᵀ.",
        "centering": "Subtract row or column means. A supplied offset takes precedence.",
        "normalization": "Divide by vector length for population or length minus one for sample.",
        "scale": "Additional multiplier applied to the result.",
        "precision": "Floating-point precision for input conversion and the product.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("samples", "offset"):
            if _configured(p[field]):
                _matrix(p[field], field)

    def process(self, inputs):
        p = self.parameters
        samples = _matrix(inputs.get("samples", p["samples"]), "samples")
        by_columns = p["axis"] == "columns"
        size = samples.shape[1] if by_columns else samples.shape[0]
        length = samples.shape[0] if by_columns else samples.shape[1]
        if size**2 > MAX_SAMPLE_VALUES:
            raise ValidationError("Gram matrix output is limited to 2 million values", "axis")
        divisor = 1
        if p["normalization"] != "none":
            divisor = length - (p["normalization"] == "sample")
            if divisor == 0:
                raise ValidationError(
                    "Sample normalization requires vectors of length two or more", "normalization"
                )

        precision, depth = _PRECISION[p["precision"]]
        samples = samples.astype(precision, copy=False)
        offset = None
        if "offset" in inputs or _configured(p["offset"]):
            offset = _matrix(inputs.get("offset", p["offset"]), "offset")
            rows, columns = samples.shape
            if offset.shape[0] not in (1, rows) or offset.shape[1] not in (1, columns):
                raise ValidationError(
                    "Offset must match both sample dimensions or have size one on an axis", "offset"
                )
        elif p["centering"] != "none":
            axis = 0 if p["centering"] == "column_mean" else 1
            offset = np.mean(samples, axis=axis, dtype=np.float64, keepdims=True)
        if offset is not None:
            offset = offset.astype(precision, copy=False)

        product = cv2.mulTransposed(
            samples, by_columns, delta=offset, scale=float(p["scale"]) / divisor, dtype=depth
        )
        if not np.isfinite(product).all():
            raise ValidationError(
                "Gram matrix exceeded the working precision", "scale", "precision"
            )
        return Result(
            {"matrix": product.tolist(), "offset": [[0.0]] if offset is None else offset.tolist()},
            message=f"Computed {size} × {size} Gram matrix",
        )
