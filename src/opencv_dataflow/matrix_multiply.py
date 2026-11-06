"""Compute weighted matrix products for numeric graph tables."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_MATRIX_FIELDS = ("left", "right", "addend")
_LIMIT = 1e100


def _configured(value):
    return not isinstance(value, (list, tuple)) or bool(value)


def _matrix(value, field):
    try:
        return finite_features(value, limit=_LIMIT, max_columns=100_000)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid {field} matrix: {exc}", field) from exc


class MultiplyMatrices(Block):
    type_id, title = "cv.multiply_matrices", "Multiply matrices"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = tuple(
        Port(key, "data", required=False, required_when_connected=True) for key in _MATRIX_FIELDS
    )
    outputs = (Port("matrix", "data"),)
    process_inputs_readonly = True
    defaults = {
        "left": [],
        "right": [],
        "addend": [],
        "product_weight": 1.0,
        "addend_weight": 1.0,
        "transpose_left": False,
        "transpose_right": False,
        "transpose_addend": False,
    }
    parameter_ranges = {
        "product_weight": NumericRange(-_LIMIT, _LIMIT),
        "addend_weight": NumericRange(-_LIMIT, _LIMIT),
    }
    parameter_help = {
        "left": "Left factor. A wired matrix takes precedence.",
        "right": "Right factor. A wired matrix takes precedence.",
        "addend": "Optional matrix with the product's shape after transposition.",
        "product_weight": "Scale applied to the matrix product.",
        "addend_weight": "Scale applied to the optional addend.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in _MATRIX_FIELDS:
            if _configured(p[field]):
                _matrix(p[field], field)

    def process(self, inputs):
        p = self.parameters
        left = _matrix(inputs.get("left", p["left"]), "left")
        right = _matrix(inputs.get("right", p["right"]), "right")
        left_shape = left.T.shape if p["transpose_left"] else left.shape
        right_shape = right.T.shape if p["transpose_right"] else right.shape
        if left_shape[1] != right_shape[0]:
            raise ValidationError(
                "Left columns must match right rows after transposition", "left", "right"
            )
        shape = (left_shape[0], right_shape[1])
        if shape[0] * shape[1] > MAX_SAMPLE_VALUES:
            raise ValidationError("Matrix product is limited to 2 million values", "left", "right")
        flags = cv2.GEMM_1_T if p["transpose_left"] else 0
        if p["transpose_right"]:
            flags |= cv2.GEMM_2_T
        addend, addend_weight = None, 0.0
        if "addend" in inputs or _configured(p["addend"]):
            addend = _matrix(inputs.get("addend", p["addend"]), "addend")
            addend_shape = addend.shape
            if p["transpose_addend"]:
                flags |= cv2.GEMM_3_T
                addend_shape = addend.T.shape
            if addend_shape != shape:
                raise ValidationError(
                    "Addend must match the product shape after transposition", "addend"
                )
            addend_weight = float(p["addend_weight"])
        result = cv2.gemm(
            left, right, float(p["product_weight"]), addend, addend_weight, flags=flags
        )
        if not np.isfinite(result).all():
            raise ValidationError("Matrix product produced nonfinite values. Rescale the inputs.")
        return Result(
            {"matrix": result.tolist()}, message=f"Computed {shape[0]} × {shape[1]} matrix"
        )
