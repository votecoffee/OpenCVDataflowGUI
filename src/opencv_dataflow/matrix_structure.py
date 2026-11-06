"""Construct, measure, and complete the structure of real numeric matrices."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_MATRIX_OUTPUT = (Port("matrix", "data"),)
_DIMENSION = NumericRange(1, 100_000, whole=True)


def _diagonal(value, length):
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Diagonal must be a vector of real numbers", "diagonal") from exc
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.ravel()
    if vector.ndim != 1 or vector.size != length or vector.dtype.kind not in "iuf":
        raise ValidationError(f"Diagonal needs {length} real values", "diagonal", "rows", "columns")
    with np.errstate(over="ignore"):
        vector = vector.astype(np.float64, copy=False)
    if not np.isfinite(vector).all():
        raise ValidationError("Diagonal values must be finite float64 numbers", "diagonal")
    return vector


class CreateDiagonalMatrix(Block):
    type_id, title = "cv.create_diagonal_matrix", "Create diagonal matrix"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    process_inputs_readonly = True
    inputs = (Port("diagonal", "data", required=False, required_when_connected=True),)
    outputs = _MATRIX_OUTPUT
    defaults = {"rows": 3, "columns": 3, "diagonal_value": 1.0, "diagonal": []}
    parameter_ranges = {"rows": _DIMENSION, "columns": _DIMENSION, "diagonal_value": NumericRange()}
    parameter_help = {
        "rows": "Output row count. The matrix may be rectangular.",
        "columns": "Output column count. The diagonal length is the smaller dimension.",
        "diagonal_value": "Value on the main diagonal when no vector is supplied.",
        "diagonal": "Optional vector of main diagonal values. A wired vector takes precedence.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        rows, columns = int(p["rows"]), int(p["columns"])
        if rows * columns > MAX_SAMPLE_VALUES:
            raise ValidationError("Matrix is limited to 2 million values", "rows", "columns")
        if not (isinstance(p["diagonal"], (list, tuple)) and not p["diagonal"]):
            _diagonal(p["diagonal"], min(rows, columns))

    def process(self, inputs):
        p = self.parameters
        rows, columns = int(p["rows"]), int(p["columns"])
        value = inputs.get("diagonal", p["diagonal"])
        if isinstance(value, (list, tuple)) and not value and "diagonal" not in inputs:
            matrix = np.empty((rows, columns), dtype=np.float64)
            cv2.setIdentity(matrix, float(p["diagonal_value"]))
        else:
            vector = _diagonal(value, min(rows, columns))
            matrix = np.zeros((rows, columns), dtype=np.float64)
            positions = np.arange(vector.size)
            matrix[positions, positions] = vector
        return Result({"matrix": matrix.tolist()}, message=f"Created {rows} × {columns} matrix")


class _MatrixInput(Block):
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("matrix", "data", required=False, required_when_connected=True),)
    defaults = {"matrix": []}
    parameter_help = {
        "matrix": "Rectangular table of finite real numbers. A wired table takes precedence.",
    }

    @staticmethod
    def read_matrix(value):
        try:
            return finite_features(value, limit=np.finfo(np.float64).max, max_columns=100_000)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValidationError(f"Invalid matrix: {exc}", "matrix") from exc

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not (isinstance(p["matrix"], (list, tuple)) and not p["matrix"]):
            self.read_matrix(p["matrix"])


class MeasureMatrixDiagonal(_MatrixInput):
    type_id, title = "cv.matrix_diagonal", "Measure matrix diagonal"
    process_inputs_readonly = True
    outputs = (Port("diagonal", "data"), Port("trace", "number"))

    def process(self, inputs):
        matrix = self.read_matrix(inputs.get("matrix", self.parameters["matrix"]))
        trace = float(cv2.trace(matrix)[0])
        if not np.isfinite(trace):
            raise ValidationError("Matrix trace exceeds the float64 range", "matrix")
        return Result(
            {"diagonal": matrix.diagonal().tolist(), "trace": trace},
            message=f"Trace {trace:.6g}",
        )


class CompleteSymmetricMatrix(_MatrixInput):
    type_id, title = "cv.complete_symmetric_matrix", "Complete symmetric matrix"
    process_inputs_readonly = True
    outputs = _MATRIX_OUTPUT
    defaults = {**_MatrixInput.defaults, "source_triangle": "upper"}
    parameter_choices = {"source_triangle": ("upper", "lower")}
    parameter_help = {
        "matrix": "Square matrix. Enter finite placeholders in the triangle to be replaced.",
        "source_triangle": "Triangle to mirror across the diagonal. Diagonal values stay in place.",
    }

    @staticmethod
    def read_matrix(value):
        matrix = _MatrixInput.read_matrix(value)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValidationError("Symmetric completion requires a square matrix", "matrix")
        return matrix

    def process(self, inputs):
        matrix = self.read_matrix(inputs.get("matrix", self.parameters["matrix"])).copy()
        triangle = self.parameters["source_triangle"]
        cv2.completeSymm(matrix, lowerToUpper=triangle == "lower")
        return Result({"matrix": matrix.tolist()}, message=f"Mirrored the {triangle} triangle")
