"""Invert numeric matrices and report rank, conditioning, and determinant."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .payloads import NO_VALUE
from .sample_tables import finite_features

_METHODS = ("svd", "lu", "cholesky")


def _matrix(value):
    try:
        matrix = finite_features(value, max_columns=100_000)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid matrix: {exc}", "matrix") from exc
    if min(matrix.shape) > 128:
        raise ValidationError("The smaller matrix dimension is limited to 128", "matrix")
    return matrix


def _determinant(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return NO_VALUE
    value = float(cv2.determinant(matrix))
    if np.isfinite(value) and value != 0:
        return value
    sign, log_magnitude = np.linalg.slogdet(matrix)
    if sign == 0:
        return 0.0
    # LU's intermediate products can overflow even when the determinant fits.
    with np.errstate(over="ignore", under="ignore"):
        value = float(sign * np.exp(log_magnitude))
    return value if np.isfinite(value) and value != 0 else NO_VALUE


class InvertMatrix(Block):
    type_id, title = "cv.invert_matrix", "Invert matrix"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("matrix", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("inverse", "data"),
        Port("rank", "number"),
        Port("reciprocal_condition", "number"),
        Port("determinant", "number"),
    )
    process_inputs_readonly = True
    defaults = {"matrix": [], "method": "svd"}
    parameter_choices = {"method": _METHODS}
    parameter_help = {
        "matrix": "Rectangular table of real numbers. A wired table takes precedence.",
        "method": (
            "SVD handles rectangular and rank-deficient matrices. LU needs a full-rank square "
            "matrix. Cholesky also needs symmetry and positive definiteness."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        value = p["matrix"]
        if not (isinstance(value, (list, tuple)) and not value):
            _matrix(value)

    def process(self, inputs):
        p = self.parameters
        matrix = _matrix(inputs.get("matrix", p["matrix"]))
        method = p["method"]
        rows, columns = matrix.shape
        if method != "svd" and rows != columns:
            raise ValidationError("LU and Cholesky require a square matrix", "method", "matrix")

        # Normalize before factorization so tiny coefficients keep their relative scale.
        scale = float(np.max(np.abs(matrix))) or 1.0
        scaled = matrix / scale
        flags = 0 if method == "svd" else cv2.SVD_NO_UV
        singular, u, vt = cv2.SVDecomp(scaled, flags=flags)
        singular = singular.ravel()
        if not np.isfinite(singular).all():
            raise ValidationError("OpenCV returned nonfinite singular values", "matrix")
        cutoff = singular[0] * np.finfo(np.float64).eps * max(matrix.shape)
        rank = int(np.count_nonzero(singular > cutoff))
        full_rank = rank == min(matrix.shape)

        if method == "svd":
            inverse = vt[:rank].T @ (u[:, :rank].T / singular[:rank, None])
        else:
            if not full_rank:
                raise ValidationError("Matrix is numerically rank deficient. Choose SVD.", "method")
            mode = cv2.DECOMP_LU
            if method == "cholesky":
                if not np.array_equal(matrix, matrix.T):
                    raise ValidationError("Cholesky requires a symmetric matrix", "matrix")
                try:
                    np.linalg.cholesky(scaled)
                except np.linalg.LinAlgError as exc:
                    raise ValidationError(
                        "Cholesky requires a positive definite matrix", "matrix"
                    ) from exc
                mode = cv2.DECOMP_CHOLESKY
            success, inverse = cv2.invert(scaled, flags=mode)
            if not success:
                raise ValidationError(
                    f"OpenCV {method.upper()} could not invert the matrix", "matrix"
                )

        with np.errstate(over="ignore", invalid="ignore"):
            inverse = inverse / scale
        if not np.isfinite(inverse).all():
            raise ValidationError("Inverse exceeds the float64 range", "matrix")
        condition = float(singular[-1] / singular[0]) if full_rank else 0.0
        return Result(
            {
                "inverse": inverse.tolist(),
                "rank": rank,
                "reciprocal_condition": condition,
                "determinant": _determinant(matrix),
            },
            message=f"Rank {rank}/{min(matrix.shape)}, reciprocal condition {condition:.4g}",
        )
