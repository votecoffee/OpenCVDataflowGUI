"""Compute real eigenpairs with a scale-independent residual check."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .projection import profile_plot
from .sample_tables import finite_features

_BASIS_RANK_TOLERANCE = 1e-10


def _square_matrix(value):
    try:
        matrix = finite_features(value, limit=np.finfo(np.float64).max)
    except ValidationError as exc:
        raise ValidationError(f"Invalid matrix: {exc}", "matrix") from exc
    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Eigendecomposition requires a square matrix", "matrix")
    return matrix


class MatrixEigen(Block):
    type_id, title = "cv.matrix_eigen", "Eigendecomposition"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("matrix", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("eigenvalues", "data"),
        Port("eigenvectors", "data"),
        Port("spectral_radius", "number"),
        Port("basis_rank", "number"),
        Port("relative_residuals", "data"),
        Port("max_relative_residual", "number"),
    )
    process_inputs_readonly = True
    defaults = {"matrix": [], "method": "auto", "max_relative_residual": 1e-9}
    parameter_choices = {"method": ("auto", "symmetric", "general_real")}
    parameter_ranges = {"max_relative_residual": NumericRange(0, 1)}
    parameter_help = {
        "matrix": "Square real matrix of order 1 to 128. A wired matrix takes precedence.",
        "method": (
            "Auto uses the symmetric solver for exactly symmetric inputs. "
            "General real requires a real spectrum."
        ),
        "max_relative_residual": (
            "Largest allowed normalized eigenpair error. Zero disables the limit."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        value = p["matrix"]
        if not (isinstance(value, (list, tuple)) and not value):
            _square_matrix(value)

    def process(self, inputs):
        p = self.parameters
        matrix = _square_matrix(inputs.get("matrix", p["matrix"]))
        symmetric = np.array_equal(matrix, matrix.T)
        method = p["method"]
        if method == "symmetric" and not symmetric:
            raise ValidationError(
                "Symmetric mode requires matching entries across the diagonal", "matrix"
            )
        use_symmetric = symmetric and method != "general_real"
        scale = float(np.max(np.abs(matrix))) or 1.0
        scaled = matrix / scale
        if use_symmetric:
            ok, values, vectors = cv2.eigen(scaled)
            if not ok:
                raise ValidationError("OpenCV could not solve the symmetric eigensystem", "matrix")
        else:
            # OpenCV drops the imaginary components. Screen the spectrum first.
            if np.any(np.linalg.eigvals(scaled).imag != 0):
                raise ValidationError("This matrix has complex eigenvalues", "matrix")
            values, vectors = cv2.eigenNonSymmetric(scaled)
        values = values.ravel()
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValidationError("OpenCV returned nonfinite eigenpairs", "matrix")

        row_scale = np.max(np.abs(vectors), axis=1)
        if np.any(row_scale == 0):
            raise ValidationError("OpenCV returned a zero eigenvector", "matrix")
        vectors = vectors / row_scale[:, None]
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        largest = np.argmax(np.abs(vectors), axis=1)
        vectors *= np.where(vectors[np.arange(len(values)), largest] < 0, -1, 1)[:, None]

        errors = scaled @ vectors.T - vectors.T * values
        relative = np.linalg.norm(errors, axis=0) / (
            (np.linalg.norm(scaled) or 1.0) + np.abs(values)
        )
        worst = float(relative.max())
        with np.errstate(over="ignore"):
            spectrum = values * scale
        if not np.isfinite(spectrum).all():
            raise ValidationError("Eigenvalues exceeded the float64 range", "matrix")

        limit = float(p["max_relative_residual"])
        passed = not limit or worst <= limit
        reasons = []
        rank = NO_VALUE
        if passed:
            rank = (
                len(values)
                if use_symmetric
                else int(np.linalg.matrix_rank(vectors, rtol=_BASIS_RANK_TOLERANCE))
            )
        else:
            reasons.append(
                {
                    "code": "eigenpair_residual_exceeds_limit",
                    "message": f"Maximum relative residual {worst:.4g} exceeds {limit:.4g}",
                }
            )
        graphics = {}
        if self.generate_graphics and passed:
            graphics["spectrum"] = profile_plot(
                spectrum[:, None],
                np.ones(len(spectrum), dtype=bool),
                ("value",),
                "eigenpair index",
                "eigenvalue",
            )
        return Result(
            {
                "eigenvalues": spectrum.tolist() if passed else NO_VALUE,
                "eigenvectors": vectors.tolist() if passed else NO_VALUE,
                "spectral_radius": float(np.max(np.abs(spectrum))) if passed else NO_VALUE,
                "basis_rank": rank,
                "relative_residuals": relative.tolist(),
                "max_relative_residual": worst,
            },
            passed=passed,
            fail_reasons=reasons,
            graphics=graphics,
            message=f"{len(values)} eigenpairs, maximum relative residual {worst:.4g}",
        )
