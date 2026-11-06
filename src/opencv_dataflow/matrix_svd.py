"""Decompose a numeric table and measure the cost of truncating its spectrum."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .projection import profile_plot
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_MODES = {"thin": 0, "full": cv2.SVD_FULL_UV, "values_only": cv2.SVD_NO_UV}
_MAX_COMPONENTS = 128


def _matrix(value):
    try:
        matrix = finite_features(value, limit=np.finfo(np.float64).max, max_columns=100_000)
    except ValidationError as exc:
        raise ValidationError(f"Invalid matrix: {exc}", "matrix") from exc
    if min(matrix.shape) > _MAX_COMPONENTS:
        raise ValidationError(
            f"The smaller matrix dimension is limited to {_MAX_COMPONENTS}", "matrix"
        )
    return matrix


class MatrixSVD(Block):
    type_id, title = "cv.matrix_svd", "Singular value decomposition"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("matrix", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("singular_values", "data"),
        Port("left_vectors", "data"),
        Port("right_vectors_t", "data"),
        Port("approximation", "data"),
        Port("rank", "number"),
        Port("retained_energy", "number"),
        Port("relative_truncation_error", "number"),
    )
    process_inputs_readonly = True
    defaults = {"matrix": [], "mode": "thin", "components": 0, "relative_rank_tolerance": 0.0}
    parameter_choices = {"mode": tuple(_MODES)}
    parameter_ranges = {
        "components": NumericRange(0, _MAX_COMPONENTS, whole=True),
        "relative_rank_tolerance": NumericRange(0, 1),
    }
    parameter_help = {
        "matrix": "Rectangular table of real numbers. A wired table takes precedence.",
        "mode": (
            "Thin returns compact factors. Full returns complete orthogonal bases. "
            "Values only skips both bases and the reconstruction."
        ),
        "components": "Leading terms in the approximation and energy report. Zero keeps all terms.",
        "relative_rank_tolerance": (
            "Count singular values above this fraction of the largest. "
            "Zero uses float64 epsilon times the larger matrix dimension."
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
        rows, columns = matrix.shape
        size = min(rows, columns)
        components = int(p["components"]) or size
        if components > size:
            raise ValidationError(f"This matrix has at most {size} components", "components")
        mode = p["mode"]
        if mode == "full" and rows**2 + columns**2 > MAX_SAMPLE_VALUES:
            raise ValidationError("Full bases exceed 2 million values. Choose Thin.", "mode")

        # Normalize before factorization to protect tiny and large input scales.
        scale = float(np.max(np.abs(matrix))) or 1.0
        values, u, vt = cv2.SVDecomp(matrix / scale, flags=_MODES[mode])
        values = values.ravel()
        tolerance = float(p["relative_rank_tolerance"]) or np.finfo(np.float64).eps * max(
            rows, columns
        )
        rank = int(np.count_nonzero(values > values[0] * tolerance))
        total = np.hypot.reduce(values)
        if total:
            retained = float((np.hypot.reduce(values[:components]) / total) ** 2)
            relative_error = float(np.hypot.reduce(values[components:], initial=0) / total)
        else:
            retained, relative_error = 1.0, 0.0

        left, right_t, approximation = NO_VALUE, NO_VALUE, NO_VALUE
        with np.errstate(over="ignore", invalid="ignore"):
            singular = values * scale
            if mode != "values_only":
                reconstructed = (
                    (u[:, :components] * values[:components]) @ vt[:components]
                ) * scale
                if not all(np.isfinite(value).all() for value in (u, vt, reconstructed)):
                    raise ValidationError("SVD reconstruction exceeded the float64 range", "matrix")
                left, right_t, approximation = u.tolist(), vt.tolist(), reconstructed.tolist()
        if not np.isfinite(singular).all():
            raise ValidationError("Singular values exceeded the float64 range", "matrix")

        graphics = {}
        if self.generate_graphics:
            graphics["spectrum"] = profile_plot(
                singular[:, None],
                np.ones(size, bool),
                ("value",),
                "component index",
                "singular value",
            )
        return Result(
            {
                "singular_values": singular.tolist(),
                "left_vectors": left,
                "right_vectors_t": right_t,
                "approximation": approximation,
                "rank": rank,
                "retained_energy": retained,
                "relative_truncation_error": relative_error,
            },
            graphics=graphics,
            message=f"Rank {rank}/{size}, {components} terms retain {retained:.2%} of energy",
        )
