"""Solve real-coefficient polynomials and report the quality of each root."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError

_MAX_DEGREE = 32


def _coefficients(value, order):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(
            "Use a vector of real polynomial coefficients", "coefficients"
        ) from exc
    if raw.ndim == 2 and 1 in raw.shape:
        raw = raw.ravel()
    if raw.ndim != 1 or not 1 <= raw.size <= _MAX_DEGREE + 1 or raw.dtype.kind not in "iuf":
        raise ValidationError("Use 1 to 33 real coefficients in a vector", "coefficients")
    with np.errstate(over="ignore"):
        coeffs = raw.astype(np.float64)
    if not np.isfinite(coeffs).all():
        raise ValidationError(
            "Polynomial coefficients must be finite float64 values", "coefficients"
        )
    if order == "constant_first":
        coeffs = coeffs[::-1]
    coeffs = np.trim_zeros(coeffs, "f")
    if not coeffs.size:
        raise ValidationError("The zero polynomial has infinitely many roots", "coefficients")
    scaled = coeffs / np.max(np.abs(coeffs))
    if np.any((scaled == 0) & (coeffs != 0)):
        raise ValidationError(
            "Coefficient magnitudes are too far apart to normalize", "coefficients"
        )
    return scaled


def _relative_residual(coeffs, root):
    # Reversing the polynomial at 1/root keeps every Horner step bounded.
    if abs(root) > 1:
        coeffs = coeffs[::-1]
        root = 1 / root
    return abs(np.polyval(coeffs, root)) / np.polyval(np.abs(coeffs), abs(root))


class SolvePolynomialRoots(Block):
    type_id, title = "cv.polynomial_roots", "Solve polynomial roots"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("coefficients", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("roots", "data"),
        Port("real_roots", "data"),
        Port("relative_residuals", "data"),
        Port("max_relative_residual", "number"),
    )
    process_inputs_readonly = True
    defaults = {
        "coefficients": [],
        "coefficient_order": "highest_first",
        "method": "all_roots",
        "max_iterations": 300,
        "imaginary_tolerance": 1e-8,
        "residual_limit": 1e-8,
    }
    parameter_choices = {
        "coefficient_order": ("highest_first", "constant_first"),
        "method": ("all_roots", "analytic_real"),
    }
    parameter_ranges = {
        "max_iterations": NumericRange(1, 10_000, whole=True),
        "imaginary_tolerance": NumericRange(0),
        "residual_limit": NumericRange(0, 1),
    }
    parameter_help = {
        "coefficients": "Editable vector, row, or column. A wired value takes precedence.",
        "method": "All roots returns complex pairs. Analytic real returns distinct real roots.",
        "max_iterations": "Iteration limit for the all-roots method.",
        "imaginary_tolerance": "Maximum imaginary magnitude for inclusion in Real roots.",
        "residual_limit": "Fail the result if a relative residual exceeds this limit.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        value = p["coefficients"]
        if not (isinstance(value, (list, tuple)) and not value):
            _coefficients(value, p["coefficient_order"])

    def process(self, inputs):
        p = self.parameters
        coeffs = _coefficients(
            inputs.get("coefficients", p["coefficients"]), p["coefficient_order"]
        )
        analytic = p["method"] == "analytic_real"
        if analytic and coeffs.size > 4:
            raise ValidationError(
                "Analytic real requires a polynomial of degree 3 or lower", "method"
            )
        reduced = np.trim_zeros(coeffs, "b")
        zeros = len(coeffs) - len(reduced)
        roots = []
        if reduced.size > 1:
            if analytic:
                count, real = cv2.solveCubic(np.pad(reduced, (4 - reduced.size, 0)))
                roots = [complex(value) for value in real.ravel()[:count]]
            else:
                _, pairs = cv2.solvePoly(
                    np.ascontiguousarray(reduced[::-1]), maxIters=int(p["max_iterations"])
                )
                roots = [complex(real, imag) for real, imag in pairs.reshape(-1, 2)]
        if zeros:
            roots.extend([0j] * (1 if analytic else zeros))
        if not np.isfinite(roots).all():
            raise ValidationError(
                "OpenCV returned nonfinite roots. Rescale the variable or coefficients."
            )
        roots.sort(key=lambda root: (root.real, root.imag))
        residuals = [
            0.0 if zeros and root == 0 else float(_relative_residual(reduced, root))
            for root in roots
        ]
        maximum = max(residuals, default=0.0)
        passed = maximum <= float(p["residual_limit"])
        return Result(
            {
                "roots": [[root.real, root.imag] for root in roots],
                "real_roots": [
                    root.real for root in roots if abs(root.imag) <= p["imaginary_tolerance"]
                ],
                "relative_residuals": residuals,
                "max_relative_residual": maximum,
            },
            passed=passed,
            message=f"{len(roots)} roots, maximum relative residual {maximum:.3g}",
        )
