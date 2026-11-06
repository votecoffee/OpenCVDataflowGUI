"""Solve numeric systems and report how closely the solution fits each observation."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .projection import profile_plot
from .sample_tables import finite_features

_METHODS = {
    "svd": cv2.DECOMP_SVD,
    "qr": cv2.DECOMP_QR,
    "lu": cv2.DECOMP_LU,
    "cholesky": cv2.DECOMP_CHOLESKY,
    "eigen": cv2.DECOMP_EIG,
}


def _table(value, field):
    try:
        samples = np.asarray(value)
        if field == "observations" and samples.ndim == 1:
            samples = samples[:, None]
        return finite_features(samples)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid {field}: {exc}", field) from exc


class SolveLinearSystem(Block):
    type_id, title = "cv.solve_linear_system", "Solve linear system"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (
        Port("coefficients", "data", required=False, required_when_connected=True),
        Port("observations", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("solution", "data"),
        Port("fitted", "data"),
        Port("residuals", "data"),
        Port("rmse", "data"),
        Port("rank", "number"),
    )
    defaults = {"coefficients": [], "observations": [], "method": "svd"}
    parameter_choices = {"method": tuple(_METHODS)}
    parameter_help = {
        "coefficients": "Rows are equations and columns are unknowns. Wired data takes precedence.",
        "observations": "Enter a vector or a table with one right-hand side per column.",
        "method": (
            "SVD allows any shape and rank. QR needs at least as many equations as unknowns. "
            "LU needs a square matrix. Eigen needs symmetry, and Cholesky also needs "
            "positive definiteness."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("coefficients", "observations"):
            value = p[field]
            if isinstance(value, (list, tuple)) and not value:
                continue
            _table(value, field)

    def process(self, inputs):
        p = self.parameters
        a = _table(inputs.get("coefficients", p["coefficients"]), "coefficients")
        b = _table(inputs.get("observations", p["observations"]), "observations")
        rows, unknowns = a.shape
        if len(b) != rows:
            raise ValidationError("Use one observation row per coefficient row", "observations")
        method = p["method"]
        if method == "qr" and rows < unknowns:
            raise ValidationError("QR needs at least as many equations as unknowns", "method")
        if method in ("lu", "cholesky", "eigen") and rows != unknowns:
            raise ValidationError(f"{method.upper()} needs a square coefficient matrix", "method")
        if method in ("cholesky", "eigen") and not np.array_equal(a, a.T):
            raise ValidationError(f"{method.capitalize()} needs symmetric coefficients", "method")

        # Keep the decomposition away from absolute underflow for small coefficients.
        scale = float(np.max(np.abs(a))) or 1.0
        scaled = a / scale
        flags = 0 if method == "svd" else cv2.SVD_NO_UV
        singular, u, vt = cv2.SVDecomp(scaled, flags=flags)
        singular = singular.ravel()
        if not np.isfinite(singular).all():
            raise ValidationError("OpenCV returned nonfinite singular values")
        cutoff = singular[0] * np.finfo(np.float64).eps * max(a.shape)
        rank = int(np.count_nonzero(singular > cutoff))
        if method == "svd":
            projected = u[:, :rank].T @ b
            scaled_solution = vt[:rank].T @ (projected / singular[:rank, None])
        else:
            if rank < unknowns:
                raise ValidationError("Coefficients are numerically rank deficient. Choose SVD.")
            if method == "cholesky":
                try:
                    np.linalg.cholesky(scaled)
                except np.linalg.LinAlgError as exc:
                    raise ValidationError("Cholesky needs positive definite coefficients") from exc
            ok, scaled_solution = cv2.solve(scaled, b, flags=_METHODS[method])
            if not ok:
                raise ValidationError(f"OpenCV {method.upper()} could not solve the system")

        with np.errstate(over="ignore", invalid="ignore"):
            solution = scaled_solution / scale
            fitted = a @ solution
            residuals = b - fitted
        if not all(np.isfinite(value).all() for value in (solution, fitted, residuals)):
            raise ValidationError("Linear system produced nonfinite results. Rescale the inputs.")
        rmse = np.hypot.reduce(residuals, axis=0) / np.sqrt(rows)
        graphics = {}
        if self.generate_graphics:
            graphics["residuals"] = profile_plot(
                residuals[:, :1], np.ones(rows, dtype=bool), ("RHS 1",), "equation", "residual"
            )
        return Result(
            {
                "solution": solution.tolist(),
                "fitted": fitted.tolist(),
                "residuals": residuals.tolist(),
                "rmse": rmse.tolist(),
                "rank": rank,
            },
            graphics=graphics,
            message=f"Rank {rank}/{unknowns}, maximum RMSE {np.max(rmse):.4g}",
        )
