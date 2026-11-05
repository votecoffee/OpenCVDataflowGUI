"""Fit an empirical covariance model and score rows by their joint deviation."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .projection import profile_plot
from .sample_tables import finite_features

_MODEL = "feature_covariance_v1"
_SAMPLE_LIMIT = 1e12
_COVARIANCE_LIMIT = 1e26
_SAMPLES_HELP = "One feature vector per row. The Samples input overrides this table."


class FitFeatureCovariance(Block):
    type_id, title = "cv.feature_covariance", "Fit feature covariance"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("samples", "data", required=False),)
    outputs = (
        Port("model", "data"),
        Port("mean", "data"),
        Port("covariance", "data"),
        Port("sample_count", "number"),
    )
    defaults = {"samples": [], "normalization": "sample"}
    parameter_choices = {"normalization": ("sample", "population")}
    parameter_help = {
        "samples": _SAMPLES_HELP,
        "normalization": "Sample divides by N − 1. Population divides by N.",
    }

    def process(self, inputs):
        p = self.parameters
        samples = finite_features(inputs.get("samples", p["samples"]))
        count = len(samples)
        if p["normalization"] == "sample" and count < 2:
            raise ValidationError("Sample covariance requires at least two rows")
        covariance, mean = cv2.calcCovarMatrix(
            samples, None, cv2.COVAR_ROWS | cv2.COVAR_NORMAL | cv2.COVAR_SCALE, ctype=cv2.CV_64F
        )
        if p["normalization"] == "sample":
            covariance *= count / (count - 1)
        mean, covariance = mean.ravel().tolist(), covariance.tolist()
        model = {
            "model": _MODEL,
            "mean": mean,
            "covariance": covariance,
            "sample_count": count,
            "normalization": p["normalization"],
        }
        return Result(
            {"model": model, "mean": mean, "covariance": covariance, "sample_count": count},
            message=f"Fitted {len(mean)} features from {count} rows",
        )


def inverse_covariance(model, dimensions, ridge):
    if not isinstance(model, dict) or model.get("model") != _MODEL:
        raise ValidationError("Connect a feature covariance model")
    try:
        mean = np.asarray(model.get("mean"), dtype=np.float64)
        matrix = np.asarray(model.get("covariance"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Model mean and covariance must be numeric") from exc
    if mean.shape != (dimensions,) or matrix.shape != (dimensions, dimensions):
        raise ValidationError("Sample feature count must match the covariance model")
    if (
        not np.isfinite(mean).all()
        or np.max(np.abs(mean)) > _SAMPLE_LIMIT
        or not np.isfinite(matrix).all()
        or np.max(np.abs(matrix)) > _COVARIANCE_LIMIT
        or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0)
    ):
        raise ValidationError("Model requires a finite mean and symmetric, bounded covariance")
    matrix = (matrix + matrix.T) / 2
    ok, eigenvalues, _ = cv2.eigen(matrix)
    if not ok or not np.isfinite(eigenvalues).all():
        raise ValidationError("Covariance eigendecomposition failed")
    largest, smallest = float(eigenvalues[0, 0]), float(eigenvalues[-1, 0])
    epsilon = np.finfo(np.float64).eps * dimensions
    if smallest < -max(0.0, largest) * epsilon:
        raise ValidationError("Model covariance must be positive semidefinite")
    if smallest + ridge <= (largest + ridge) * epsilon:
        raise ValidationError("Covariance is singular or ill-conditioned. Increase Ridge.")
    matrix.flat[:: dimensions + 1] += ridge
    condition, inverse = cv2.invert(matrix, flags=cv2.DECOMP_SVD)
    if condition <= epsilon or not np.isfinite(inverse).all():
        raise ValidationError("Covariance inversion is unstable. Increase Ridge.")
    return mean, inverse


class MahalanobisDistance(Block):
    type_id, title = "inspect.mahalanobis_distance", "Mahalanobis distance"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("model", "data"), Port("samples", "data", required=False))
    outputs = (
        Port("distances", "data"),
        Port("maximum_distance", "number"),
        Port("outlier_indices", "data"),
        Port("outlier_count", "number"),
    )
    defaults = {"samples": [], "ridge": 0.0, "check_distance": False, "distance_limit": 3.0}
    parameter_ranges = {
        "ridge": NumericRange(0, _COVARIANCE_LIMIT),
        "distance_limit": NumericRange(0),
    }
    parameter_help = {
        "samples": _SAMPLES_HELP,
        "ridge": "Add this value to each covariance diagonal entry before inversion.",
        "check_distance": "Fail when any row exceeds the distance limit.",
        "distance_limit": "Maximum Mahalanobis distance for an accepted row.",
    }

    def process(self, inputs):
        p = self.parameters
        samples = finite_features(inputs.get("samples", p["samples"]))
        dimensions = samples.shape[1]
        if samples.size * dimensions > 64_000_000:
            raise ValidationError("Scoring batch is too large for this feature count")
        mean, inverse = inverse_covariance(inputs["model"], dimensions, float(p["ridge"]))
        values = np.asarray([cv2.Mahalanobis(sample, mean, inverse) for sample in samples])
        if not np.isfinite(values).all():
            raise ValidationError("Mahalanobis distances overflowed")
        limit = float(p["distance_limit"])
        outliers = np.flatnonzero(values > limit).tolist()
        maximum = float(values.max())
        passed = not p["check_distance"] or not outliers
        graphics = {}
        if self.generate_graphics:
            graphics["distances"] = profile_plot(
                values[:, None],
                np.ones(len(values), dtype=bool),
                ("distance",),
                "sample row",
                "Mahalanobis distance",
            )
        return Result(
            {
                "distances": values.tolist(),
                "maximum_distance": maximum,
                "outlier_indices": outliers,
                "outlier_count": len(outliers),
            },
            graphics=graphics,
            passed=passed,
            message=f"Maximum distance {maximum:.4g}, {len(outliers)} rows above limit",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "mahalanobis_distance_out_of_tolerance",
                    "message": f"{len(outliers)} rows exceed distance limit {limit:.4g}",
                }
            ],
        )
