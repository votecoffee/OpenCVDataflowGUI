"""Learn and apply a row-oriented OpenCV principal component basis."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .projection import profile_plot
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_MODEL = "principal_components_v1"
_SAMPLES_HELP = "One numeric vector per row. The Samples input overrides this table."


class FitPrincipalComponents(Block):
    type_id, title = "cv.fit_pca", "Fit principal components"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("samples", "data", required=False),)
    outputs = (
        Port("model", "data"),
        Port("component_count", "number"),
        Port("explained_variance_ratios", "data"),
        Port("retained_variance", "number"),
    )
    defaults = {
        "samples": [],
        "selection": "variance",
        "component_count": 2,
        "minimum_variance": 0.95,
    }
    parameter_choices = {"selection": ("variance", "count")}
    parameter_ranges = {
        "component_count": NumericRange(1, 128, whole=True),
        "minimum_variance": NumericRange(0.000001, 1),
    }
    parameter_help = {
        "samples": _SAMPLES_HELP,
        "selection": "Choose a fixed count or the smallest basis meeting a variance target.",
        "component_count": "Number of components in Count mode, up to the fitted rank.",
        "minimum_variance": "Fraction of total variance to retain in Variance mode.",
    }

    def process(self, inputs):
        p = self.parameters
        samples = finite_features(inputs.get("samples", p["samples"]))
        if len(samples) < 2:
            raise ValidationError("PCA requires at least two rows")
        mean, vectors, eigenvalues = cv2.PCACompute2(samples, None)
        eigenvalues = eigenvalues.ravel()
        if not np.isfinite(eigenvalues).all() or not np.isfinite(vectors).all():
            raise ValidationError("OpenCV returned a nonfinite PCA basis")
        cutoff = eigenvalues[0] * np.finfo(np.float64).eps * samples.shape[1]
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        if not rank:
            raise ValidationError("PCA requires variation in at least one feature")
        eigenvalues = eigenvalues[:rank]
        ratios = eigenvalues / eigenvalues.sum()
        cumulative = ratios.cumsum()
        if p["selection"] == "count":
            count = int(p["component_count"])
            if count > rank:
                raise ValidationError(
                    f"Component count exceeds fitted rank of {rank}", "component_count"
                )
        else:
            target = float(p["minimum_variance"])
            count = min(rank, int(np.searchsorted(cumulative, target, side="left")) + 1)
        basis = vectors[:count].copy()
        # Eigenvector signs are arbitrary. Make the largest loading positive for stable scores.
        largest = np.argmax(np.abs(basis), axis=1)
        basis *= np.where(basis[np.arange(count), largest] < 0, -1, 1)[:, None]
        explained = ratios[:count].tolist()
        retained = float(min(1, cumulative[count - 1]))
        graphics = {}
        if self.generate_graphics:
            graphics["variance"] = profile_plot(
                ratios[:count, None],
                np.ones(count, dtype=bool),
                ("fraction",),
                "component index",
                "explained variance",
            )
        return Result(
            {
                "model": {"model": _MODEL, "mean": mean.ravel().tolist(), "basis": basis.tolist()},
                "component_count": count,
                "explained_variance_ratios": explained,
                "retained_variance": retained,
            },
            graphics=graphics,
            message=f"Retained {count} components explaining {retained:.2%} of variance",
        )


class TransformPrincipalComponents(Block):
    type_id, title = "cv.transform_pca", "Transform principal components"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("model", "data"), Port("samples", "data", required=False))
    outputs = (Port("samples", "data"), Port("feature_count", "number"))
    defaults = {"samples": [], "direction": "project"}
    parameter_choices = {"direction": ("project", "reconstruct")}
    parameter_help = {
        "samples": _SAMPLES_HELP,
        "direction": "Project original features or reconstruct from component scores.",
    }

    def process(self, inputs):
        p = self.parameters
        model = inputs["model"]
        if not isinstance(model, dict) or model.get("model") != _MODEL:
            raise ValidationError("Connect a fitted principal component model")
        mean = finite_features([model.get("mean")])
        basis = finite_features(model.get("basis"))
        if basis.shape[1] != mean.shape[1] or not np.allclose(
            basis @ basis.T, np.eye(len(basis)), rtol=0, atol=1e-7
        ):
            raise ValidationError("PCA model requires an orthonormal basis matching the mean")
        samples = finite_features(inputs.get("samples", p["samples"]), limit=1e100)
        projecting = p["direction"] == "project"
        expected = basis.shape[1] if projecting else len(basis)
        if samples.shape[1] != expected:
            raise ValidationError(f"PCA {p['direction']} expects {expected} features per row")
        width = len(basis) if projecting else basis.shape[1]
        if len(samples) * width > MAX_SAMPLE_VALUES:
            raise ValidationError("PCA output exceeds 2 million values. Use a smaller batch.")
        transform = cv2.PCAProject if projecting else cv2.PCABackProject
        result = transform(samples, mean, basis)
        return Result(
            {"samples": result.tolist(), "feature_count": result.shape[1]},
            message=f"Transformed {len(result)} rows to {result.shape[1]} features",
        )
