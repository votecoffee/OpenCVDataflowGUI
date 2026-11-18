"""Bounded native PCA with portable image-descriptor and projection identities."""

import copy
import json
import math

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    feature_contract,
    model_revision,
    prediction_drawing,
    validate_prediction_image,
)
from .classifier_common import (
    descriptor_rows64 as descriptor_rows,
)
from .classifier_common import (
    normalize_rows64 as normalize_rows,
)
from .core import Block, Port, Result
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .projection import profile_plot
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

MODEL = "principal_components_v2"
MAX_MODEL_BYTES = 8 * 1024**2
MODEL_KEYS = {
    "model",
    "mean",
    "basis",
    "normalization",
    "feature_contract",
    "sample_count",
    "eigenvalues",
    "total_variance",
    "retained_variance",
    "centering",
    "revision",
}
LIMIT_DEFAULTS = {"max_work": 64_000_000, "max_bytes_mb": 128.0}
LIMIT_RANGES = {
    "max_work": NumericRange(1, 1_000_000_000, whole=True),
    "max_bytes_mb": NumericRange(0.01, 4096),
}
LIMIT_HELP = {
    "max_work": "Bound estimated scalar multiply work before native PCA calls.",
    "max_bytes_mb": "Bound estimated tables/basis/native workspace; not measured process RSS.",
}


def revision(model):
    return model["revision"] if "revision" in model else model_revision(model)


def projected_contract(model):
    return feature_contract(
        {
            "model": "principal_components_projection_v1",
            "pca_revision": revision(model),
            "source": model.get("feature_contract"),
            "normalization": model.get("normalization", "none"),
            "dimensions": len(model["basis"]),
        }
    )


def read_pca(model):
    if not isinstance(model, dict) or model.get("model") not in (MODEL, "principal_components_v1"):
        raise ValidationError("Connect a fitted principal component model")
    legacy = model["model"] == "principal_components_v1"
    if set(model) != ({"model", "mean", "basis"} if legacy else MODEL_KEYS):
        raise ValidationError("PCA model fields differ from its schema")
    mean = finite_features([model.get("mean")], limit=1e100, max_columns=65536)
    basis = finite_features(model.get("basis"), max_columns=65536)
    if basis.shape[1] != mean.shape[1] or len(basis) > min(2048, basis.shape[1]):
        raise ValidationError("PCA model requires an orthonormal basis matching the mean")
    if len(basis) ** 2 * basis.shape[1] > 256_000_000:
        raise ValidationError("PCA model basis verification exceeds work limit")
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), rtol=0, atol=1e-7):
        raise ValidationError("PCA model requires an orthonormal basis matching the mean")
    if len(canonical_json(model).encode("utf-8")) > MAX_MODEL_BYTES:
        raise ValidationError("PCA model exceeds 8 MiB")
    if not legacy:
        if model["revision"] != model_revision(model):
            raise ValidationError("PCA model revision differs")
        if model["normalization"] not in ("none", "l2") or model["centering"] not in (
            "estimated",
            "provided",
        ):
            raise ValidationError("PCA normalization/centering differs")
        if type(model["sample_count"]) is not int or not 2 <= model["sample_count"] <= 100_000:
            raise ValidationError("PCA training count is invalid")
        feature_contract(model["feature_contract"])
        eigenvalues = finite_features([model["eigenvalues"]], limit=1e30, max_columns=2048).ravel()
        total, retained = model["total_variance"], model["retained_variance"]
        if (
            len(eigenvalues) != len(basis)
            or np.any(eigenvalues <= 0)
            or np.any(np.diff(eigenvalues) > 0)
            or type(total) not in (int, float)
            or not math.isfinite(total)
            or total <= 0
            or type(retained) not in (int, float)
            or not math.isfinite(retained)
            or not 0 < retained <= 1
            or not math.isclose(float(eigenvalues.sum()) / total, retained, rel_tol=1e-9)
        ):
            raise ValidationError("PCA variance metadata differs")
        projected_contract(model)
    return mean, basis


def budget(parameters, work, values):
    if work > parameters["max_work"]:
        raise ValidationError("PCA exceeds Max work before native execution", "max_work")
    if values * 8 > parameters["max_bytes_mb"] * 1024**2:
        raise ValidationError("PCA exceeds estimated workspace allowance", "max_bytes_mb")


class _PCA(Block):
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True
    payload_reference = "samples"


class FitPrincipalComponents(_PCA):
    type_id, title = "cv.fit_pca", "Fit principal components"
    inputs = (Port("samples", "any", required=False, required_when_connected=True),)
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
        "normalization": "none",
        "mean": [],
        **LIMIT_DEFAULTS,
    }
    parameter_choices = {"selection": ("variance", "count"), "normalization": ("none", "l2")}
    parameter_ranges = {
        "component_count": NumericRange(1, 2048, whole=True),
        "minimum_variance": NumericRange(0.000001, 1),
        **LIMIT_RANGES,
    }
    parameter_help = {
        "samples": "Numeric rows or a batch of image descriptors. Wired input wins.",
        "selection": "Fixed count or the smallest positive-rank basis meeting the variance target.",
        "component_count": "Count mode only, up to fitted positive rank and the model work limit.",
        "minimum_variance": "Fraction of training variance, not held-out accuracy or confidence.",
        "normalization": "Saved per-row L2 or None. L2 reconstruction loses original magnitude.",
        "mean": "Empty estimates mean; otherwise a finite vector in normalized feature units.",
        **LIMIT_HELP,
    }
    input_help = {"samples": "Up to 65,536 dimensions and 2M values; grouped contracts must agree."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["mean"], list):
            raise ValidationError("Mean must be an empty list or a numeric vector", "mean")
        if p["mean"]:
            finite_features([p["mean"]], max_columns=65536)

    def process(self, inputs):
        p = self.parameters
        samples, contract, _ = descriptor_rows(self, inputs.get("samples", p["samples"]))
        n, d = samples.shape
        if n < 2:
            raise ValidationError("PCA requires at least two rows")
        m = min(n, d)
        budget(p, n * d * m + m**3, 3 * n * d + 4 * m * m + 2 * m * d + 8 * d)
        if np.max(np.abs(samples)) > 1e12:
            raise ValidationError("PCA training features must be within ±1e12")
        samples = normalize_rows(samples, p["normalization"])
        provided = finite_features([p["mean"]], max_columns=65536) if p["mean"] else None
        if provided is not None and provided.shape[1] != d:
            raise ValidationError("Provided mean must match the feature dimension", "mean")
        mean, vectors, eigenvalues = cv2.PCACompute2(samples, provided)
        eigenvalues = eigenvalues.ravel()
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(eigenvalues).all()
            or not np.isfinite(vectors).all()
        ):
            raise ValidationError("OpenCV returned a nonfinite PCA basis")
        cutoff = eigenvalues[0] * np.finfo(np.float64).eps * d
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
            count = min(
                rank, int(np.searchsorted(cumulative, p["minimum_variance"], side="left")) + 1
            )
        basis = vectors[:count].copy()
        largest = np.argmax(np.abs(basis), axis=1)
        basis *= np.where(basis[np.arange(count), largest] < 0, -1, 1)[:, None]
        explained = ratios[:count].tolist()
        retained = float(min(1, cumulative[count - 1]))
        model = {
            "model": MODEL,
            "mean": mean.ravel().tolist(),
            "basis": basis.tolist(),
            "normalization": p["normalization"],
            "feature_contract": contract,
            "sample_count": n,
            "eigenvalues": eigenvalues[:count].tolist(),
            "total_variance": float(eigenvalues.sum()),
            "retained_variance": retained,
            "centering": "provided" if provided is not None else "estimated",
        }
        model["revision"] = model_revision(model)
        read_pca(model)
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
                "model": model,
                "component_count": count,
                "explained_variance_ratios": explained,
                "retained_variance": retained,
            },
            graphics=graphics,
            message=f"Fitted {count}/{d} components; training variance {retained:.2%}",
        )


class TransformPrincipalComponents(_PCA):
    type_id, title = "cv.transform_pca", "Transform principal components"
    inputs = (
        Port("model", "data"),
        Port("samples", "any", required=False, required_when_connected=True),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("samples", "data"),
        Port("feature_count", "number"),
        Port("reconstruction_errors", "data", required=False),
        Port("maximum_error", "number", required=False),
        Port("report", "data"),
        Port("image", required=False),
    )
    defaults = {
        "samples": [],
        "direction": "project",
        "contract_policy": "require_match",
        "check_reconstruction_error": False,
        "error_limit": 0.1,
        **LIMIT_DEFAULTS,
    }
    parameter_choices = {
        "direction": ("project", "reconstruct"),
        "contract_policy": ("require_match", "dimensions_only"),
    }
    parameter_ranges = {"error_limit": NumericRange(0, 1e100), **LIMIT_RANGES}
    parameter_help = {
        "samples": "A descriptor, table or descriptor batch. Wired input overrides this table.",
        "direction": "Project original features or reconstruct scores. L2 loses magnitude.",
        "contract_policy": "Require descriptor/model identity, or check dimensions only.",
        "check_reconstruction_error": "Project only: fail if any feature RMSE exceeds the limit.",
        "error_limit": "Inclusive per-row feature RMSE, not calibrated anomaly confidence.",
        **LIMIT_HELP,
    }
    input_help = {
        "image": "Optional unchanged image for one descriptor; ticket/grid/recipe must agree.",
        "samples": "Image contracts and PCA revisions are retained on projected descriptor wires.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["check_reconstruction_error"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "check_reconstruction_error")
        if p["direction"] == "reconstruct" and p["check_reconstruction_error"]:
            raise ValidationError(
                "Error judgment is available in Project mode only", "check_reconstruction_error"
            )

    def process(self, inputs):
        p, model = self.parameters, inputs["model"]
        mean, basis = read_pca(model)
        raw = inputs.get("samples", p["samples"])
        samples, contract, context = descriptor_rows(self, raw)
        projecting = p["direction"] == "project"
        source_contract = model.get("feature_contract")
        expected_contract = source_contract if projecting else projected_contract(model)
        untagged_numeric = not projecting and source_contract is None and contract is None
        if (
            p["contract_policy"] == "require_match"
            and not untagged_numeric
            and contract != expected_contract
        ):
            raise ValidationError(
                "PCA preprocessing/projection model revision differs from trained contract"
            )
        expected = basis.shape[1] if projecting else len(basis)
        if samples.shape[1] != expected:
            raise ValidationError(f"PCA {p['direction']} expects {expected} features per row")
        width = len(basis) if projecting else basis.shape[1]
        if len(samples) * width > MAX_SAMPLE_VALUES:
            raise ValidationError("PCA output exceeds 2 million values. Use a smaller batch.")
        n, d, k = len(samples), basis.shape[1], len(basis)
        budget(p, n * d * k * (2 if projecting else 1), 4 * n * d + 3 * n * k + k * d)
        if isinstance(raw, FlowGroup) and inputs.get("image", NO_VALUE) is not NO_VALUE:
            raise ValidationError("Connect a review image only for a single descriptor packet")
        image = validate_prediction_image(self, inputs, n)
        errors, maximum = NO_VALUE, NO_VALUE
        if projecting:
            samples = normalize_rows(samples, model.get("normalization", "none"))
            values = cv2.PCAProject(samples, mean, basis)
            restored = cv2.PCABackProject(values, mean, basis)
            errors = np.sqrt(np.mean(np.square(samples - restored), axis=1)).tolist()
            maximum = max(errors)
        else:
            values = cv2.PCABackProject(samples, mean, basis)
        if not np.isfinite(values).all() or (projecting and not math.isfinite(maximum)):
            raise ValidationError("Native PCA returned a nonfinite projection/reconstruction")
        output_contract = projected_contract(model)
        if not projecting:
            output_contract["model"] = "principal_components_reconstruction_v1"
            output_contract["dimensions"] = basis.shape[1]
        if p["contract_policy"] == "dimensions_only":
            output_contract["verification"] = "dimensions_only"
            output_contract["actual_source"] = contract
        feature_contract(output_contract)
        context["feature_contract"] = output_contract
        context["pca_revision"] = revision(model)
        passed = not p["check_reconstruction_error"] or maximum <= p["error_limit"]
        text = f"PCA {p['direction']}: {n} rows, {expected} to {width} features"
        if projecting:
            text += f"; maximum RMSE {maximum:.5g}"
        report = {
            "model_revision": context["pca_revision"],
            "direction": p["direction"],
            "rows": n,
            "input_dimensions": expected,
            "output_dimensions": width,
            "normalization": model.get("normalization", "none"),
            "contract_verified": p["contract_policy"] == "require_match" and contract is not None,
            "legacy_model": model["model"] != MODEL,
            "maximum_reconstruction_rmse": maximum if projecting else None,
            "score_is_probability": False,
        }
        graphics = {}
        if projecting and self.generate_graphics:
            graphics["reconstruction_error"] = profile_plot(
                np.asarray(errors)[:, None],
                np.ones(n, bool),
                ("RMSE",),
                "descriptor row",
                "feature units",
            )
        payloads = {"samples": context}
        if image is not NO_VALUE:
            payloads["image"] = copy.deepcopy(self.input_payloads["image"])
        return Result(
            {
                "samples": values.tolist(),
                "feature_count": width,
                "reconstruction_errors": errors,
                "maximum_error": maximum,
                "report": report,
                "image": image,
            },
            payloads=payloads,
            graphics=graphics,
            drawings=prediction_drawing(self, image, text, passed, "PCA reconstruction"),
            passed=passed,
            message=text,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "pca_reconstruction_error",
                    "message": f"Descriptor RMSE {maximum:.5g} exceeds {p['error_limit']:.5g}",
                }
            ],
        )


class SavePrincipalComponents(_PCA):
    type_id, title = "cv.save_pca", "Save principal component model"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "principal-components.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_pca(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result({"path": str(path)}, message="Saved PCA model; existing files preserved")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate PCA model JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("Nonfinite PCA model JSON number: " + value)


class LoadPrincipalComponents(_PCA):
    type_id, title = "cv.load_pca", "Load principal component model"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("PCA model exceeds 8 MiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )
        read_pca(model)
        return Result(
            {"model": model}, message=f"Loaded PCA basis with {len(model['basis'])} components"
        )
