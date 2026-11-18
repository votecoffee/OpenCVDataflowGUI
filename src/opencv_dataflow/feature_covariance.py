"""Portable image-descriptor covariance and correlation-aware outlier judgments."""

import copy
import json

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    descriptor_rows64,
    feature_contract,
    model_revision,
    normalize_rows64,
    validate_prediction_image,
)
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .projection import profile_plot
from .sample_tables import finite_features

MODEL = "feature_covariance_v2"
LEGACY = "feature_covariance_v1"
MAX_MODEL_BYTES = 8 * 1024**2
MAX_DIMENSIONS = 512
SAMPLE_LIMIT = 1e12
COVARIANCE_LIMIT = 1e26
MODEL_KEYS = {
    "model",
    "mean",
    "covariance",
    "sample_count",
    "normalization",
    "feature_normalization",
    "feature_contract",
    "centering",
    "revision",
}
LIMITS = {"max_work": 64_000_000, "max_bytes_mb": 128.0}
LIMIT_RANGES = {
    "max_work": NumericRange(1, 1_000_000_000, whole=True),
    "max_bytes_mb": NumericRange(0.01, 4096),
}
LIMIT_HELP = {
    "max_work": "Estimated covariance/scoring work bound before native calls.",
    "max_bytes_mb": "Estimated table/matrix allowance, not measured process RSS.",
}
SAMPLES_HELP = "Numeric descriptor rows or a batch; wired Samples overrides this table."


def read_covariance(model):
    """Pure shape/finite/identity checks; caller gates resources before native work."""
    if not isinstance(model, dict) or model.get("model") not in (MODEL, LEGACY):
        raise ValidationError("Connect a feature covariance model")
    legacy = model["model"] == LEGACY
    if not legacy and set(model) != MODEL_KEYS:
        raise ValidationError("Covariance model fields differ from its schema")
    try:
        mean = np.asarray(model.get("mean"), dtype=np.float64)
        matrix = np.asarray(model.get("covariance"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Model mean and covariance must be numeric") from exc
    if (
        mean.ndim != 1
        or not 1 <= len(mean) <= MAX_DIMENSIONS
        or matrix.shape != (len(mean), len(mean))
    ):
        raise ValidationError("Covariance model requires 1..512 matching feature dimensions")
    if (
        not np.isfinite(mean).all()
        or np.max(np.abs(mean)) > SAMPLE_LIMIT
        or not np.isfinite(matrix).all()
        or np.max(np.abs(matrix)) > COVARIANCE_LIMIT
        or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0)
    ):
        raise ValidationError("Model requires a finite mean and symmetric, bounded covariance")
    matrix = (matrix + matrix.T) / 2
    if not legacy:
        if len(canonical_json(model).encode("utf-8")) > MAX_MODEL_BYTES:
            raise ValidationError("Covariance model exceeds 8 MiB")
        if model["revision"] != model_revision(model):
            raise ValidationError("Covariance model revision differs")
        if model["normalization"] not in ("sample", "population") or model[
            "feature_normalization"
        ] not in ("none", "l2"):
            raise ValidationError("Invalid covariance or feature normalization")
        count = model["sample_count"]
        if (
            type(count) is not int
            or not (2 if model["normalization"] == "sample" else 1) <= count <= 100_000
        ):
            raise ValidationError("Invalid covariance training sample count")
        if model["centering"] not in ("estimated", "provided"):
            raise ValidationError("Invalid covariance centering policy")
        feature_contract(model["feature_contract"])
    return mean, matrix


def positive_covariance(matrix):
    ok, eigenvalues, _ = cv2.eigen(matrix)
    if not ok or not np.isfinite(eigenvalues).all():
        raise ValidationError("Covariance eigendecomposition failed")
    largest, smallest = float(eigenvalues[0, 0]), float(eigenvalues[-1, 0])
    epsilon = np.finfo(np.float64).eps * len(matrix)
    if smallest < -max(0.0, largest) * epsilon:
        raise ValidationError("Model covariance must be positive semidefinite")
    return largest, smallest, epsilon


def inverse_arrays(mean, matrix, ridge):
    largest, smallest, epsilon = positive_covariance(matrix)
    if smallest + ridge <= (largest + ridge) * epsilon:
        raise ValidationError("Covariance is singular or ill-conditioned. Increase Ridge.")
    effective = matrix.copy()
    effective.flat[:: len(mean) + 1] += ridge
    condition, inverse = cv2.invert(effective, flags=cv2.DECOMP_SVD)
    if condition <= epsilon or not np.isfinite(inverse).all():
        raise ValidationError("Covariance inversion is unstable. Increase Ridge.")
    return mean, inverse


def inverse_covariance(model, dimensions, ridge):
    """Retain the original helper interface for numeric integrations."""
    mean, matrix = read_covariance(model)
    if len(mean) != dimensions:
        raise ValidationError("Sample feature count must match the covariance model")
    return inverse_arrays(mean, matrix, ridge)


def budget(p, work, values):
    if work > p["max_work"]:
        raise ValidationError("Covariance operation exceeds Max work", "max_work")
    if values * 8 > p["max_bytes_mb"] * 1024**2:
        raise ValidationError("Covariance operation exceeds workspace allowance", "max_bytes_mb")


class _Covariance(Block):
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True
    payload_reference = "samples"


class FitFeatureCovariance(_Covariance):
    type_id, title = "cv.feature_covariance", "Fit feature covariance"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("samples", "any", required=False, required_when_connected=True),)
    outputs = (
        Port("model", "data"),
        Port("mean", "data"),
        Port("covariance", "data"),
        Port("sample_count", "number"),
    )
    defaults = {
        "samples": [],
        "normalization": "sample",
        "feature_normalization": "none",
        "mean": [],
        **LIMITS,
    }
    parameter_choices = {
        "normalization": ("sample", "population"),
        "feature_normalization": ("none", "l2"),
    }
    parameter_ranges = LIMIT_RANGES
    parameter_help = {
        "samples": SAMPLES_HELP,
        "normalization": "Sample divides by N-1; Population divides by N.",
        "feature_normalization": "Saved None or per-row L2, also applied during scoring.",
        "mean": "Empty estimates the mean; otherwise a fixed center in normalized feature units.",
        **LIMIT_HELP,
    }
    input_help = {"samples": "Up to 512 features; grouped descriptor contracts must agree."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["mean"], list):
            raise ValidationError("Mean must be empty or a finite numeric list", "mean")
        if p["mean"]:
            finite_features([p["mean"]], max_columns=MAX_DIMENSIONS)

    def process(self, inputs):
        p = self.parameters
        samples, contract, _ = descriptor_rows64(
            self, inputs.get("samples", p["samples"]), max_columns=MAX_DIMENSIONS
        )
        n, d = samples.shape
        if p["normalization"] == "sample" and n < 2:
            raise ValidationError("Sample covariance requires at least two rows")
        if np.max(np.abs(samples)) > SAMPLE_LIMIT:
            raise ValidationError("Covariance features must be within ±1e12")
        budget(p, n * d * d + d**3, 3 * n * d + 8 * d * d)
        samples = normalize_rows64(samples, p["feature_normalization"])
        provided = finite_features([p["mean"]], max_columns=MAX_DIMENSIONS) if p["mean"] else None
        if provided is not None and provided.shape[1] != d:
            raise ValidationError("Provided mean must match the descriptor dimension", "mean")
        flags = cv2.COVAR_ROWS | cv2.COVAR_NORMAL | cv2.COVAR_SCALE
        if provided is not None:
            flags |= cv2.COVAR_USE_AVG
        covariance, mean = cv2.calcCovarMatrix(samples, provided, flags, ctype=cv2.CV_64F)
        if p["normalization"] == "sample":
            covariance *= n / (n - 1)
        model = {
            "model": MODEL,
            "mean": mean.ravel().tolist(),
            "covariance": covariance.tolist(),
            "sample_count": n,
            "normalization": p["normalization"],
            "feature_normalization": p["feature_normalization"],
            "feature_contract": contract,
            "centering": "provided" if provided is not None else "estimated",
        }
        model["revision"] = model_revision(model)
        _, matrix = read_covariance(model)
        positive_covariance(matrix)
        graphics = {}
        if self.generate_graphics:
            graphics["feature_std"] = profile_plot(
                np.sqrt(np.maximum(0, np.diag(matrix)))[:, None],
                np.ones(d, bool),
                ("std",),
                "feature index",
                "feature units",
            )
        return Result(
            {
                "model": model,
                "mean": model["mean"],
                "covariance": model["covariance"],
                "sample_count": n,
            },
            graphics=graphics,
            message=f"Fitted {d} features from {n} rows; {p['normalization']} covariance",
        )


class MahalanobisDistance(_Covariance):
    type_id, title = "inspect.mahalanobis_distance", "Mahalanobis distance"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("model", "data"),
        Port("samples", "any", required=False, required_when_connected=True),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("distances", "data"),
        Port("maximum_distance", "number"),
        Port("outlier_indices", "data"),
        Port("outlier_count", "number"),
        Port("image", required=False),
        Port("report", "data"),
    )
    defaults = {
        "samples": [],
        "ridge": 0.0,
        "check_distance": False,
        "distance_limit": 3.0,
        "contract_policy": "require_match",
        **LIMITS,
    }
    parameter_choices = {"contract_policy": ("require_match", "dimensions_only")}
    parameter_ranges = {
        "ridge": NumericRange(0, COVARIANCE_LIMIT),
        "distance_limit": NumericRange(0),
        **LIMIT_RANGES,
    }
    parameter_help = {
        "samples": SAMPLES_HELP,
        "ridge": "Add to each covariance diagonal in squared feature units; changes score scale.",
        "check_distance": "Fail when any row exceeds Distance limit; otherwise report data only.",
        "distance_limit": "Inclusive Mahalanobis distance limit, not a probability or pixel error.",
        "contract_policy": "Require trained descriptor/PCA identity, or check dimensions only.",
        **LIMIT_HELP,
    }
    input_help = {"image": "Optional unchanged review image for one current descriptor packet."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["check_distance"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "check_distance")

    def process(self, inputs):
        p, model = self.parameters, inputs["model"]
        raw = inputs.get("samples", p["samples"])
        samples, contract, context = descriptor_rows64(self, raw, max_columns=MAX_DIMENSIONS)
        mean, matrix = read_covariance(model)
        if len(mean) != samples.shape[1]:
            raise ValidationError("Sample feature count must match the covariance model")
        trained_contract = model.get("feature_contract") if model["model"] == MODEL else None
        if p["contract_policy"] == "require_match" and contract != trained_contract:
            raise ValidationError("Scoring preprocessing differs from the trained feature contract")
        if np.max(np.abs(samples)) > SAMPLE_LIMIT:
            raise ValidationError("Covariance features must be within ±1e12")
        n, d = samples.shape
        budget(p, n * d * d + d**3, 4 * n * d + 8 * d * d)
        if isinstance(raw, FlowGroup) and inputs.get("image", NO_VALUE) is not NO_VALUE:
            raise ValidationError("Connect review image only for a single descriptor packet")
        image = validate_prediction_image(self, inputs, n)
        samples = normalize_rows64(
            samples,
            model.get("feature_normalization", "none") if model["model"] == MODEL else "none",
        )
        mean, inverse = inverse_arrays(mean, matrix, float(p["ridge"]))
        values = np.asarray([cv2.Mahalanobis(sample, mean, inverse) for sample in samples])
        if not np.isfinite(values).all():
            raise ValidationError("Mahalanobis distances overflowed")
        limit = float(p["distance_limit"])
        outliers = np.flatnonzero(values > limit).tolist()
        maximum = float(values.max())
        passed = not p["check_distance"] or not outliers
        text = f"Maximum distance {maximum:.4g}, {len(outliers)} rows above {limit:.4g}"
        graphics = {}
        if self.generate_graphics:
            graphics["distances"] = profile_plot(
                values[:, None],
                np.ones(n, bool),
                ("distance",),
                "sample row",
                "Mahalanobis distance",
            )
        revision = model.get("revision") if model["model"] == MODEL else None
        context["feature_contract"] = feature_contract(
            {
                "model": "mahalanobis_distances_v1",
                "covariance_revision": revision,
                "source": trained_contract,
                "ridge": p["ridge"],
                "verification": p["contract_policy"],
                "actual_source": contract if p["contract_policy"] == "dimensions_only" else None,
            }
        )
        payloads = {"distances": context}
        if image is not NO_VALUE:
            payloads["image"] = copy.deepcopy(self.input_payloads["image"])
        report = {
            "model_revision": revision,
            "legacy_model": model["model"] == LEGACY,
            "contract_verified": p["contract_policy"] == "require_match" and contract is not None,
            "feature_normalization": model.get("feature_normalization", "none"),
            "rows": n,
            "dimensions": d,
            "ridge": p["ridge"],
            "distance_limit": limit,
            "judgment_enabled": p["check_distance"],
            "score_is_probability": False,
        }
        drawings = {}
        if image is not NO_VALUE and self.generate_graphics:
            h, w = image.shape[:2]
            color = (100, 220, 90) if passed else (70, 80, 240)
            drawings["Feature window decision"] = GraphicDrawing(
                (
                    graphic_object(
                        "polyline",
                        [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
                        color,
                        closed=True,
                    ),
                )
            )
        return Result(
            {
                "distances": values.tolist(),
                "maximum_distance": maximum,
                "outlier_indices": outliers,
                "outlier_count": len(outliers),
                "image": image,
                "report": report,
            },
            graphics=graphics,
            drawings=drawings,
            payloads=payloads,
            passed=passed,
            message=text,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "mahalanobis_distance_out_of_tolerance",
                    "message": f"{len(outliers)} rows exceed distance limit {limit:.4g}",
                }
            ],
        )


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate covariance model JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("Nonfinite covariance model JSON number: " + value)


class SaveFeatureCovariance(_Covariance):
    type_id, title = "cv.save_feature_covariance", "Save feature covariance"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "feature-covariance.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        _, matrix = read_covariance(model)
        positive_covariance(matrix)
        raw = canonical_json(model) + "\n"
        if len(raw.encode("utf-8")) > MAX_MODEL_BYTES:
            raise ValidationError("Covariance model exceeds 8 MiB")
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
        return Result(
            {"path": str(path)}, message="Saved covariance model; existing files preserved"
        )


class LoadFeatureCovariance(_Covariance):
    type_id, title = "cv.load_feature_covariance", "Load feature covariance"
    category, block_type, subtype = "ML", "ML", "Image feature models"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("Covariance model exceeds 8 MiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )
        mean, matrix = read_covariance(model)
        positive_covariance(matrix)
        return Result(
            {"model": model}, message=f"Loaded covariance model with {len(mean)} features"
        )
