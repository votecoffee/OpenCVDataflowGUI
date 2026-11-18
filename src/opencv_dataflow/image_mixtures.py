"""Native Gaussian mixtures for selected image colors and local surface texture."""

import copy
import json
import math
from dataclasses import replace

import cv2
import numpy as np

from .classifier_common import canonical_json, model_revision
from .cluster_models import color_values
from .clustering import cluster_colors
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import image_format
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .selection import SELECTION_INPUTS

MODEL = "opencv_image_em_v1"
MAX_MODEL_BYTES = 65536
SPACES = ("bgr", "lab", "gray", "gray_texture")
COVARIANCES = {
    "diagonal": cv2.ml.EM_COV_MAT_DIAGONAL,
    "spherical": cv2.ml.EM_COV_MAT_SPHERICAL,
    "generic": cv2.ml.EM_COV_MAT_GENERIC,
}
SELECTION = tuple(replace(p, required_when_connected=p.name != "image") for p in SELECTION_INPUTS)
LIMITS = {"max_work": 128_000_000, "max_bytes_mb": 256.0}
LIMIT_RANGES = {
    "max_work": NumericRange(1, 1_000_000_000, whole=True),
    "max_bytes_mb": NumericRange(0.01, 4096),
}
LIMIT_HELP = {
    "max_work": "Bound estimated fitting/prediction work before native EM calls.",
    "max_bytes_mb": "Estimated array/workspace allowance; not measured process RSS.",
}


def feature_contract(space, window):
    return {
        "model": "em_image_features_v1",
        "space": space,
        "texture_window": window if space == "gray_texture" else 1,
        "scaling": "unit_BGR_gray;Lab_L/100_ab/128;texture_mean_std",
        "opencv_version": cv2.__version__,
    }


def dimensions(space):
    return 2 if space == "gray_texture" else 1 if space == "gray" else 3


def features(inputs, contract):
    space, window = contract["space"], contract["texture_window"]
    values, requested = color_values(inputs, "gray" if space == "gray_texture" else space)
    source = inputs["image"]
    if space in ("gray", "gray_texture") and image_format(source) == "gray":
        unit = np.asarray(source, np.float32)
        if source.dtype.kind == "u":
            unit = unit / np.iinfo(source.dtype).max
        values = unit[requested, None]
    selected = requested.copy()
    if space == "gray_texture":
        gray = np.zeros(requested.shape, np.float64)
        gray[requested] = values[:, 0]
        kernel = np.ones((window, window), np.uint8)
        selected = (
            cv2.erode(
                requested.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            != 0
        )
        mean = cv2.boxFilter(gray, -1, (window, window), borderType=cv2.BORDER_CONSTANT)
        square = cv2.boxFilter(gray * gray, -1, (window, window), borderType=cv2.BORDER_CONSTANT)
        deviation = np.sqrt(np.maximum(0, square - mean * mean))
        values = np.column_stack((mean[selected], deviation[selected]))
    if not selected.any():
        raise ValidationError("Selection contains no supported visible feature pixels")
    return np.ascontiguousarray(values, np.float64), selected, requested


def matrix(value, shape, name, limit=100):
    try:
        result = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValidationError(name + " must be a numeric array") from exc
    if (
        result.shape != shape
        or result.dtype.kind not in "iuf"
        or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, object).flat)
        or not np.isfinite(result).all()
        or np.any(np.abs(result) > limit)
    ):
        raise ValidationError(name + " has invalid shape, type or finite bounds")
    return np.ascontiguousarray(result, np.float64)


def native_matrix(values):
    return {
        "type_id": "opencv-matrix",
        "rows": values.shape[0],
        "cols": values.shape[1],
        "dt": "d",
        "data": values.ravel().tolist(),
    }


def read_matrix(value, shape, name):
    if (
        not isinstance(value, dict)
        or set(value) != {"type_id", "rows", "cols", "dt", "data"}
        or value["type_id"] != "opencv-matrix"
        or value["dt"] != "d"
        or type(value["rows"]) is not int
        or type(value["cols"]) is not int
        or (value["rows"], value["cols"]) != shape
    ):
        raise ValidationError("Invalid native EM matrix: " + name)
    return matrix(value["data"], (shape[0] * shape[1],), name).reshape(shape)


def check_covariances(covariances, kind):
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
            raise ValidationError("EM covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] < 1e-10 or eigenvalues[-1] / eigenvalues[0] > 1e10:
            raise ValidationError("EM covariance must be positive definite and conditioned")
        diagonal = np.diag(covariance)
        if kind != "generic" and np.any(covariance != np.diag(diagonal)):
            raise ValidationError("EM covariance structure differs")
        if kind == "spherical" and not np.allclose(diagonal, diagonal[0], rtol=1e-10, atol=1e-12):
            raise ValidationError("EM spherical covariance differs")


def read_model(model):
    if (
        not isinstance(model, dict)
        or set(model) != {"model", "features", "native", "training", "revision"}
        or model.get("model") != MODEL
    ):
        raise ValidationError("Connect a portable image EM model")
    if len(canonical_json(model).encode("utf-8")) > MAX_MODEL_BYTES:
        raise ValidationError("Image EM model exceeds 64 KiB")
    if model["revision"] != model_revision(model):
        raise ValidationError("Image EM model revision differs")
    contract = model["features"]
    if not isinstance(contract, dict) or contract.get("space") not in SPACES:
        raise ValidationError("Invalid EM image feature contract")
    window = contract.get("texture_window")
    if type(window) is not int or window not in range(1, 64, 2):
        raise ValidationError("Invalid EM texture window")
    if contract["space"] == "gray_texture" and window < 3:
        raise ValidationError("Texture models need at least a 3x3 neighborhood")
    if contract != feature_contract(contract["space"], window):
        raise ValidationError("EM preprocessing/OpenCV version differs")
    native = model["native"]
    if (
        not isinstance(native, dict)
        or set(native) != {"format", "training_params", "means", "weights", "covs"}
        or type(native["format"]) is not int
        or native["format"] != 3
    ):
        raise ValidationError("Invalid native EM document")
    p = native["training_params"]
    if not isinstance(p, dict) or set(p) != {"nclusters", "cov_mat_type", "epsilon", "iterations"}:
        raise ValidationError("Invalid native EM training settings")
    k, d = p["nclusters"], dimensions(contract["space"])
    if type(k) is not int or not 1 <= k <= 16 or p["cov_mat_type"] not in COVARIANCES:
        raise ValidationError("Invalid EM component count/covariance mode")
    NumericRange(1, 1000, whole=True).check(p["iterations"], "Iterations", "max_iterations")
    NumericRange(1e-12, 1).check(p["epsilon"], "Epsilon", "epsilon")
    means = read_matrix(native["means"], (k, d), "Means")
    weights = read_matrix(native["weights"], (1, k), "Weights").ravel()
    if np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1, abs_tol=1e-9):
        raise ValidationError("EM component weights must be positive and sum to one")
    if not isinstance(native["covs"], list) or len(native["covs"]) != k:
        raise ValidationError("Invalid EM covariance count")
    covariances = np.array([read_matrix(c, (d, d), "Covariance") for c in native["covs"]])
    check_covariances(covariances, p["cov_mat_type"])
    training = model["training"]
    if not isinstance(training, dict) or set(training) != {
        "sample_count",
        "selected_pixels",
        "supported_pixels",
        "sampling",
        "initialization",
        "covariance_type",
        "covariance_floor",
        "fitted_min_eigenvalues",
    }:
        raise ValidationError("Invalid EM training provenance")
    for name in ("sample_count", "selected_pixels", "supported_pixels"):
        if type(training[name]) is not int or not k <= training[name] <= 1_000_000:
            raise ValidationError("Invalid EM training sample counts")
    if not (
        training["sample_count"]
        <= min(100_000, training["supported_pixels"])
        <= training["selected_pixels"]
    ):
        raise ValidationError("EM training sample counts differ")
    if training["sampling"] != "even_supported_row_major" or training["initialization"] not in (
        "auto",
        "manual",
    ):
        raise ValidationError("Invalid EM training method")
    fit_kind = training["covariance_type"]
    if fit_kind not in COVARIANCES or p["cov_mat_type"] != (
        "diagonal" if fit_kind == "spherical" else fit_kind
    ):
        raise ValidationError("EM fitted/prediction covariance representation differs")
    check_covariances(covariances, fit_kind)
    NumericRange(1e-8, 1).check(
        training["covariance_floor"], "Covariance floor", "covariance_floor"
    )
    matrix(training["fitted_min_eigenvalues"], (k,), "Fitted covariance eigenvalues")
    return weights, means, covariances


def load_native(model):
    weights, means, covariances = read_model(model)
    storage = cv2.FileStorage(
        canonical_json(model["native"]), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
    )
    try:
        native = cv2.ml.EM_create()
        native.read(storage.root())
        if not native.isTrained() or native.getVarCount() != means.shape[1]:
            raise ValidationError("Loaded native EM model dimensions differ")
    finally:
        storage.release()
    return native, weights, means, covariances


def gaussian_terms(values, weights, means, covariances):
    terms = []
    for weight, mean, covariance in zip(weights, means, covariances, strict=True):
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, (values - mean).T)
        terms.append(
            np.log(weight)
            - 0.5
            * (
                np.square(whitened).sum(0)
                + 2 * np.log(np.diag(chol)).sum()
                + len(mean) * np.log(2 * np.pi)
            )
        )
    return np.column_stack(terms)


def evaluate(native, values, weights, means, covariances):
    _, probabilities = native.predict(values)
    if (
        probabilities.shape != (len(values), len(means))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-9)
    ):
        raise ValidationError("Native EM returned invalid component probabilities")
    terms = gaussian_terms(values, weights, means, covariances)
    peak = terms.max(1)
    likelihood = peak + np.log(np.exp(terms - peak[:, None]).sum(1))
    if not np.isfinite(likelihood).all():
        raise ValidationError("EM returned nonfinite mixture log density")
    return probabilities, likelihood


def budget(p, work, byte_count):
    if work > p["max_work"]:
        raise ValidationError("Image EM exceeds Max work", "max_work")
    if byte_count > p["max_bytes_mb"] * 1024**2:
        raise ValidationError("Image EM exceeds estimated workspace allowance", "max_bytes_mb")


class _ImageEM(Block):
    category, block_type, subtype = "ML", "ML", "Image surface models"
    preview_safe = True
    parallel_safe = True
    process_inputs_readonly = True
    payload_reference = "image"


class TrainImageEM(_ImageEM):
    type_id, title = "ml.em_train_image", "Train image Gaussian mixture"
    inputs = SELECTION
    outputs = (Port("model", "data"), Port("image"), Port("report", "data"))
    defaults = {
        "feature_space": "lab",
        "texture_window": 7,
        "components": 2,
        "covariance_type": "diagonal",
        "max_iterations": 100,
        "epsilon": 1e-6,
        "max_samples": 20_000,
        "covariance_floor": 0.0001,
        "initialization": "auto",
        "initial_means": [],
        "initial_covariances": [],
        "initial_weights": [],
        "initial_variance": 0.01,
        **LIMITS,
    }
    parameter_choices = {
        "feature_space": SPACES,
        "covariance_type": tuple(COVARIANCES),
        "initialization": ("auto", "manual"),
    }
    parameter_ranges = {
        "texture_window": NumericRange(3, 63, whole=True, odd=True),
        "components": NumericRange(1, 16, whole=True),
        "max_iterations": NumericRange(1, 1000, whole=True),
        "epsilon": NumericRange(1e-12, 1),
        "max_samples": NumericRange(2, 100_000, whole=True),
        "covariance_floor": NumericRange(1e-8, 1),
        "initial_variance": NumericRange(1e-8, 1),
        **LIMIT_RANGES,
    }
    parameter_help = {
        "feature_space": "Normalized BGR/Lab/gray, or local grayscale mean and standard deviation.",
        "texture_window": "Odd square neighborhood; requires fully selected visible support.",
        "components": "Unsupervised components, sorted by fitted means; IDs are model-specific.",
        "covariance_type": "Native fit structure; spherical uses equivalent diagonal scoring.",
        "max_iterations": "Native count-plus-epsilon stopping limit, not a convergence guarantee.",
        "epsilon": "Native EM convergence tolerance.",
        "max_samples": "Evenly spaced supported row-major pixels; rare surfaces may be missed.",
        "covariance_floor": "Explicit post-fit eigenvalue floor in squared feature units.",
        "initialization": "Native k-means, or Gaussian responsibilities followed by native trainM.",
        "initial_means": "Manual: component x feature table in normalized feature units.",
        "initial_covariances": "Manual: optional component x feature x feature covariance array.",
        "initial_weights": "Manual: optional positive weights summing to one; empty means uniform.",
        "initial_variance": "Manual covariance default when Initial covariances is empty.",
        **LIMIT_HELP,
    }
    input_help = {
        "image": "uint8/uint16 or [0,1] float gray/BGR/RGB/alpha; original pixels are retained."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        k, d = p["components"], dimensions(p["feature_space"])
        if p["max_samples"] < k:
            raise ValidationError("Max samples must cover every component", "max_samples")
        for name in ("initial_means", "initial_covariances", "initial_weights"):
            if not isinstance(p[name], list):
                raise ValidationError("Use a numeric JSON list", name)
        if p["initialization"] == "manual":
            matrix(p["initial_means"], (k, d), "Initial means")
            if p["initial_covariances"]:
                covs = matrix(p["initial_covariances"], (k, d, d), "Initial covariances")
                check_covariances(covs, p["covariance_type"])
            if p["initial_weights"]:
                weights = matrix(p["initial_weights"], (k,), "Initial weights")
                if np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1, abs_tol=1e-9):
                    raise ValidationError(
                        "Initial weights must be positive and sum to one", "initial_weights"
                    )

    def process(self, inputs):
        p = self.parameters
        contract = feature_contract(p["feature_space"], p["texture_window"])
        d, k = dimensions(p["feature_space"]), p["components"]
        pixels = inputs["image"].shape[0] * inputs["image"].shape[1]
        n = min(pixels, p["max_samples"])
        budget(p, 0, pixels * (d + 12) * 8 + n * k * 32)
        values, selected, requested = features(inputs, contract)
        if len(values) < k:
            raise ValidationError("Not enough supported pixels for the EM component count")
        if len(values) > p["max_samples"]:
            values = values[np.linspace(0, len(values) - 1, p["max_samples"], dtype=np.int64)]
        budget(
            p,
            len(values) * k * d * d * (p["max_iterations"] + 32),
            pixels * (d + 12) * 8 + len(values) * k * 32,
        )
        native = cv2.ml.EM_create()
        native.setClustersNumber(k)
        native.setCovarianceMatrixType(COVARIANCES[p["covariance_type"]])
        native.setTermCriteria(
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, p["max_iterations"], p["epsilon"])
        )
        if p["initialization"] == "manual":
            means = matrix(p["initial_means"], (k, d), "Initial means")
            covs = (
                matrix(p["initial_covariances"], (k, d, d), "Initial covariances")
                if p["initial_covariances"]
                else np.array([np.eye(d) * p["initial_variance"]] * k)
            )
            weights = np.array(p["initial_weights"] or [1 / k] * k, np.float64)
            initial = gaussian_terms(values, weights, means, covs)
            initial = np.exp(initial - initial.max(1, keepdims=True))
            initial /= initial.sum(1, keepdims=True)
            trained, _, _, _ = native.trainM(values, initial)
        else:
            trained, _, _, _ = native.trainEM(values)
        if not trained or not native.isTrained():
            raise ValidationError("Native EM training did not produce a model")
        means, covs, weights = (
            native.getMeans(),
            np.array(native.getCovs()),
            native.getWeights().ravel(),
        )
        order = sorted(range(k), key=lambda i: tuple(means[i]))
        raw_minima, effective = [], []
        for covariance in covs[order]:
            eigenvalues, vectors = np.linalg.eigh(covariance)
            raw_minima.append(float(eigenvalues[0]))
            if p["covariance_type"] == "generic":
                effective.append(
                    (vectors * np.maximum(eigenvalues, p["covariance_floor"])) @ vectors.T
                )
            else:
                diagonal = np.maximum(np.diag(covariance), p["covariance_floor"])
                effective.append(np.diag(diagonal))
        model = {
            "model": MODEL,
            "features": contract,
            "native": {
                "format": 3,
                "training_params": {
                    "nclusters": k,
                    "cov_mat_type": "diagonal"
                    if p["covariance_type"] == "spherical"
                    else p["covariance_type"],
                    "epsilon": p["epsilon"],
                    "iterations": p["max_iterations"],
                },
                "means": native_matrix(means[order]),
                "weights": native_matrix(weights[order][None]),
                "covs": [native_matrix(c) for c in effective],
            },
            "training": {
                "sample_count": len(values),
                "selected_pixels": int(requested.sum()),
                "supported_pixels": int(selected.sum()),
                "sampling": "even_supported_row_major",
                "initialization": p["initialization"],
                "covariance_type": p["covariance_type"],
                "covariance_floor": p["covariance_floor"],
                "fitted_min_eigenvalues": raw_minima,
            },
        }
        model["revision"] = model_revision(model)
        fitted, weights, means, covs = load_native(model)
        probabilities, density = evaluate(fitted, values, weights, means, covs)
        report = {
            **model["training"],
            "revision": model["revision"],
            "counts": np.bincount(probabilities.argmax(1), minlength=k).tolist(),
            "effective_mean_log_density": float(density.mean()),
            "native_training_method": "trainM" if p["initialization"] == "manual" else "trainEM",
            "prediction_covariance_type": model["native"]["training_params"]["cov_mat_type"],
            "covariance_type": p["covariance_type"],
            "feature_space": p["feature_space"],
        }
        return Result(
            {"model": model, "image": inputs["image"], "report": report},
            message=f"Trained {k} image mixtures from {len(values)} supported samples",
        )


class ApplyImageEM(_ImageEM):
    type_id, title = "ml.em_classify_image", "Classify image Gaussian mixture"
    inputs = (Port("model", "data"), *SELECTION)
    outputs = (
        Port("image"),
        Port("labels"),
        Port("winning_posterior"),
        Port("log_likelihood"),
        Port("unknown_mask"),
        Port("accepted_mask"),
        Port("valid_mask"),
        Port("report", "data"),
        Port("posteriors", "any", required=False),
    )
    defaults = {
        "min_posterior": 0.0,
        "check_log_likelihood": True,
        "min_log_likelihood": -12.0,
        "accepted_components": [],
        "max_unknown_fraction": 0.0,
        "minimum_support_fraction": 0.0,
        "chunk_size": 4096,
        **LIMITS,
    }
    parameter_ranges = {
        "min_posterior": NumericRange(0, 1),
        "min_log_likelihood": NumericRange(-1e12, 1e12),
        "max_unknown_fraction": NumericRange(0, 1),
        "minimum_support_fraction": NumericRange(0, 1),
        "chunk_size": NumericRange(1, 65536, whole=True),
        **LIMIT_RANGES,
    }
    parameter_help = {
        "min_posterior": "Minimum winning component posterior; novelty can still have posterior 1.",
        "check_log_likelihood": "Use absolute mixture density as an additional unknown-pixel gate.",
        "min_log_likelihood": "Inclusive log density in saved feature units, not a probability.",
        "accepted_components": "Empty accepts all; otherwise use canonical model-specific IDs.",
        "max_unknown_fraction": "Inclusive unsupported-class fraction among supported pixels.",
        "minimum_support_fraction": "Minimum supported/requested fraction after texture erosion.",
        "chunk_size": "Posterior/density batch bound; result maps remain full size.",
        **LIMIT_HELP,
    }
    input_help = {"image": TrainImageEM.input_help["image"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [n for n in self.exposed_outputs if n != "posteriors"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["check_log_likelihood"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "check_log_likelihood")
        ids = p["accepted_components"]
        if (
            not isinstance(ids, list)
            or len(ids) > 16
            or any(type(i) is not int or not 0 <= i < 16 for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValidationError("Use distinct component IDs from 0 to 15", "accepted_components")

    def process(self, inputs):
        p, model = self.parameters, inputs["model"]
        weights, means, covs = read_model(model)
        k, d = means.shape
        if any(i >= k for i in p["accepted_components"]):
            raise ValidationError("Accepted component ID exceeds this model", "accepted_components")
        pixels = inputs["image"].shape[0] * inputs["image"].shape[1]
        keep_posteriors = "posteriors" in self.exposed_outputs
        budget(
            p,
            0,
            pixels * (d * 8 + 80 + (k * 12 if keep_posteriors else 0))
            + min(pixels, p["chunk_size"]) * k * 32,
        )
        values, selected, requested = features(inputs, model["features"])
        budget(
            p,
            len(values) * k * d * d,
            pixels * (d * 8 + 80 + (k * 12 if keep_posteriors else 0))
            + min(len(values), p["chunk_size"]) * k * 32,
        )
        native, _, _, _ = load_native(model)
        labels, confidence, density = (
            np.empty(len(values), np.int16),
            np.empty(len(values)),
            np.empty(len(values)),
        )
        posterior = np.empty((len(values), k), np.float32) if keep_posteriors else None
        for start in range(0, len(values), p["chunk_size"]):
            stop = min(len(values), start + p["chunk_size"])
            probabilities, likelihood = evaluate(native, values[start:stop], weights, means, covs)
            labels[start:stop] = probabilities.argmax(1)
            confidence[start:stop] = probabilities.max(1)
            density[start:stop] = likelihood
            if posterior is not None:
                posterior[start:stop] = probabilities
        by_posterior = confidence >= p["min_posterior"]
        by_density = (
            density >= p["min_log_likelihood"]
            if p["check_log_likelihood"]
            else np.ones(len(values), bool)
        )
        by_id = (
            np.isin(labels, p["accepted_components"])
            if p["accepted_components"]
            else np.ones(len(values), bool)
        )
        accepted = by_posterior & by_density & by_id
        shape = selected.shape

        def mapped(rows, dtype, outside=0):
            output = np.full(shape, outside, dtype)
            output[selected] = rows
            return output

        label_map = mapped(labels, np.int16, -1)
        confidence_map = mapped(confidence, np.float64, -1)
        density_map = mapped(density, np.float64)
        unknown = mapped(~accepted, np.uint8) * 255
        support_fraction = float(selected.sum() / requested.sum())
        unknown_fraction = float((~accepted).mean())
        passed = (
            unknown_fraction <= p["max_unknown_fraction"]
            and support_fraction >= p["minimum_support_fraction"]
        )
        reasons = []
        if unknown_fraction > p["max_unknown_fraction"]:
            reasons.append(
                {
                    "code": "em_unknown_pixels",
                    "message": f"{unknown_fraction:.2%} unknown feature pixels",
                }
            )
        if support_fraction < p["minimum_support_fraction"]:
            reasons.append(
                {"code": "em_feature_support", "message": f"{support_fraction:.2%} feature support"}
            )
        report = {
            "model_revision": model["revision"],
            "fitted_covariance_type": model["training"]["covariance_type"],
            "prediction_covariance_type": model["native"]["training_params"]["cov_mat_type"],
            "feature_contract": model["features"],
            "selected_pixels": int(requested.sum()),
            "supported_pixels": len(values),
            "support_fraction": support_fraction,
            "unknown_count": int((~accepted).sum()),
            "unknown_fraction": unknown_fraction,
            "component_counts": np.bincount(labels, minlength=k).tolist(),
            "below_posterior": int((~by_posterior).sum()),
            "below_density": int((~by_density).sum()),
            "rejected_component": int((~by_id).sum()),
            "minimum_log_density": float(density.min()),
            "mean_log_density": float(density.mean()),
            "score_meaning": "accepted_pixel_fraction",
        }
        drawings, graphics = {}, {}
        if self.generate_graphics:
            preview = np.zeros((*shape, 3), np.uint8)
            preview[selected] = cluster_colors(k)[labels]
            preview[unknown != 0] = (40, 40, 240)
            graphics["components"] = preview
            contours, _ = cv2.findContours(unknown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objects, vertices = [], 0
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:128]:
                if vertices + len(contour) > 8192:
                    break
                vertices += len(contour)
                objects.append(
                    graphic_object("polyline", contour.reshape(-1, 2), (40, 40, 240), closed=True)
                )
            objects.append(
                graphic_object(
                    "label",
                    [(8, 8)],
                    (80, 220, 80) if passed else (40, 40, 240),
                    text=f"EM unknown {unknown_fraction:.1%}; support {support_fraction:.1%}",
                )
            )
            drawings["Mixture inspection"] = GraphicDrawing(tuple(objects))
        groups = NO_VALUE
        if posterior is not None:
            context = self.input_payloads["image"]
            groups = FlowGroup(
                tuple(
                    ValuePacket(
                        mapped(posterior[:, i], np.float32),
                        {
                            **copy.deepcopy(context),
                            "em_component": i,
                            "em_revision": model["revision"],
                        },
                    )
                    for i in range(k)
                ),
                tuple(f"Component {i}" for i in range(k)),
            )
        return Result(
            {
                "image": inputs["image"],
                "labels": label_map,
                "winning_posterior": confidence_map,
                "log_likelihood": density_map,
                "unknown_mask": unknown,
                "accepted_mask": mapped(accepted, np.uint8) * 255,
                "valid_mask": selected.astype(np.uint8) * 255,
                "report": report,
                "posteriors": groups,
            },
            drawings=drawings,
            graphics=graphics,
            passed=passed,
            score=1 - unknown_fraction,
            fail_reasons=reasons,
            message=f"{len(values)} feature pixels; {unknown_fraction:.1%} unknown",
        )


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate image EM JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("Nonfinite image EM JSON number: " + value)


class SaveImageEM(_ImageEM):
    type_id, title = "ml.em_save", "Save image Gaussian mixture"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "image-mixture.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_model(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result({"path": str(path)}, message="Saved image EM model; existing files preserved")


class LoadImageEM(_ImageEM):
    type_id, title = "ml.em_load", "Load image Gaussian mixture"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("Image EM model exceeds 64 KiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )
        read_model(model)
        return Result({"model": model}, message="Loaded verified image EM model")


IMAGE_EM_BLOCKS = (TrainImageEM, ApplyImageEM, SaveImageEM, LoadImageEM)
