"""Known-good image anomaly decisions with a bounded native one-class RBF SVM."""

import hashlib
import json

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    descriptor_matrix,
    feature_contract,
    model_revision,
    normalize_features,
    prediction_drawing,
    validate_prediction_image,
)
from .core import Block, Port, Result
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup
from .svm_blocks import _invalid_constant, _json_object

MODEL = "opencv_one_class_svm_v1"
MAX_ROWS, MAX_DIMENSIONS, MAX_ELEMENTS = 512, 4096, 524288
MAX_BYTES = 16 * 1024**2
DEFAULTS = {
    "nu": 0.1,
    "gamma": 0.5,
    "normalization": "none",
    "max_iterations": 5000,
    "epsilon": 1e-6,
    "minimum_inlier_margin": 0.0,
}
RANGES = {
    "nu": NumericRange(1e-5, 0.99999),
    "gamma": NumericRange(1e-8, 1e6),
    "max_iterations": NumericRange(1, 100_000, whole=True),
    "epsilon": NumericRange(1e-12, 1),
    "minimum_inlier_margin": NumericRange(-512, 512),
}


def settings(parameters):
    if not isinstance(parameters, dict) or parameters.keys() != DEFAULTS.keys():
        raise ValidationError("Invalid one-class SVM settings")
    result = dict(parameters)
    if result["normalization"] not in ("none", "l2"):
        raise ValidationError("Choose none or l2 feature normalization", "normalization")
    for key, limits in RANGES.items():
        value = result[key]
        if type(value) not in (int, float):
            raise ValidationError(f"{key} must be a number", key)
        result[key] = limits.check(value, key, key)
    return result


def rows(block, value):
    if isinstance(value, FlowGroup) and len(value.items) > MAX_ROWS:
        raise ValidationError("One-class training accepts up to 512 descriptors")
    matrix, contract = descriptor_matrix(
        value,
        max_rows=MAX_ROWS,
        max_dimensions=MAX_DIMENSIONS,
        max_elements=MAX_ELEMENTS,
        max_abs=1e6,
    )
    if not isinstance(value, FlowGroup):
        contract = block.input_payloads.get("samples", {}).get("feature_contract")
    return matrix, feature_contract(contract)


def native_predictions(native, samples):
    _, native_label = native.predict(samples)
    _, native_raw = native.predict(samples, flags=cv2.ml.StatModel_RAW_OUTPUT)
    if (
        native_label is None
        or native_raw is None
        or native_label.shape != (len(samples), 1)
        or native_raw.shape != (len(samples), 1)
        or not np.isfinite(native_raw).all()
        or not np.isin(native_label, (0, 1)).all()
    ):
        raise ValidationError("Unexpected OpenCV one-class predictions")
    raw = native_raw[:, 0]
    inlier = native_label[:, 0] == 1
    if not np.array_equal(inlier, raw > 0):
        raise ValidationError("OpenCV one-class labels disagree with the signed native margin")
    return raw, inlier


def model_bytes(model):
    try:
        raw = (canonical_json(model) + "\n").encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("One-class model must be finite JSON data") from exc
    if len(raw) > MAX_BYTES:
        raise ValidationError("One-class model exceeds 16 MiB")
    return raw


def sha_text(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def read_one_class(model):
    """Rebuild OpenCV from checked numeric data without reading supplied native model text."""
    if (
        not isinstance(model, dict)
        or set(model)
        != {
            "model",
            "revision",
            "opencv_major",
            "dimensions",
            "settings",
            "feature_contract",
            "training_sha256",
            "support_vectors",
            "alpha",
            "rho",
        }
        or model["model"] != MODEL
    ):
        raise ValidationError("Connect a valid one-class SVM model")
    if type(model["opencv_major"]) is not int or model["opencv_major"] != int(
        cv2.__version__.split(".")[0]
    ):
        raise ValidationError("One-class model belongs to a different OpenCV major version")
    p = settings(model["settings"])
    feature_contract(model["feature_contract"])
    dimensions = model["dimensions"]
    if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValidationError("Invalid one-class feature dimensions")
    vectors = model["support_vectors"]
    if (
        not isinstance(vectors, list)
        or not 1 <= len(vectors) <= MAX_ROWS
        or len(vectors) * dimensions > MAX_ELEMENTS
        or any(not isinstance(row, list) or len(row) != dimensions for row in vectors)
        or any(type(v) not in (int, float) or not -1e6 <= v <= 1e6 for row in vectors for v in row)
    ):
        raise ValidationError("Invalid bounded one-class support vectors")
    # Keep signed native weights; the OpenCV solver can return them on small fits.
    alpha, rho = model["alpha"], model["rho"]
    if (
        not isinstance(alpha, list)
        or len(alpha) != len(vectors)
        or any(type(v) not in (int, float) or not -1.000001 <= v <= 1.000001 for v in alpha)
        or type(rho) not in (int, float)
        or not 0 < rho <= sum(abs(v) for v in alpha) + 1e-6
    ):
        raise ValidationError("Invalid one-class support weights or bias")
    if not sha_text(model["training_sha256"]):
        raise ValidationError("Invalid training descriptor identity")
    model_bytes(model)
    if model["revision"] != model_revision(model):
        raise ValidationError("One-class model revision mismatch")
    node = {
        "format": 3,
        "svmType": "ONE_CLASS",
        "kernel": {"type": "RBF", "gamma": p["gamma"]},
        "nu": p["nu"],
        "term_criteria": {"epsilon": p["epsilon"], "iterations": p["max_iterations"]},
        "var_count": dimensions,
        "class_count": 1,
        "sv_total": len(vectors),
        "support_vectors": vectors,
        "decision_functions": [{"sv_count": len(vectors), "rho": rho, "alpha": alpha}],
    }
    storage = cv2.FileStorage(
        canonical_json({"svm": node}), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
    )
    try:
        native = cv2.ml.SVM_create()
        native.read(storage.getNode("svm"))
        if (
            not native.isTrained()
            or native.getType() != cv2.ml.SVM_ONE_CLASS
            or native.getKernelType() != cv2.ml.SVM_RBF
            or native.getVarCount() != dimensions
            or native.getGamma() != p["gamma"]
            or native.getNu() != p["nu"]
        ):
            raise ValidationError("OpenCV could not restore the expected one-class model")
    finally:
        storage.release()
    return native


class TrainOneClassSVM(Block):
    type_id, title = "ml.one_class_svm_train", "Train one-class SVM on known-good images"
    category, block_type, subtype = "ML", "ML", "Training"
    inputs = (Port("samples", "any"),)
    outputs = (Port("model", "data"), Port("report", "data"))
    defaults = DEFAULTS
    parameter_choices = {"normalization": ("none", "l2")}
    parameter_ranges = RANGES
    parameter_help = {
        "nu": "Native one-class training parameter for the RBF boundary. "
        "Use independent samples to measure rejection rates.",
        "gamma": "RBF kernel scale. Recheck after changes to descriptor scaling.",
        "normalization": "Stored row normalization used identically in training and scoring.",
        "max_iterations": "Maximum native solver iterations.",
        "epsilon": "Native solver tolerance.",
        "minimum_inlier_margin": "Accept only margins strictly above this cutoff. "
        "0 matches the native decision. Stored in the model revision.",
    }
    input_help = {
        "samples": "Known-good descriptor rows or a FlowGroup of descriptor packets. "
        "At least three distinct rows after normalization; never mix feature settings.",
    }
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        settings(self.parameters if parameters is None else parameters)

    def process(self, inputs):
        p = settings(self.parameters)
        samples, contract = rows(self, inputs["samples"])
        samples = normalize_features(samples, p["normalization"])
        if len(samples) < 3 or len(np.unique(samples, axis=0)) < 3:
            raise ValidationError("Train on at least three distinct known-good descriptors")
        native = cv2.ml.SVM_create()
        native.setType(cv2.ml.SVM_ONE_CLASS)
        native.setKernel(cv2.ml.SVM_RBF)
        native.setNu(p["nu"])
        native.setGamma(p["gamma"])
        native.setTermCriteria(
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, p["max_iterations"], p["epsilon"])
        )
        if not native.train(samples, cv2.ml.ROW_SAMPLE, np.ones(len(samples), np.int32)):
            raise ValidationError("OpenCV could not train on these known-good samples")
        rho, alpha, indices = native.getDecisionFunction(0)
        support = native.getSupportVectors()
        if support is None or alpha is None or indices is None:
            raise ValidationError("OpenCV returned an empty one-class decision function")
        support = support[indices.reshape(-1)]
        model = {
            "model": MODEL,
            "opencv_major": int(cv2.__version__.split(".")[0]),
            "dimensions": samples.shape[1],
            "settings": p,
            "feature_contract": contract,
            "training_sha256": hashlib.sha256(
                samples.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "support_vectors": support.tolist(),
            "alpha": alpha.reshape(-1).tolist(),
            "rho": float(rho),
        }
        model["revision"] = model_revision(model)
        raw, native_inlier = native_predictions(native, samples)
        restored, restored_inlier = native_predictions(read_one_class(model), samples)
        if not np.array_equal(raw, restored) or not np.array_equal(native_inlier, restored_inlier):
            raise ValidationError("Saved one-class model does not reproduce native decisions")
        report = {
            "model_revision": model["revision"],
            "training_sha256": model["training_sha256"],
            "samples": len(samples),
            "dimensions": samples.shape[1],
            "support_vectors": len(support),
            "training_native_inliers": int(native_inlier.sum()),
            "training_accepted": int((raw > p["minimum_inlier_margin"]).sum()),
            "minimum_inlier_margin": p["minimum_inlier_margin"],
            "feature_contract_recorded": contract is not None,
            "opencv_version": cv2.__version__,
        }
        return Result(
            {"model": model, "report": report},
            message=f"Trained RBF boundary on {len(samples)} known-good descriptors",
        )


class PredictOneClassSVM(Block):
    type_id, title = "ml.one_class_svm_predict", "Inspect image using one-class SVM"
    category, block_type, subtype = "ML", "ML", "Anomaly inspection"
    inputs = (
        Port("model", "data"),
        Port("samples", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("anomaly_score", "number"),
        Port("accepted", "boolean"),
        Port("report", "data"),
        Port("image", required=False),
    )
    input_help = {
        "samples": "One descriptor using the same preprocessing contract as training.",
        "image": "Optional unchanged review image from the same frame and ticket.",
    }
    output_help = {
        "anomaly_score": "Negative native margin. Higher means farther toward rejection. "
        "This signed score is not a probability.",
        "accepted": "True when native margin strictly exceeds the saved cutoff.",
    }
    payload_reference = "samples"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True

    def process(self, inputs):
        model = inputs["model"]
        native = read_one_class(model)
        samples, contract = rows(self, inputs["samples"])
        if len(samples) != 1 or samples.shape[1] != model["dimensions"]:
            raise ValidationError("Use one descriptor matching the trained feature dimensions")
        if contract != model["feature_contract"]:
            raise ValidationError(
                "Prediction preprocessing differs from the trained feature contract"
            )
        image = validate_prediction_image(self, inputs, len(samples))
        samples = normalize_features(samples, model["settings"]["normalization"])
        raw, native_inlier = native_predictions(native, samples)
        margin = float(raw[0])
        cutoff = model["settings"]["minimum_inlier_margin"]
        passed = margin > cutoff
        report = {
            "model_revision": model["revision"],
            "raw_margin": margin,
            "anomaly_score": -margin,
            "native_inlier": bool(native_inlier[0]),
            "minimum_inlier_margin": cutoff,
            "feature_contract_verified": contract is not None,
            "accepted": passed,
        }
        message = f"{'Accepted' if passed else 'Anomaly'}: margin {margin:.5g}, cutoff {cutoff:g}"
        return Result(
            {"anomaly_score": -margin, "accepted": passed, "report": report, "image": image},
            passed=passed,
            drawings=prediction_drawing(self, image, message, passed, "One-class SVM decision"),
            fail_reasons=[]
            if passed
            else [
                {"code": "svm.anomaly", "message": "Native margin did not exceed the saved cutoff"}
            ],
            event_data=report,
            message=message,
        )


class SaveOneClassSVM(Block):
    type_id, title = "ml.one_class_svm_save", "Save one-class SVM model"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "text"),)
    defaults = {"path": "one-class-model.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_one_class(model)
        path = local_path(self, self.parameters["path"])
        with path.open("xb") as stream:
            stream.write(model_bytes(model))
        return Result({"path": str(path)}, message="Saved one-class SVM model")


class LoadOneClassSVM(Block):
    type_id, title = "ml.one_class_svm_load", "Load one-class SVM model"
    category, block_type, subtype = "ML", "ML", "Anomaly inspection"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("One-class model exceeds 16 MiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_object, parse_constant=_invalid_constant
        )
        read_one_class(model)
        return Result({"model": model}, message=f"Loaded {len(model['alpha'])} support vectors")


ONE_CLASS_SVM_BLOCKS = (TrainOneClassSVM, PredictOneClassSVM, SaveOneClassSVM, LoadOneClassSVM)
