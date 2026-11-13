"""Portable OpenCV C-SVC image-descriptor classifiers, without native handles on wires."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, metadata, require_same_frame
from .tickets import ticket_key

MAX_ELEMENTS = 2_000_000
MAX_ROWS = 2048
MAX_DIMENSIONS = 65_536
MAX_MODEL_BYTES = 8 * 1024**2
KERNELS = {
    "linear": cv2.ml.SVM_LINEAR,
    "rbf": cv2.ml.SVM_RBF,
    "polynomial": cv2.ml.SVM_POLY,
    "sigmoid": cv2.ml.SVM_SIGMOID,
    "chi_squared": cv2.ml.SVM_CHI2,
    "intersection": cv2.ml.SVM_INTER,
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate model JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(token):
    raise ValidationError(f"Invalid model JSON number: {token}")


def _strings(value, name, *, maximum=MAX_ROWS):
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(s, str) or not 1 <= len(s) <= 128 for s in value)
    ):
        raise ValidationError(
            f"{name} requires up to {maximum} nonempty strings (1..128 characters)"
        )
    return value


def _matrix(value):
    contracts = []
    if isinstance(value, FlowGroup):
        contracts = [metadata(item).get("feature_contract") for item in value.items]
        value = [item.value for item in value.items]
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValidationError("Features must be finite numeric values, not strings or booleans")
    if array.ndim == 1:
        array = array[None, :]
    if (
        array.ndim != 2
        or not 1 <= array.shape[0] <= MAX_ROWS
        or not 1 <= array.shape[1] <= MAX_DIMENSIONS
        or array.size > MAX_ELEMENTS
        or not np.isfinite(array).all()
        or np.any(np.abs(array) > np.finfo(np.float32).max)
    ):
        raise ValidationError("Features exceed finite float32 row/dimension/element limits")
    if contracts and any(contract != contracts[0] for contract in contracts):
        raise ValidationError("Training features have different preprocessing contracts")
    converted = np.ascontiguousarray(array, np.float32)
    if np.any((array != 0) & (converted == 0)):
        raise ValidationError("Feature values underflow float32; rescale before classification")
    contract = copy.deepcopy(contracts[0]) if contracts else None
    return converted, contract


def _normalize(values, mode):
    if mode == "none":
        return values
    if mode != "l2":
        raise ValidationError("Unknown SVM feature normalization")
    wide = values.astype(np.float64)
    peak = np.max(np.abs(wide), axis=1, keepdims=True)
    if np.any(peak == 0):
        raise ValidationError("Cannot L2 normalize a zero feature vector")
    scaled = wide / peak
    return (scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))).astype(np.float32)


def _revision(model):
    return hashlib.sha256(
        _json({k: v for k, v in model.items() if k != "revision"}).encode()
    ).hexdigest()


def read_svm(model):
    if (
        not isinstance(model, dict)
        or model.get("model") != "opencv_svm_classifier_v1"
        or model.get("revision") != _revision(model)
    ):
        raise ValidationError("Connect a valid SVM classifier bundle with matching revision")
    if len(_json(model).encode()) > MAX_MODEL_BYTES:
        raise ValidationError("SVM bundle exceeds 8 MiB")
    contract = model.get("feature_contract")
    if contract is not None and (not isinstance(contract, dict) or len(_json(contract)) > 16384):
        raise ValidationError("Feature contract must be a dictionary of at most 16 KiB")
    classes = _strings(model.get("class_labels"), "Class mapping", maximum=64)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValidationError("SVM needs at least two distinct class labels")
    dimensions = model.get("dimensions")
    if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValidationError("Invalid SVM feature dimensions")
    if model.get("normalization") not in ("none", "l2"):
        raise ValidationError("Invalid SVM feature normalization")
    if model.get("opencv_major") != int(cv2.__version__.split(".")[0]):
        raise ValidationError("SVM bundle belongs to a different OpenCV major version")
    text = model.get("svm_yaml")
    if not isinstance(text, str) or not 1 <= len(text.encode()) <= MAX_MODEL_BYTES:
        raise ValidationError("SVM native model is missing or exceeds 8 MiB")
    storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        node = storage.getNode("svm")
        if node.empty() or int(node.getNode("var_count").real()) != dimensions:
            raise ValidationError("SVM native feature dimensions differ from the bundle")
        count = int(node.getNode("sv_total").real())
        if (
            int(node.getNode("class_count").real()) != len(classes)
            or not 1 <= count <= MAX_ROWS
            or count * dimensions > MAX_ELEMENTS
        ):
            raise ValidationError("SVM native classes/support vectors exceed the bundle limits")
        labels = node.getNode("class_labels").getNode("data")
        if [labels.at(i).real() for i in range(labels.size())] != list(range(len(classes))):
            raise ValidationError("SVM native class IDs differ from the explicit class mapping")
        native = cv2.ml.SVM_create()
        native.read(node)
        if (
            not native.isTrained()
            or native.getType() != cv2.ml.SVM_C_SVC
            or native.getKernelType() not in KERNELS.values()
        ):
            raise ValidationError("Bundle must contain a trained C-SVC classifier")
    finally:
        storage.release()
    return native


class _SVMBlock(Block):
    category, block_type, subtype = "ML", "ML", "Classical classification"
    process_inputs_readonly = True
    parallel_safe = True


class TrainSVM(_SVMBlock):
    type_id, title = "ml.svm_train", "Train OpenCV SVM classifier"
    inputs = (
        Port("samples", "any"),
        Port("labels", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("model", "data"), Port("report", "data"))
    defaults = {
        "kernel": "rbf",
        "c": 2.0,
        "gamma": 0.1,
        "degree": 3.0,
        "coef0": 0.0,
        "normalization": "none",
        "max_iterations": 5000,
        "epsilon": 1e-6,
        "class_weights": [],
        "labels": [],
    }
    parameter_choices = {"kernel": tuple(KERNELS), "normalization": ("none", "l2")}
    parameter_ranges = {
        "c": NumericRange(1e-9, 1e9),
        "gamma": NumericRange(1e-9, 1e9),
        "degree": NumericRange(0.01, 32),
        "coef0": NumericRange(-1e6, 1e6),
        "max_iterations": NumericRange(1, 100_000, whole=True),
        "epsilon": NumericRange(1e-12, 1),
    }
    parameter_help = {
        "kernel": "OpenCV C-SVC kernel. Chi-squared requires nonnegative features.",
        "c": "Misclassification penalty; optionally multiplied by per-class weights.",
        "gamma": "RBF/polynomial/sigmoid/chi-squared kernel scale; ignored by linear/intersection.",
        "degree": "Polynomial kernel exponent.",
        "coef0": "Polynomial/sigmoid constant term.",
        "normalization": "Persisted per-row normalization; applied identically at prediction.",
        "class_weights": "Weights in first-appearance class order; [] gives equal weights.",
        "labels": "One string class per sample row; a wired Labels input takes precedence.",
        "max_iterations": "Maximum native training iterations.",
        "epsilon": "Native solver termination tolerance; used with iteration limit.",
    }
    input_help = {
        "samples": "Feature rows, or a batch of vectors. Known feature "
        "contracts must agree; HOG supplies its complete window/settings contract."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _strings(p["labels"], "Training labels")
        weights = p["class_weights"]
        if (
            not isinstance(weights, list)
            or len(weights) > 64
            or any(
                type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= 1e6
                for value in weights
            )
        ):
            raise ValidationError(
                "Class weights must be positive finite values <=1e6", "class_weights"
            )

    def process(self, inputs):
        p = self.parameters
        values, contract = _matrix(inputs["samples"])
        if not isinstance(inputs["samples"], FlowGroup):
            contract = copy.deepcopy(self.input_payloads["samples"].get("feature_contract"))
        labels = _strings(inputs.get("labels", p["labels"]), "Training labels")
        if len(labels) != len(values):
            raise ValidationError("Supply exactly one class label per training row", "labels")
        classes = list(dict.fromkeys(labels))
        if not 2 <= len(classes) <= 64:
            raise ValidationError("Training requires 2..64 classes", "labels")
        values = _normalize(values, p["normalization"])
        if p["kernel"] == "chi_squared" and np.any(values < 0):
            raise ValidationError("Chi-squared kernels require nonnegative features")
        native = cv2.ml.SVM_create()
        native.setType(cv2.ml.SVM_C_SVC)
        native.setKernel(KERNELS[p["kernel"]])
        native.setC(float(p["c"]))
        native.setGamma(float(p["gamma"]))
        native.setDegree(float(p["degree"]))
        native.setCoef0(float(p["coef0"]))
        native.setTermCriteria(
            (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                int(p["max_iterations"]),
                float(p["epsilon"]),
            )
        )
        weights = p["class_weights"]
        if not isinstance(weights, list):
            raise ValidationError("Class weights must be a list", "class_weights")
        if weights:
            if (
                len(weights) != len(classes)
                or any(type(v) not in (int, float) for v in weights)
                or not np.isfinite(weights).all()
                or min(weights) <= 0
                or max(weights) > 1e6
            ):
                raise ValidationError(
                    "Provide one positive finite weight <=1e6 per class", "class_weights"
                )
            native.setClassWeights(np.asarray(weights, np.float32))
        ids = np.int32([classes.index(label) for label in labels])
        if not native.train(values, cv2.ml.ROW_SAMPLE, ids):
            raise ValidationError("OpenCV could not train the SVM")
        storage = cv2.FileStorage(
            "", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML
        )
        native.write(storage, "svm")
        model = {
            "model": "opencv_svm_classifier_v1",
            "opencv_version": cv2.__version__,
            "opencv_major": int(cv2.__version__.split(".")[0]),
            "dimensions": values.shape[1],
            "class_labels": classes,
            "normalization": p["normalization"],
            "feature_contract": contract,
            "svm_yaml": storage.releaseAndGetString(),
        }
        model["revision"] = _revision(model)
        read_svm(model)
        predictions = native.predict(values)[1].reshape(-1).astype(int)
        report = {
            "classes": classes,
            "class_counts": [labels.count(c) for c in classes],
            "samples": len(values),
            "dimensions": values.shape[1],
            "training_accuracy": float(np.mean(predictions == ids)),
            "revision": model["revision"],
            "kernel": p["kernel"],
        }
        return Result(
            {"model": model, "report": report},
            message=f"Trained {len(classes)} classes on {len(values)} descriptors",
        )


class PredictSVM(_SVMBlock):
    type_id, title = "ml.svm_predict", "Predict OpenCV SVM classes"
    inputs = (
        Port("model", "data"),
        Port("samples", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("labels", "data"),
        Port("label", "data", required=False),
        Port("image", required=False),
        Port("report", "data"),
    )
    payload_reference = "samples"
    defaults = {"contract_policy": "require_match", "accepted_labels": []}
    parameter_choices = {"contract_policy": ("require_match", "dimensions_only")}
    parameter_help = {
        "contract_policy": "Require the trained preprocessing contract, or explicitly check "
        "only feature dimensions. Normalization always comes from the model.",
        "accepted_labels": "Optional allowed classes for a tool pass/fail result; [] accepts all. "
        "SVM class labels are closed-set predictions, not confidence or unknown-class detection.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _strings(p["accepted_labels"], "Accepted labels", maximum=64)

    def process(self, inputs):
        model = inputs["model"]
        native = read_svm(model)
        values, _ = _matrix(inputs["samples"])
        contract = self.input_payloads["samples"].get("feature_contract")
        if self.parameters["contract_policy"] == "require_match" and contract != model.get(
            "feature_contract"
        ):
            raise ValidationError(
                "Prediction preprocessing differs from the trained feature contract"
            )
        if values.shape[1] != model["dimensions"]:
            raise ValidationError("Prediction feature dimensions differ from the trained model")
        values = _normalize(values, model["normalization"])
        if native.getKernelType() == cv2.ml.SVM_CHI2 and np.any(values < 0):
            raise ValidationError("Chi-squared kernels require nonnegative features")
        predicted = native.predict(values)[1].reshape(-1)
        if (
            not np.isfinite(predicted).all()
            or np.any(predicted != np.rint(predicted))
            or np.any(predicted < 0)
            or np.any(predicted >= len(model["class_labels"]))
        ):
            raise ValidationError("OpenCV returned an invalid class ID")
        labels = [model["class_labels"][int(value)] for value in predicted]
        accepted = _strings(self.parameters["accepted_labels"], "Accepted labels", maximum=64)
        if not set(accepted) <= set(model["class_labels"]):
            raise ValidationError("Accepted labels must be in the trained class mapping")
        passed = not accepted or all(label in accepted for label in labels)
        image, drawings = inputs.get("image", NO_VALUE), {}
        if image is not NO_VALUE:
            require_same_frame(
                (image, ValuePacket(None, self.input_payloads["samples"])),
                message="Prediction image frame differs",
            )
            image_ticket = self.input_payloads["image"].get("ticket")
            feature_ticket = self.input_payloads["samples"].get("ticket")
            if (
                image_ticket is not None
                and feature_ticket is not None
                and ticket_key(image_ticket) != ticket_key(feature_ticket)
            ):
                raise ValidationError("Prediction image and features belong to different tickets")
            if len(labels) != 1:
                raise ValidationError("Connect an image only for a single descriptor prediction")
            # The image stays untouched; the label is a drawable diagnostic object.
            if self.generate_graphics:
                color = (100, 220, 90) if passed else (70, 80, 240)
                drawings = {
                    "SVM class": GraphicDrawing(
                        (
                            graphic_object(
                                "label",
                                [(8, 8)],
                                color,
                                text=labels[0],
                            ),
                        )
                    )
                }
        report = {
            "labels": labels,
            "count": len(labels),
            "revision": model["revision"],
            "class_mapping": model["class_labels"],
            "contract_policy": self.parameters["contract_policy"],
            "accepted": bool(passed),
            "probabilities_available": False,
        }
        return Result(
            {
                "labels": labels,
                "label": labels[0] if len(labels) == 1 else NO_VALUE,
                "image": image,
                "report": report,
            },
            drawings=drawings,
            passed=bool(passed),
            fail_reasons=[] if passed else ["svm_unaccepted_label"],
            message="Class " + labels[0] if len(labels) == 1 else f"{len(labels)} predictions",
        )


class SaveSVM(_SVMBlock):
    type_id, title = "ml.svm_save", "Save OpenCV SVM classifier"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs = (Port("model", "data"),)
    outputs = (Port("path", "data"),)
    defaults = {"path": "classifier.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_svm(model)
        content = _json(model) + "\n"
        if len(content.encode()) > MAX_MODEL_BYTES:
            raise ValidationError("Serialized SVM bundle exceeds 8 MiB")
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        return Result({"path": str(path)}, message="Saved SVM classifier; existing files preserved")


class LoadSVM(_SVMBlock):
    type_id, title = "ml.svm_load", "Load OpenCV SVM classifier"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        if path.stat().st_size > MAX_MODEL_BYTES:
            raise ValidationError("SVM bundle exceeds 8 MiB")
        model = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_invalid_constant,
            object_pairs_hook=_json_object,
        )
        read_svm(model)
        return Result({"model": model}, message=f"Loaded {len(model['class_labels'])} SVM classes")


SVM_BLOCKS = (TrainSVM, PredictSVM, SaveSVM, LoadSVM)
