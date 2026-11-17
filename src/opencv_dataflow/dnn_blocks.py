"""OpenCV DNN inference with explicit preprocessing and portable tensor outputs."""

import copy
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import format_image
from .ml_common import array, file_key, selected_tensor, tensor_record
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    canonical,
    configuration,
    environment,
    image_preprocessing,
    model_bytes,
    relative_path,
    resolve_contract,
    revision,
    tensor_description,
)
from .parameter_validation import ValidationError
from .payloads import NO_VALUE
from .processing import finite, image_value, integer, letterbox_bounds
from .selection import SELECTION_INPUTS, selection_mask

ENGINES = {
    "auto": cv2.dnn.ENGINE_AUTO,
    "classic": cv2.dnn.ENGINE_CLASSIC,
    "new": cv2.dnn.ENGINE_NEW,
    "ort": cv2.dnn.ENGINE_ORT,
}
BACKENDS = {
    "opencv": cv2.dnn.DNN_BACKEND_OPENCV,
    "cuda": cv2.dnn.DNN_BACKEND_CUDA,
    "default": cv2.dnn.DNN_BACKEND_DEFAULT,
}
TARGETS = {
    "cpu": cv2.dnn.DNN_TARGET_CPU,
    "opencl": cv2.dnn.DNN_TARGET_OPENCL,
    "opencl_fp16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
    "cuda": cv2.dnn.DNN_TARGET_CUDA,
    "cuda_fp16": cv2.dnn.DNN_TARGET_CUDA_FP16,
}


def preprocess(inputs, p):
    source = inputs["image"]
    canonical = image_value(source, canonical=True).astype(np.float32)
    image = np.asarray(format_image(canonical, p["color_format"], "unchanged"))
    mask = selection_mask(inputs, source)
    if mask is not None:
        image = image.copy()
        image[mask == 0] = 0
    w, h = (integer(p[k], k, 1, 4096) for k in ("width", "height"))
    sh, sw = image.shape[:2]
    mode = p["resize_mode"]
    sx, sy, tx, ty = w / sw, h / sh, 0.0, 0.0
    if mode == "letterbox":
        tx, ty, rw, rh = letterbox_bounds(sw, sh, w, h)
        sx, sy = rw / sw, rh / sh
        resized = cv2.resize(image, (rw, rh))
        shape = (h, w) if image.ndim == 2 else (h, w, image.shape[2])
        image = np.full(shape, finite(p["padding_value"], "Padding value"), np.float32)
        image[int(ty) : int(ty) + rh, int(tx) : int(tx) + rw] = resized
    elif mode == "stretch":
        image = cv2.resize(image, (w, h))
    else:
        raise ValueError("Resize mode must be stretch or letterbox")
    channels = 1 if image.ndim == 2 else image.shape[2]
    mean, std = (np.asarray(p[k], np.float32) for k in ("mean", "std"))
    if (
        mean.shape != (channels,)
        or std.shape != (channels,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Mean/std must have one finite value per channel; std must be positive")
    hwc = image[..., None] if image.ndim == 2 else image
    hwc = (hwc - mean) * finite(p["scale"], "Scale") / std
    blob = np.ascontiguousarray(hwc.transpose(2, 0, 1)[None], np.float32)
    if not np.isfinite(blob).all():
        raise ValueError("Preprocessing overflowed")
    matrix = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], np.float64)
    geometry = {
        "input_size": [sw, sh],
        "network_size": [w, h],
        "input_to_network": matrix.tolist(),
        "network_to_input": np.linalg.inv(matrix).tolist(),
        "resize_mode": mode,
        "color_format": p["color_format"],
        "preprocessing_contract": image_preprocessing(p),
    }
    return blob, geometry


class DNNInference(Block):
    type_id, title = "ml.dnn", "DNN ONNX inference"
    category, block_type, subtype = "ML", "ML", "DNN inference"
    project_path_parameters = ("model_path",)
    project_asset_parameters = (*Block.project_asset_parameters, "contract_path")
    inputs = SELECTION_INPUTS
    outputs = (
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "width": 224,
        "height": 224,
        "color_format": "rgb",
        "resize_mode": "stretch",
        "padding_value": 0.0,
        "scale": 1 / 255,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "input_name": "",
        "output_names": [],
        "engine": "auto",
        "backend": "opencv",
        "target": "cpu",
        "fusion": True,
        "max_output_elements": 1_000_000,
    }
    parameter_choices = {
        **CONTRACT_CHOICES,
        "color_format": ("rgb", "bgr", "gray"),
        "resize_mode": ("stretch", "letterbox"),
        "engine": tuple(ENGINES),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_help = {
        **CONTRACT_HELP,
        "model_path": "Local ONNX path; Contract path instead uses its bundled model.",
        "mean": "Per-channel values in source intensity units, before scaling.",
        "std": "Per-channel divisors after mean subtraction and scaling.",
        "output_names": "JSON list; empty selects all unconnected output layers.",
        "engine": "OpenCV 5 engine. Non-CPU targets require classic.",
        "max_output_elements": "Total tensor values retained per activation (bounded).",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [name for name in self.exposed_outputs if name != "receipt"]
        self.reset_runtime()

    def reset_runtime(self):
        self._net, self._key = None, None
        self._loaded_file_key = self._model_identity = self._snapshot = None
        self._receipt_calls = []

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["contract_path"]:
            relative_path(p["contract_path"])
        elif not isinstance(p["contract_path"], str):
            raise ValidationError("Contract path must be text", "contract_path")

    def project_assets(self):
        assets = super().project_assets()
        if self.parameters["contract_path"]:
            path, contract = resolve_contract(self)
            _, identity = model_bytes(path, bounded=True)
            if identity != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ONNX model checksum differs", "contract_path")
            assets.append((Path(self.parameters["contract_path"]).parent / "model.onnx").as_posix())
        return assets

    def begin_inference(self):
        self._receipt_calls, self._snapshot = [], None
        self._fallback_reported = False

    def finalize_result(self, result, inputs):
        result = super().finalize_result(result, inputs)
        result.outputs["receipt"] = (
            {
                "model": "onnx_inference_receipt_v1",
                "schema_version": 1,
                "receipt_mode": self.parameters["receipt_mode"],
                "configuration": configuration(self),
                "configuration_revision": revision(configuration(self)),
                "calls": copy.deepcopy(self._receipt_calls),
            }
            if self.parameters["receipt_mode"] != "off" and self._receipt_calls
            else NO_VALUE
        )
        context = result.payloads.get("tensors", result.payloads.get("info"))
        if context is not None:
            result.payloads["receipt"] = copy.deepcopy(context)
        self._snapshot = None  # Do not retain another copy of the model between activations.
        return result

    def process(self, inputs):
        self.begin_inference()
        blob, geometry = preprocess(inputs, self.parameters)
        tensors, filename = self.infer_tensors(
            ((self.parameters["input_name"], blob),),
            preprocessing={self.parameters["input_name"]: geometry["preprocessing_contract"]},
        )
        return Result(
            {"tensors": tensors, "preprocessing": geometry},
            message=f"{filename}: {len(tensors)} tensor outputs",
        )

    def infer_tensors(self, inputs, cache_token=(), *, array_outputs=False, preprocessing=None):
        """Bind every input for this activation; never keep a partially rebound network."""
        p = self.parameters
        inputs = tuple(inputs)
        try:
            limit = integer(p["max_output_elements"], "Maximum output elements", 1, 8_000_000)
            names = p["output_names"]
            if (
                not isinstance(names, list)
                or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("Output names must be a list of unique nonempty strings")
            if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
                raise ValueError("CUDA backend and CUDA target must be selected together")
            if p["target"] != "cpu" and p["engine"] != "classic":
                raise ValueError("Select the classic engine for non-CPU targets")
            if not p["fusion"] and p["engine"] != "classic":
                raise ValueError("Select the classic engine to disable fusion")
            selected = {k: p[k] for k in ("engine", "backend", "target", "fusion")}
            requested = dict(selected)
            available_targets = [
                int(t) for t in cv2.dnn.getAvailableTargets(BACKENDS[p["backend"]])
            ]
            selected_available = available_targets
            fallback = None
            if TARGETS[p["target"]] not in available_targets:
                if p["fallback_policy"] == "error":
                    raise ValueError(
                        f"DNN target {p['target']} is unavailable for backend {p['backend']}"
                    )
                selected_available = [
                    int(t) for t in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV)
                ]
                if cv2.dnn.DNN_TARGET_CPU not in selected_available:
                    raise ValueError("Requested target and OpenCV CPU fallback are unavailable")
                selected.update(engine="classic", backend="opencv", target="cpu")
                fallback = "requested_target_unavailable"
                if not self._fallback_reported:
                    self.logger.warning(
                        f"DNN {p['backend']}/{p['target']} unavailable; using OpenCV CPU"
                    )
                    self._fallback_reported = True
            if self._snapshot is None:
                path, contract = resolve_contract(self)
                current_file_key = file_key(path)
                if contract is not None or current_file_key != self._loaded_file_key:
                    raw, identity = model_bytes(path, bounded=contract is not None)
                else:
                    raw, identity = None, self._model_identity
                if contract is not None and identity != {
                    k: contract["artifact"][k] for k in ("sha256", "bytes")
                }:
                    raise ValidationError("Bundled ONNX model checksum differs", "contract_path")
                self._snapshot = path, contract, current_file_key, raw, identity
            path, contract, current_file_key, raw, identity = self._snapshot
            if (
                contract
                and contract["preprocessing"]
                and canonical(contract["preprocessing"]) != canonical(preprocessing or {})
            ):
                raise ValidationError(
                    "Prepared input preprocessing differs from ONNX contract", "contract_path"
                )
            key = (
                *current_file_key,
                identity["sha256"],
                *selected.values(),
                cache_token,
            )
            reused = key == self._key
            if key != self._key:
                if raw is None:
                    raw, identity = model_bytes(path)
                    self._snapshot = path, contract, current_file_key, raw, identity
                    key = (*current_file_key, identity["sha256"], *selected.values(), cache_token)
                net = cv2.dnn.readNetFromONNX(
                    np.frombuffer(raw, np.uint8), ENGINES[selected["engine"]]
                )
                if selected["engine"] == "classic":
                    net.setPreferableBackend(BACKENDS[selected["backend"]])
                    net.setPreferableTarget(TARGETS[selected["target"]])
                    net.enableFusion(bool(selected["fusion"]))
                self._net, self._key = net, key
                self._loaded_file_key, self._model_identity = current_file_key, identity
            for name, blob in inputs:
                self._net.setInput(blob, name)
            available = list(self._net.getUnconnectedOutLayersNames())
            names = names or available
            if any(name not in available for name in names):
                raise ValidationError("Select existing terminal model output names", "output_names")
            # This OpenCV 5 engine requires the complete terminal-output set even when
            # callers retain a subset. Preserve requested order after one native forward.
            values = self._net.forward(available)
            if isinstance(values, np.ndarray):
                values = [values]
            if len(values) != len(available):
                raise ValueError("Network returned an unexpected number of terminal outputs")
            by_name = dict(zip(available, values, strict=True))
            values = [by_name[name] for name in names]
            if sum(v.size for v in values) > limit:
                raise ValueError("Network output exceeds configured element limit")
            # Batch adapters split native rows before making portable records, avoiding
            # an unnecessary full-batch Python-list conversion and reconstruction.
            tensors = {
                name: array(value, "Network output", limit)
                if array_outputs
                else tensor_record(value, limit)
                for name, value in zip(names, values)
            }
            if p["receipt_mode"] != "off":
                detailed = p["receipt_mode"] == "detailed"
                self._receipt_calls.append(
                    {
                        "artifact": dict(identity),
                        "contract_revision": contract["revision"] if contract else None,
                        "configuration_verified": contract is not None,
                        "external_preprocessing_verified": bool(
                            contract and contract["preprocessing"]
                        ),
                        "environment": environment(),
                        "requested_execution": requested,
                        "selected_execution": selected,
                        "requested_available_target_ids": available_targets,
                        "selected_available_target_ids": selected_available,
                        "fallback": fallback,
                        "native_placement_verified": False,
                        "native_placement_note": (
                            "Selection/availability do not prove per-layer execution placement."
                        ),
                        "cache_reused": reused,
                        "preprocessing": copy.deepcopy(preprocessing or {}),
                        "inputs": [
                            tensor_description(name, value, detailed) for name, value in inputs
                        ],
                        "outputs": [
                            tensor_description(name, value, detailed)
                            for name, value in zip(names, values)
                        ],
                    }
                )
            return tensors, path.name
        except Exception:
            self.reset_runtime()
            raise


class TensorVector(Block):
    type_id, title = "ml.tensor_vector", "Tensor to embedding vector"
    category, block_type, subtype = "ML", "ML", "Embeddings"
    inputs = (Port("tensors", "data"),)
    outputs = (Port("vector", "data"), Port("dimensions", "number"))
    defaults = {"output_name": "", "pooling": "flatten", "channel_axis": 1, "normalize": "l2"}
    parameter_choices = {"pooling": ("flatten", "global_average"), "normalize": ("none", "l2")}

    def process(self, inputs):
        from .ml_common import normalized_vectors

        p = self.parameters
        tensor = selected_tensor(inputs["tensors"], p["output_name"])
        if tensor.shape[0] != 1:
            raise ValueError("Embedding conversion requires a single-image batch")
        if p["pooling"] == "global_average":
            axis = integer(p["channel_axis"], "Channel axis", 1, tensor.ndim - 1)
            vector = tensor.mean(axis=tuple(i for i in range(tensor.ndim) if i != axis))
        else:
            vector = tensor.reshape(-1)
        vector = normalized_vectors(vector, p["normalize"])
        return Result({"vector": vector.tolist(), "dimensions": int(vector.size)})


class ClassifyTensor(Block):
    type_id, title = "ml.classify", "Classify DNN tensor"
    category, block_type, subtype = "ML", "ML", "Classification"
    inputs = (Port("tensors", "data"),)
    outputs = (Port("predictions", "data"), Port("label", "data"), Port("confidence", "number"))
    defaults = {
        "output_name": "",
        "labels": [],
        "softmax": True,
        "top_k": 5,
        "minimum_confidence": 0.0,
        "accepted_labels": [],
    }

    def process(self, inputs):
        p = self.parameters
        scores = selected_tensor(inputs["tensors"], p["output_name"])
        if scores.ndim > 1 and scores.shape[0] != 1:
            raise ValueError("Classification requires a single-image batch")
        scores = scores.reshape(-1).astype(np.float64)
        if p["softmax"]:
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
        elif np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Disable softmax only for probability outputs in [0,1]")
        labels = p["labels"]
        if not isinstance(labels, list) or (
            labels and (len(labels) != scores.size or any(not isinstance(v, str) for v in labels))
        ):
            raise ValueError("Labels must be empty or one string per class")
        accepted = p["accepted_labels"]
        if not isinstance(accepted, list) or any(not isinstance(v, str) for v in accepted):
            raise ValueError("Accepted labels must be a string list")
        threshold = finite(p["minimum_confidence"], "Minimum confidence", 0)
        if threshold > 1:
            raise ValueError("Minimum confidence must be in [0,1]")
        order = np.argsort(-scores, kind="stable")[: integer(p["top_k"], "Top K", 1, 10000)]
        predictions = [
            {
                "index": int(i),
                "label": labels[i] if labels else str(i),
                "confidence": float(scores[i]),
            }
            for i in order
        ]
        best = predictions[0]
        passed = best["confidence"] >= threshold and (not accepted or best["label"] in accepted)
        return Result(
            {"predictions": predictions, "label": best["label"], "confidence": best["confidence"]},
            passed=passed,
            score=best["confidence"],
            message=f"{best['label']}: {best['confidence']:.1%}",
            fail_reasons=[]
            if passed
            else [{"code": "classification", "message": "Class or confidence rejected"}],
        )


DNN_BLOCKS = (DNNInference, TensorVector, ClassifyTensor)
