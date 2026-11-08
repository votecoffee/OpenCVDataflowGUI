"""OpenCV DNN inference with explicit preprocessing and portable tensor outputs."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import format_image
from .ml_common import file_key, local_path, selected_tensor, tensor_record
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
    }
    return blob, geometry


class DNNInference(Block):
    type_id, title = "ml.dnn", "DNN ONNX inference"
    category, block_type, subtype = "ML", "ML", "DNN inference"
    inputs = SELECTION_INPUTS
    outputs = (Port("tensors", "data"), Port("preprocessing", "data"))
    defaults = {
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
        "color_format": ("rgb", "bgr", "gray"),
        "resize_mode": ("stretch", "letterbox"),
        "engine": tuple(ENGINES),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_help = {
        "model_path": "Local ONNX model; relative paths resolve beside the project.",
        "mean": "Per-channel values in source intensity units, before scaling.",
        "std": "Per-channel divisors after mean subtraction and scaling.",
        "output_names": "JSON list; empty selects all unconnected output layers.",
        "engine": "OpenCV 5 engine. Non-CPU targets require classic.",
        "max_output_elements": "Total tensor values retained per activation (bounded).",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._net, self._key = None, None

    close = reset_runtime

    def process(self, inputs):
        p = self.parameters
        blob, geometry = preprocess(inputs, p)
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
        if TARGETS[p["target"]] not in cv2.dnn.getAvailableTargets(BACKENDS[p["backend"]]):
            raise ValueError(f"DNN target {p['target']} is unavailable for backend {p['backend']}")
        path = local_path(self, p["model_path"])
        key = (*file_key(path), p["engine"], p["backend"], p["target"], p["fusion"])
        if key != self._key:
            net = cv2.dnn.readNetFromONNX(
                np.frombuffer(path.read_bytes(), np.uint8), ENGINES[p["engine"]]
            )
            if p["engine"] == "classic":
                net.setPreferableBackend(BACKENDS[p["backend"]])
                net.setPreferableTarget(TARGETS[p["target"]])
                net.enableFusion(bool(p["fusion"]))
            self._net, self._key = net, key
        self._net.setInput(blob, p["input_name"])
        names = names or list(self._net.getUnconnectedOutLayersNames())
        values = self._net.forward(names)
        if isinstance(values, np.ndarray):
            values = [values]
        if len(values) != len(names) or sum(v.size for v in values) > limit:
            raise ValueError("Network output exceeds configured element limit")
        tensors = {name: tensor_record(value, limit) for name, value in zip(names, values)}
        return Result(
            {"tensors": tensors, "preprocessing": geometry},
            message=f"{path.name}: {len(tensors)} tensor outputs",
        )


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
