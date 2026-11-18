"""Portable native HOG detectors, linear image-SVM export and bounded scale searches."""

import copy
import json

import cv2
import numpy as np

from .classifier_common import canonical_json, model_revision
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .hog import HOGFeatures, _sizes
from .images import convert_depth
from .ml_common import local_path
from .native_runtime import cpu_execution as cpu_search
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image, image_value
from .selection import contains, selection_mask
from .svm_blocks import read_svm

MODEL = "opencv_hog_detector_v1"
MAX_BYTES = 8 * 1024**2
PRESETS = ("default_people", "daimler_people")


def geometry(p):
    settings = {key: p[key] for key in HOGFeatures.defaults}
    validator = HOGFeatures(parameters=settings)
    try:
        validator.validate_parameters()
    finally:
        validator.close()
    for key in ("cell_size", "block_size", "block_stride"):
        if any(type(v) is not int for v in settings[key]):
            raise ValidationError("HOG dimensions must use integers", key)
    if type(settings["bins"]) is not int:
        raise ValidationError("HOG bins must use an integer", "bins")
    size = p["window_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or not 8 <= v <= 2048 for v in size)
    ):
        raise ValidationError("Window size needs two integers in 8..2048", "window_size")
    cell, block, stride = _sizes(settings)
    if any(value & (value - 1) for value in stride):
        raise ValidationError(
            "Native HOG detector cache needs power-of-two block strides", "block_stride"
        )
    if any(n < b or (n - b) % s for n, b, s in zip(size, block, stride, strict=True)):
        raise ValidationError("Whole HOG blocks must fit the detector window", "window_size")
    length = int(
        np.prod([(n - b) // s + 1 for n, b, s in zip(size, block, stride, strict=True)])
        * np.prod([b // c for b, c in zip(block, cell, strict=True)])
        * settings["bins"]
    )
    if not 1 <= length <= 65536 or size[0] * size[1] > 1_048_576:
        raise ValidationError("Detector exceeds 65,536 features or one million window pixels")
    if p["working_encoding"] not in ("gray", "bgr"):
        raise ValidationError("Working encoding must be gray or bgr", "working_encoding")
    return {
        **copy.deepcopy(settings),
        "window_size": list(size),
        "working_encoding": p["working_encoding"],
    }, length


def make_model(g, coefficients, label, source):
    _, length = geometry(g)
    if not isinstance(label, str) or not 1 <= len(label) <= 256:
        raise ValidationError("Detector label must be nonempty text of at most 256 characters")
    values = np.asarray(coefficients, np.float64)
    if (
        values.ndim != 1
        or len(values) not in (length, length + 1)
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 1e6)
    ):
        raise ValidationError(
            "Supply one finite weight per feature and optional bias, within +/-1e6", "coefficients"
        )
    if len(values) == length:
        values = np.r_[values, 0.0]
    model = {
        "model": MODEL,
        "opencv_version": cv2.__version__,
        "geometry": g,
        "coefficients": values.astype(np.float32).tolist(),
        "label": label,
        "source": source,
    }
    model["revision"] = model_revision(model)
    read_model(model)
    return model


def read_model(model):
    if (
        not isinstance(model, dict)
        or set(model)
        != {"model", "opencv_version", "geometry", "coefficients", "label", "source", "revision"}
        or model.get("model") != MODEL
    ):
        raise ValidationError("Connect a portable HOG detector model")
    if len(canonical_json(model).encode("utf-8")) > MAX_BYTES or model[
        "revision"
    ] != model_revision(model):
        raise ValidationError("HOG model exceeds 8 MiB or its revision differs")
    if model["opencv_version"] != cv2.__version__:
        raise ValidationError("HOG model belongs to a different OpenCV version")
    g = model["geometry"]
    if not isinstance(g, dict) or set(g) != set(HOGFeatures.defaults) | {
        "window_size",
        "working_encoding",
    }:
        raise ValidationError("HOG geometry fields differ")
    _, length = geometry(g)
    values = np.asarray(model["coefficients"], np.float64)
    if (
        values.shape != (length + 1,)
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 1e6)
        or not np.array_equal(values, values.astype(np.float32).astype(np.float64))
    ):
        raise ValidationError("HOG coefficients must retain bounded exact float32 values and bias")
    if (
        not isinstance(model["label"], str)
        or not 1 <= len(model["label"]) <= 256
        or not isinstance(model["source"], dict)
        or len(canonical_json(model["source"]).encode("utf-8")) > 16384
    ):
        raise ValidationError("Invalid HOG label/source metadata")
    return g, values.astype(np.float32)


def builtin_model(preset, encoding="gray"):
    if preset not in PRESETS:
        raise ValidationError("Choose a supported native people detector")
    g = {
        **HOGFeatures.defaults,
        "window_size": [64, 128] if preset == "default_people" else [48, 96],
        "working_encoding": encoding,
    }
    values = (
        cv2.HOGDescriptor.getDefaultPeopleDetector()
        if preset == "default_people"
        else cv2.HOGDescriptor.getDaimlerPeopleDetector()
    )
    return make_model(g, values, "person", {"preset": preset})


def native_hog(g, levels=64):
    return cv2.HOGDescriptor(
        _winSize=tuple(g["window_size"]),
        _cellSize=tuple(g["cell_size"]),
        _blockSize=tuple(g["block_size"]),
        _blockStride=tuple(g["block_stride"]),
        _nbins=g["bins"],
        _winSigma=g["win_sigma"],
        _L2HysThreshold=g["l2_hys_threshold"],
        _gammaCorrection=g["gamma_correction"],
        _signedGradient=g["signed_gradient"],
        _nlevels=levels,
    )


def svm_geometry(model):
    value = model.get("feature_contract")
    if (
        not isinstance(value, dict)
        or value.get("model") != "opencv_hog_window_v1"
        or value.get("input_depth") != "uint8"
        or model.get("normalization") != "none"
    ):
        raise ValidationError("Train an unnormalized SVM on native uint8 HOG window descriptors")
    settings = value.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(HOGFeatures.defaults):
        raise ValidationError("HOG training settings are missing")
    g, length = geometry(
        {
            **settings,
            "window_size": value.get("window_size"),
            "working_encoding": value.get("working_encoding"),
        }
    )
    cell, block, stride = _sizes(settings)
    expected = {
        "model": "opencv_hog_window_v1",
        "opencv_version": cv2.__version__,
        "working_encoding": g["working_encoding"],
        "input_depth": "uint8",
        "window_size": g["window_size"],
        "cell_grid": [n // c for n, c in zip(g["window_size"], cell, strict=True)],
        "block_grid": [
            (n - b) // s + 1 for n, b, s in zip(g["window_size"], block, stride, strict=True)
        ],
        "descriptor_length": length,
        "order": ["block_x", "block_y", "cell_x", "cell_y", "bin"],
        "settings": settings,
    }
    if value != expected or model.get("dimensions") != length:
        raise ValidationError("SVM's HOG window/settings contract differs")
    return g


class _Detector(Block):
    category, block_type, subtype = "ML", "ML", "HOG detection"
    process_inputs_readonly = parallel_safe = preview_safe = True


class HOGDetectorModel(_Detector):
    type_id, title = "ml.hog_detector_model", "Configure HOG detector model"
    inputs = (Port("svm", "data", required=False, required_when_connected=True),)
    outputs = (Port("model", "data"), Port("report", "data"))
    defaults = {
        **HOGFeatures.defaults,
        "preset": "default_people",
        "window_size": [64, 128],
        "working_encoding": "gray",
        "coefficients": [],
        "label": "object",
        "positive_label": "part",
    }
    parameter_choices = {
        "preset": (*PRESETS, "coefficients", "linear_svm"),
        "working_encoding": ("gray", "bgr"),
    }
    parameter_ranges = HOGFeatures.parameter_ranges
    parameter_help = {
        **HOGFeatures.parameter_help,
        "preset": "Native people geometry is fixed. Coefficients uses editable geometry; "
        "linear_svm derives it from the HOG training contract.",
        "coefficients": "Float32 weights and optional bias; required in coefficients mode.",
        "window_size": "[Width,height] in pixels; used by coefficients mode.",
        "working_encoding": "Grayscale or canonical BGR. SVM mode uses its training contract.",
        "positive_label": "Binary linear-SVM class that should produce positive detector margins.",
        "label": "Label for coefficients; other presets use person or Positive label.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        geometry(p)
        if not isinstance(p["coefficients"], list) or len(p["coefficients"]) > 65537:
            raise ValidationError("Coefficients must be a bounded JSON list", "coefficients")
        for key in ("label", "positive_label"):
            if not isinstance(p[key], str) or not 1 <= len(p[key]) <= 256:
                raise ValidationError("Use nonempty label text of at most 256 characters", key)
        if p["preset"] == "coefficients":
            g, _ = geometry(p)
            try:
                make_model(g, p["coefficients"], p["label"], {"preset": "coefficients"})
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(str(exc), "coefficients") from exc

    def process(self, inputs):
        p = self.parameters
        if p["preset"] in PRESETS:
            model = builtin_model(p["preset"], p["working_encoding"])
        elif p["preset"] == "coefficients":
            g, _ = geometry(p)
            model = make_model(g, p["coefficients"], p["label"], {"preset": "coefficients"})
        else:
            svm_model = inputs.get("svm")
            if not isinstance(svm_model, dict):
                raise ValidationError("Connect a binary linear HOG SVM")
            g = svm_geometry(svm_model)
            svm = read_svm(svm_model)
            labels = svm_model["class_labels"]
            if (
                svm.getKernelType() != cv2.ml.SVM_LINEAR
                or len(labels) != 2
                or p["positive_label"] not in labels
            ):
                raise ValidationError(
                    "Use a binary linear SVM and one of its class labels", "positive_label"
                )
            rho, alpha, indices = svm.getDecisionFunction(0)
            vector = alpha.reshape(-1) @ svm.getSupportVectors()[indices.reshape(-1)]
            sign = 1 if labels.index(p["positive_label"]) == 0 else -1
            model = make_model(
                g,
                np.r_[sign * vector, -sign * rho],
                p["positive_label"],
                {
                    "preset": "linear_svm",
                    "svm_revision": svm_model["revision"],
                    "positive_label": p["positive_label"],
                },
            )
        return Result(
            {
                "model": model,
                "report": {
                    "revision": model["revision"],
                    "feature_count": len(model["coefficients"]) - 1,
                    "source": model["source"],
                    "geometry": model["geometry"],
                },
            },
            message="Configured " + model["label"] + " HOG detector",
        )


def search_plan(shape, g, p):
    stride, padding = p["win_stride"], p["padding"]
    if any(s % b for s, b in zip(stride, g["block_stride"], strict=True)):
        raise ValidationError(
            "Window stride must be a multiple of detector block stride", "win_stride"
        )
    pad = [int(np.ceil(v / b)) * b for v, b in zip(padding, g["block_stride"], strict=True)]
    scale, levels, windows, pixels = 1.0, [], 0, 0
    height, width = shape[:2]
    for _ in range(p["pyramid_levels"]):
        size = [round(width / scale), round(height / scale)]
        if any(n < w for n, w in zip(size, g["window_size"], strict=True)):
            break
        counts = [
            (n + 2 * a - w) // s + 1
            for n, a, w, s in zip(size, pad, g["window_size"], stride, strict=True)
        ]
        windows += int(np.prod(counts))
        pixels += (size[0] + 2 * pad[0]) * (size[1] + 2 * pad[1])
        levels.append({"scale": scale, "size": size, "windows": int(np.prod(counts))})
        scale *= p["scale_factor"]
    _, length = geometry(g)
    work = windows * length
    memory = pixels * 32 + windows * 32
    if windows > p["maximum_windows"] or work > p["max_work"]:
        raise ValidationError(
            "HOG search exceeds window/work allowance", "maximum_windows", "max_work"
        )
    if memory > p["max_bytes_mb"] * 1024**2:
        raise ValidationError("HOG search exceeds estimated workspace allowance", "max_bytes_mb")
    return {
        "levels": levels,
        "windows": windows,
        "pyramid_pixels": pixels,
        "feature_work": work,
        "estimated_workspace_bytes": memory,
        "effective_padding": pad,
    }


def suppress(boxes, scores, threshold):
    pending = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), *boxes[i], i))
    keep = []
    while pending:
        i = pending.pop(0)
        keep.append(i)
        if not pending:
            break
        other = boxes[pending]
        low = np.maximum(boxes[i, :2], other[:, :2])
        high = np.minimum(boxes[i, :2] + boxes[i, 2:], other[:, :2] + other[:, 2:])
        intersection = np.maximum(0, high - low).prod(axis=1)
        union = boxes[i, 2:].prod() + other[:, 2:].prod(axis=1) - intersection
        pending = [
            index
            for index, overlap in zip(pending, intersection / union, strict=True)
            if overlap <= threshold
        ]
    return keep


class DetectHOG(_Detector):
    type_id, title = "ml.hog_detect", "HOG sliding-window detection"
    requires_opencl_context = True
    inputs = (
        Port("image"),
        Port("model", "data", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("collection", "data"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "detector": "default_people",
        "hit_threshold": 0.0,
        "win_stride": [8, 8],
        "padding": [0, 0],
        "scale_factor": 1.05,
        "pyramid_levels": 64,
        "grouping": "none",
        "group_threshold": 2.0,
        "nms_enabled": True,
        "nms_iou": 0.4,
        "selection_policy": "center",
        "minimum_coverage": 1.0,
        "score_mode": "margin",
        "minimum_count": 1,
        "maximum_count": 0,
        "maximum_image_pixels": 1_048_576,
        "maximum_windows": 100000,
        "maximum_candidates": 2048,
        "maximum_results": 512,
        "max_work": 256_000_000,
        "max_bytes_mb": 128.0,
    }
    parameter_choices = {
        "detector": PRESETS,
        "grouping": ("none", "rectangles", "meanshift"),
        "selection_policy": ("center", "coverage", "center_and_coverage"),
        "score_mode": ("margin", "sigmoid"),
    }
    parameter_ranges = {
        "hit_threshold": NumericRange(-1e12, 1e12),
        "scale_factor": NumericRange(1.001, 4),
        "pyramid_levels": NumericRange(1, 128, whole=True),
        "group_threshold": NumericRange(0, 100),
        "nms_iou": NumericRange(0, 1),
        "minimum_coverage": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 10000, whole=True),
        "maximum_count": NumericRange(0, 10000, whole=True),
        "maximum_image_pixels": NumericRange(1, 4_194_304, whole=True),
        "maximum_windows": NumericRange(1, 200000, whole=True),
        "maximum_candidates": NumericRange(1, 10000, whole=True),
        "maximum_results": NumericRange(1, 10000, whole=True),
        "max_work": NumericRange(1, 1_000_000_000, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
    }
    parameter_help = {
        "detector": "Fallback native people model only when Model is not wired.",
        "hit_threshold": "Inclusive native SVM margin, including saved bias; not a probability.",
        "win_stride": "[X,Y] step in level pixels; must use whole model block strides.",
        "padding": "Native symmetric border padding in level pixels; rounded up to block stride.",
        "scale_factor": "Pyramid scale increment; Pyramid levels=1 selects a single scale.",
        "pyramid_levels": "Native maximum search levels, also bounded by image/window size.",
        "grouping": "Native grouping before selection and optional IoU suppression.",
        "group_threshold": "Native group threshold; rectangle grouping requires a whole number.",
        "nms_iou": "Suppress overlap strictly above this IoU; raw native weights determine rank.",
        "selection_policy": "Filter visible boxes by center and/or selected coverage. "
        "Descriptors still read the whole image.",
        "minimum_coverage": "Inclusive selected fraction of each returned visible box.",
        "score_mode": "Raw native weight or its sigmoid for normalized consumers; "
        "sigmoid is not calibrated confidence.",
        "maximum_windows": "Bound the complete padded multiscale search before native work.",
        "maximum_candidates": "Reject excess candidates before suppression; no truncation.",
        "maximum_results": "Reject excess retained detections.",
        "max_work": "Estimated feature comparisons plus suppression pair allowance.",
        "max_bytes_mb": "Estimated workspace, not measured process RSS.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if rule.whole and type(p[key]) is not int:
                raise ValidationError("Use an integer value", key)
        for key, minimum in (("win_stride", 1), ("padding", 0)):
            values = p[key]
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(type(v) is not int or not minimum <= v <= 2048 for v in values)
            ):
                raise ValidationError(f"{key} needs two integers in {minimum}..2048", key)
        if type(p["nms_enabled"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "nms_enabled")
        if p["grouping"] == "rectangles" and p["group_threshold"] % 1:
            raise ValidationError(
                "Rectangle group threshold must be a whole number", "group_threshold"
            )
        if p["maximum_count"] and p["maximum_count"] < p["minimum_count"]:
            raise ValidationError(
                "Maximum count must be zero or at least Minimum count",
                "maximum_count",
                "minimum_count",
            )

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        if (
            max(source.shape[:2]) > 16384
            or source.shape[0] * source.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError("Source exceeds HOG image allowance", "maximum_image_pixels")
        model = inputs.get("model", NO_VALUE)
        if model is NO_VALUE:
            model = builtin_model(p["detector"])
        g, coefficients = read_model(model)
        plan = search_plan(source.shape, g, p)
        selected = selection_mask(inputs, source)
        if source.ndim == 3 and source.shape[2] == 4:
            alpha = np.uint8(source[..., 3] > 0) * 255
            selected = alpha if selected is None else cv2.bitwise_and(selected, alpha)
        if source.dtype.kind == "f" and (np.any(source < 0) or np.any(source > 1)):
            raise ValidationError("Floating images must use fixed [0,1] intensity units")
        if g["working_encoding"] == "gray":
            working = gray_image(source, (np.uint8,), auto_depth=True)
        else:
            working = image_value(
                source, (np.uint8, np.uint16, np.float32, np.float64), canonical=True
            )
            working = convert_depth(working, "uint8")
            if working.ndim == 2:
                working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
            elif working.shape[2] == 4:
                working = working[:, :, :3]
        boxes, weights = np.empty((0, 4)), np.empty(0)
        if plan["windows"] and (selected is None or np.any(selected)):
            with cpu_search():
                native = native_hog(g, p["pyramid_levels"])
                native.setSVMDetector(coefficients)
                raw, values = native.detectMultiScale(
                    working,
                    hitThreshold=p["hit_threshold"],
                    winStride=tuple(p["win_stride"]),
                    padding=tuple(p["padding"]),
                    scale=p["scale_factor"],
                    groupThreshold=0 if p["grouping"] == "none" else p["group_threshold"],
                    useMeanshiftGrouping=p["grouping"] == "meanshift",
                )
            boxes = np.asarray(raw, np.float64).reshape(-1, 4)
            weights = np.asarray(values, np.float64).reshape(-1)
        if (
            len(boxes) != len(weights)
            or len(boxes) > p["maximum_candidates"]
            or not np.isfinite(boxes).all()
            or not np.isfinite(weights).all()
        ):
            raise ValidationError(
                "Native HOG candidates are invalid or exceed Maximum candidates",
                "maximum_candidates",
            )
        if len(boxes) and (
            np.any(boxes[:, 2:] <= 0)
            or np.any(boxes[:, :2] < 0)
            or np.any(boxes[:, :2] + boxes[:, 2:] > [source.shape[1], source.shape[0]])
        ):
            raise ValidationError("Native HOG boxes are outside the visible source image")
        keep = []
        for i, (x, y, w, h) in enumerate(boxes):
            center = contains(selected, [x + (w - 1) / 2, y + (h - 1) / 2])
            coverage = (
                1.0
                if selected is None
                else float(np.count_nonzero(selected[int(y) : int(y + h), int(x) : int(x + w)]))
                / (w * h)
            )
            if (p["selection_policy"] == "coverage" or center) and (
                p["selection_policy"] == "center" or coverage >= p["minimum_coverage"]
            ):
                keep.append(i)
        if p["nms_enabled"] and plan["feature_work"] + len(keep) ** 2 > p["max_work"]:
            raise ValidationError("HOG suppression exceeds Max work", "max_work")
        ordered = (
            suppress(boxes[keep], weights[keep], p["nms_iou"])
            if p["nms_enabled"]
            else sorted(range(len(keep)), key=lambda i: (-weights[keep[i]], *boxes[keep[i]], i))
        )
        retained = [keep[i] for i in ordered]
        if len(retained) > p["maximum_results"]:
            raise ValidationError("HOG detections exceed Maximum results", "maximum_results")
        items, drawings = [], []
        for rank, index in enumerate(retained):
            x, y, w, h = map(float, boxes[index])
            weight = float(weights[index])
            score = (
                weight
                if p["score_mode"] == "margin"
                else float(1 / (1 + np.exp(-np.clip(weight, -80, 80))))
            )
            points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            items.append(
                {
                    "id": rank,
                    "native_index": index,
                    "class_id": 0,
                    "label": model["label"],
                    "score": score,
                    "native_weight": weight,
                    "score_kind": "native_hog_weight"
                    if p["score_mode"] == "margin"
                    else "sigmoid_native_hog_weight",
                    "orientation_measured": False,
                    "metrics": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "center_x": x + w / 2,
                        "center_y": y + h / 2,
                        "aspect_ratio": w / h,
                        "score": score,
                        "angle": None,
                    },
                    "geometry": {"points": points, "bbox": [x, y, w, h]},
                }
            )
            if self.generate_graphics:
                drawings.extend(
                    (
                        graphic_object("polyline", points, (70, 220, 130), closed=True),
                        graphic_object(
                            "label",
                            [[x, y]],
                            (70, 220, 130),
                            text=f"{model['label']}: {weight:.3g}",
                        ),
                    )
                )
        count = len(items)
        passed = count >= p["minimum_count"] and (
            p["maximum_count"] == 0 or count <= p["maximum_count"]
        )
        context = copy.deepcopy(self.input_payloads["image"])
        outputs = {
            "image": source,
            "collection": collection("boxes", items, source, context),
            "count": count,
            "report": {
                "model_revision": model["revision"],
                "raw_candidates": len(boxes),
                "selected_candidates": len(keep),
                "retained": count,
                "native_grouping": p["grouping"],
                "score_mode": p["score_mode"],
                "score_is_probability": False,
                "plan": plan,
                "geometry": g,
            },
        }
        return Result(
            outputs,
            payloads={key: copy.deepcopy(context) for key in outputs},
            passed=passed,
            message=f"{count} HOG detections; {len(boxes)} native candidates",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "hog.count",
                    "message": f"Retained {count}; minimum {p['minimum_count']}, "
                    f"maximum {p['maximum_count'] or 'unbounded'}",
                }
            ],
            drawings={"HOG detections": GraphicDrawing(tuple(drawings))}
            if self.generate_graphics
            else {},
        )


class SaveHOGDetector(_Detector):
    type_id, title = "ml.hog_detector_save", "Save HOG detector model"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "hog-detector.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_model(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result({"path": str(path)}, message="Saved HOG model; existing files preserved")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate HOG model JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("Nonfinite HOG model number: " + value)


class LoadHOGDetector(_Detector):
    type_id, title = "ml.hog_detector_load", "Load HOG detector model"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        with local_path(self, self.parameters["path"]).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("HOG model exceeds 8 MiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )
        read_model(model)
        return Result({"model": model}, message="Loaded " + model["label"] + " HOG model")


HOG_DETECTION_BLOCKS = (HOGDetectorModel, DetectHOG, SaveHOGDetector, LoadHOGDetector)
