"""Bounded native HAAR/LBP detection with distinct stage and neighbor evidence."""

import copy
import hashlib
import math
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import local_path
from .native_runtime import cpu_execution
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .selection import contains, selection_mask


def model_metadata(raw):
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValidationError("Cascade XML must not contain declarations for external data", "path")
    try:
        root = ET.fromstring(raw)
        node = root.find("cascade")
        if root.tag != "opencv_storage" or node is None or node.findtext("stageType") != "BOOST":
            raise ValueError("Choose modern OpenCV BOOST cascade XML")
        feature = node.findtext("featureType")
        if feature not in ("HAAR", "LBP"):
            raise ValueError("Choose a HAAR or LBP cascade")
        window = [int(node.findtext(key)) for key in ("width", "height")]
        stages = node.findall("stages/_")
        features = node.findall("features/_")
        if any(not 4 <= n <= 2048 for n in window) or not stages or not features:
            raise ValueError("Invalid cascade window, stages or features")
        if int(node.findtext("stageNum")) != len(stages):
            raise ValueError("Cascade stage count differs")
        stride = (
            4
            if feature == "HAAR"
            else 3 + math.ceil(int(node.findtext("featureParams/maxCatCount")) / 32)
        )
        if not 4 <= stride <= 35:
            raise ValueError("Invalid cascade feature categories")
        nodes = 0
        for tree in node.findall("stages/_/weakClassifiers/_"):
            values = tree.findtext("internalNodes", "").split()
            if not values or len(values) % stride:
                raise ValueError("Invalid cascade internal-node record")
            nodes += len(values) // stride
        if not 1 <= nodes <= 200000 or len(features) > 100000 or len(stages) > 4096:
            raise ValueError("Cascade exceeds supported model complexity")
    except (ET.ParseError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(str(exc), "path") from exc
    return {
        "feature_type": feature,
        "window_size": window,
        "stages": len(stages),
        "tree_nodes": nodes,
        "features": len(features),
    }


def search_plan(shape, model, p):
    """Mirror native scale selection; conservatively bound every pixel start."""
    height, width = shape[:2]
    base = model["window_size"]
    scales, factor = [], 1.0
    while round(base[0] * factor) <= width and round(base[1] * factor) <= height:
        if len(scales) >= p["maximum_levels"]:
            raise ValidationError(
                "Cascade scale inventory exceeds Maximum levels", "maximum_levels", "scale_factor"
            )
        scales.append(float(np.float32(factor)))
        factor *= p["scale_factor"]
    minimum, maximum = p["min_size"], p["max_size"]
    maximum = [width, height] if maximum == [0, 0] else maximum

    def size(f):
        return [round(float(np.float32(v) * np.float32(f))) for v in base]

    chosen = [
        f
        for f in scales
        if all(lo <= n <= hi for n, lo, hi in zip(size(f), minimum, maximum, strict=True))
    ]
    fallback = bool(scales and not chosen)
    if fallback:
        chosen = [
            min(
                scales,
                key=lambda f: sum((n - lo) ** 2 for n, lo in zip(size(f), minimum, strict=True)),
            )
        ]
    pixels = sum((math.ceil(width / f) + 2) * (math.ceil(height / f) + 2) for f in chosen)
    work = pixels * model["tree_nodes"]
    memory = pixels * 64
    if pixels > p["maximum_search_windows"] or work > p["max_work"]:
        raise ValidationError(
            "Cascade search exceeds window/work allowance", "maximum_search_windows", "max_work"
        )
    if memory > p["max_bytes_mb"] * 1024**2:
        raise ValidationError(
            "Cascade search exceeds estimated workspace allowance", "max_bytes_mb"
        )
    return {
        "levels": [{"factor": f, "window_size": size(f)} for f in chosen],
        "native_nearest_scale_fallback": fallback,
        "window_upper_bound": pixels,
        "node_work_upper_bound": work,
        "estimated_workspace_bytes": memory,
    }


class DetectCascade(Block):
    type_id, title = "ml.cascade_detect", "Cascade object detection"
    category, block_type, subtype = "ML", "ML", "Cascade detection"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("collection", "data"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    process_inputs_readonly = parallel_safe = preview_safe = True
    requires_opencl_context = True
    project_asset_parameters = (*Block.project_asset_parameters, "path")
    defaults = {
        "path": "",
        "model_sha256": "",
        "label": "object",
        "result_mode": "weights",
        "scale_factor": 1.1,
        "min_neighbors": 3,
        "min_size": [0, 0],
        "max_size": [0, 0],
        "strict_size": True,
        "equalize_histogram": False,
        "minimum_weight_enabled": False,
        "minimum_weight": 0.0,
        "selection_policy": "center",
        "minimum_coverage": 1.0,
        "minimum_count": 1,
        "maximum_count": 0,
        "maximum_image_pixels": 1_048_576,
        "maximum_model_mb": 16.0,
        "maximum_levels": 256,
        "maximum_search_windows": 2_000_000,
        "maximum_candidates": 10000,
        "maximum_results": 1024,
        "max_work": 20_000_000_000,
        "max_bytes_mb": 256.0,
    }
    parameter_choices = {
        "result_mode": ("weights", "neighbors", "boxes"),
        "selection_policy": ("center", "coverage", "center_and_coverage"),
    }
    parameter_ranges = {
        "scale_factor": NumericRange(1.01, 4),
        "min_neighbors": NumericRange(0, 1000, whole=True),
        "minimum_weight": NumericRange(-1e12, 1e12),
        "minimum_coverage": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 10000, whole=True),
        "maximum_count": NumericRange(0, 10000, whole=True),
        "maximum_image_pixels": NumericRange(1, 4_194_304, whole=True),
        "maximum_model_mb": NumericRange(0.001, 64),
        "maximum_levels": NumericRange(1, 4096, whole=True),
        "maximum_search_windows": NumericRange(1, 10_000_000, whole=True),
        "maximum_candidates": NumericRange(1, 50000, whole=True),
        "maximum_results": NumericRange(1, 10000, whole=True),
        "max_work": NumericRange(1, 200_000_000_000, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
    }
    parameter_help = {
        "path": "Local modern OpenCV HAAR/LBP XML. Sealed with the project on Save As.",
        "model_sha256": "Optional exact content pin; empty accepts the current local file and "
        "reports its hash.",
        "result_mode": "Weights: final-stage margin/reject level. Neighbors: grouped "
        "positive-window support. Boxes: no score.",
        "scale_factor": "Native scale step. Limits apply before native loading/search.",
        "min_neighbors": "Native grouped support threshold. Zero returns ungrouped positives.",
        "min_size": "[Width,height] object minimum; [0,0] uses the native model minimum.",
        "max_size": "[Width,height] object maximum; [0,0] uses image size. Equal nonzero "
        "min/max selects a native scale.",
        "strict_size": "Filter returned boxes to requested sizes, including native nearest-scale "
        "fallback and clipped boxes.",
        "equalize_histogram": "Equalize the uint8 gray working image; source pixels stay "
        "unchanged.",
        "minimum_weight_enabled": "Only weights mode: filter the native final-stage margin.",
        "minimum_weight": "Native weight, not probability. Enabled separately to preserve "
        "negative weights.",
        "selection_policy": "Filter returned visible boxes; descriptors still read the whole "
        "image.",
        "minimum_coverage": "Selected fraction of a returned visible box; mask and region "
        "intersect.",
        "maximum_count": "Zero disables the inspection count upper limit.",
        "maximum_search_windows": "Conservative pixel-start upper bound across native selected "
        "scales.",
        "max_work": "Window upper bound times all model tree nodes; not measured native "
        "operations.",
        "max_bytes_mb": "Estimated pyramid/candidate workspace; not measured process RSS.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if type(p[key]) not in (int, float):
                raise ValidationError("Use a numeric value", key)
            if rule.whole and type(p[key]) is not int:
                raise ValidationError("Use an integer value", key)
        for key in ("path", "model_sha256", "label"):
            if not isinstance(p[key], str) or len(p[key]) > (4096 if key == "path" else 256):
                raise ValidationError("Use bounded text", key)
        if not p["label"].strip():
            raise ValidationError("Use a nonempty label", "label")
        if p["model_sha256"] and not re.fullmatch("[0-9a-fA-F]{64}", p["model_sha256"]):
            raise ValidationError("SHA-256 needs 64 hexadecimal characters", "model_sha256")
        for key in ("min_size", "max_size"):
            value = p[key]
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(type(n) is not int or not 0 <= n <= 16384 for n in value)
                or (0 in value and value != [0, 0])
            ):
                raise ValidationError("Use [0,0] or two positive pixel dimensions up to 16384", key)
        if p["max_size"] != [0, 0] and any(
            a > b for a, b in zip(p["min_size"], p["max_size"], strict=True)
        ):
            raise ValidationError(
                "Maximum size must be at least Minimum size", "min_size", "max_size"
            )
        for key in ("strict_size", "equalize_histogram", "minimum_weight_enabled"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        if p["minimum_weight_enabled"] and p["result_mode"] != "weights":
            raise ValidationError(
                "Minimum weight requires Weights mode", "result_mode", "minimum_weight_enabled"
            )
        if p["maximum_count"] and p["maximum_count"] < p["minimum_count"]:
            raise ValidationError(
                "Maximum count must be zero or at least Minimum count",
                "minimum_count",
                "maximum_count",
            )

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        if (
            max(source.shape[:2]) > 16384
            or source.shape[0] * source.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError("Source exceeds cascade image allowance", "maximum_image_pixels")
        if source.dtype.kind == "f" and (np.any(source < 0) or np.any(source > 1)):
            raise ValidationError("Floating images must use fixed [0,1] intensity units")
        path = local_path(self, p["path"])
        limit = int(p["maximum_model_mb"] * 1024**2)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if not raw or len(raw) > limit:
            raise ValidationError(
                "Cascade exceeds model byte allowance or is empty", "path", "maximum_model_mb"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if p["model_sha256"] and digest != p["model_sha256"].lower():
            raise ValidationError("Cascade model SHA-256 differs", "model_sha256", "path")
        model = model_metadata(raw)
        plan = search_plan(source.shape, model, p)
        selected = selection_mask(inputs, source)
        if source.ndim == 3 and source.shape[2] == 4:
            alpha = np.uint8(source[..., 3] > 0) * 255
            selected = alpha if selected is None else cv2.bitwise_and(selected, alpha)
        working = gray_image(source, (np.uint8,), auto_depth=True)
        if p["equalize_histogram"]:
            working = cv2.equalizeHist(working)
        boxes, scores, rejects = np.empty((0, 4)), [], []
        with cpu_execution():
            storage = cv2.FileStorage(
                raw.decode("utf-8-sig"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
            )
            try:
                native = cv2.CascadeClassifier()
                if not native.read(storage.getFirstTopLevelNode()) or native.empty():
                    raise ValidationError("Native OpenCV rejected the cascade", "path")
                if list(native.getOriginalWindowSize()) != model[
                    "window_size"
                ] or native.getFeatureType() != (0 if model["feature_type"] == "HAAR" else 1):
                    raise ValidationError("Native cascade metadata differs", "path")
                if plan["levels"] and (selected is None or np.any(selected)):
                    kwargs = dict(
                        scaleFactor=p["scale_factor"],
                        minNeighbors=p["min_neighbors"],
                        flags=0,
                        minSize=tuple(p["min_size"]),
                        maxSize=tuple(p["max_size"]),
                    )
                    if p["result_mode"] == "weights":
                        boxes, rejects, scores = native.detectMultiScale3(
                            working, outputRejectLevels=True, **kwargs
                        )
                    elif p["result_mode"] == "neighbors":
                        boxes, scores = native.detectMultiScale2(working, **kwargs)
                    else:
                        boxes = native.detectMultiScale(working, **kwargs)
            finally:
                storage.release()
        boxes = np.asarray(boxes, np.int64).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        rejects = np.asarray(rejects).reshape(-1)
        if len(boxes) > p["maximum_candidates"]:
            raise ValidationError(
                "Native cascade candidates exceed allowance", "maximum_candidates"
            )
        if (
            p["result_mode"] != "boxes"
            and (len(scores) != len(boxes) or not np.isfinite(scores).all())
        ) or (p["result_mode"] == "weights" and len(rejects) != len(boxes)):
            raise ValidationError("Native cascade result evidence differs")
        if len(boxes) and (
            np.any(boxes[:, :2] < 0)
            or np.any(boxes[:, 2:] <= 0)
            or np.any(boxes[:, :2] + boxes[:, 2:] > [source.shape[1], source.shape[0]])
        ):
            raise ValidationError("Native cascade boxes are outside visible source")
        items, drawings = [], []
        kind = {
            "weights": "cascade_stage_weight",
            "neighbors": "cascade_group_neighbors",
            "boxes": "unscored",
        }[p["result_mode"]]
        for index in sorted(range(len(boxes)), key=lambda i: (*boxes[i], i)):
            x, y, w, h = map(int, boxes[index])
            if p["strict_size"] and any(
                n < lo or (hi and n > hi)
                for n, lo, hi in zip([w, h], p["min_size"], p["max_size"], strict=True)
            ):
                continue
            if p["minimum_weight_enabled"] and scores[index] < p["minimum_weight"]:
                continue
            center = contains(selected, [x + (w - 1) / 2, y + (h - 1) / 2])
            coverage = (
                1.0
                if selected is None
                else float(np.count_nonzero(selected[y : y + h, x : x + w])) / (w * h)
            )
            if (p["selection_policy"] != "coverage" and not center) or (
                p["selection_policy"] != "center" and coverage < p["minimum_coverage"]
            ):
                continue
            score = None if p["result_mode"] == "boxes" else float(scores[index])
            points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            item = {
                "id": len(items),
                "native_index": index,
                "class_id": 0,
                "label": p["label"],
                "score": score,
                "score_kind": kind,
                "orientation_measured": False,
                "geometry": {"bbox": [x, y, w, h], "points": points},
                "metrics": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "center_x": x + w / 2,
                    "center_y": y + h / 2,
                    "aspect_ratio": w / h,
                    "angle": None,
                    "score": score,
                },
            }
            if p["result_mode"] == "weights":
                item.update(native_weight=score, reject_level=int(rejects[index]))
            elif p["result_mode"] == "neighbors":
                item["neighbor_count"] = int(scores[index])
            items.append(item)
            if self.generate_graphics:
                suffix = (
                    ""
                    if score is None
                    else f": {score:.3g}"
                    + (" neighbors" if p["result_mode"] == "neighbors" else " weight")
                )
                drawings.extend(
                    (
                        graphic_object("polyline", points, (70, 220, 130), closed=True),
                        graphic_object("label", [[x, y]], (70, 220, 130), text=p["label"] + suffix),
                    )
                )
        if len(items) > p["maximum_results"]:
            raise ValidationError("Cascade detections exceed result allowance", "maximum_results")
        context = copy.deepcopy(self.input_payloads.get("image", {}))
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        outputs = {
            "image": source,
            "collection": collection("boxes", items, source, context),
            "count": count,
            "report": {
                "model_sha256": digest,
                "model_bytes": len(raw),
                "opencv": cv2.__version__,
                "model": model,
                "plan": plan,
                "native_candidates": len(boxes),
                "score_kind": kind,
                "score_is_probability": False,
                "selection_filters_boxes_only": True,
                "strict_size": p["strict_size"],
            },
        }
        return Result(
            outputs,
            payloads={key: copy.deepcopy(context) for key in outputs},
            passed=passed,
            message=f"{count} cascade detections; {len(boxes)} native candidates",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "cascade.count",
                    "message": f"Retained {count}; minimum {p['minimum_count']}, "
                    f"maximum {p['maximum_count'] or 'unbounded'}",
                }
            ],
            drawings={"Cascade detections": GraphicDrawing(tuple(drawings))}
            if self.generate_graphics
            else {},
        )
