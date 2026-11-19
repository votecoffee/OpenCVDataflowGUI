"""Extract learned SuperPoint features from explicit dense ONNX heads."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import opencv_image
from .local_features import MODEL, feature_arrays
from .native_lightglue import (
    ARTIFACT_DEFAULTS,
    ARTIFACT_HELP,
    ARTIFACT_RANGES,
    snapshot,
    validate_artifact,
)
from .onnx_loading import load_onnx_snapshot
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .selection import selection_mask

STRIDE = 8


def network_input(source, maximum_edge):
    pixels = opencv_image(source)
    pixels = pixels.astype(np.float32)
    if source.dtype.kind == "u":
        pixels /= np.iinfo(source.dtype).max
    if np.any(pixels < 0) or np.any(pixels > 1):
        raise ValidationError("Floating image intensities must use [0,1]")
    if pixels.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if pixels.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        pixels = cv2.cvtColor(pixels, code)
    h, w = pixels.shape
    factor = min(1, maximum_edge / max(h, w))
    rw, rh = max(1, round(w * factor)), max(1, round(h * factor))
    if (rw, rh) != (w, h):
        pixels = cv2.resize(pixels, (rw, rh), interpolation=cv2.INTER_AREA)
    # Preserve the content's top-left and pixel centers. Extend only right/bottom.
    padded = tuple(max(32, (n + STRIDE - 1) // STRIDE * STRIDE) for n in (rh, rw))
    blob = np.zeros((1, 1, *padded), np.float32)
    blob[0, 0, :rh, :rw] = pixels
    return blob, (rw, rh)


def checked_head(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValidationError(f"Expected finite float32 {name} head with shape {shape}")
    return value[0]


def heatmap(logits):
    # The 65th channel is the no-keypoint class. Keep it in the denominator.
    probabilities = np.exp(logits - logits.max(axis=0))
    probabilities /= probabilities.sum(axis=0)
    _, h, w = probabilities.shape
    subpixels = probabilities[:-1].reshape(STRIDE, STRIDE, h, w)
    return subpixels.transpose(2, 0, 3, 1).reshape(h * STRIDE, w * STRIDE)


def peak_scores(scores, radius):
    if not radius:
        return scores
    span = 2 * radius + 1
    kernel = np.ones((span, span), np.uint8)
    maxima = scores == cv2.dilate(scores, kernel)
    for _ in range(2):
        suppressed = cv2.dilate(maxima.astype(np.uint8), kernel) != 0
        available = np.where(suppressed, 0, scores)
        maxima |= (available == cv2.dilate(available, kernel)) & ~suppressed
    return np.where(maxima, scores, 0)


def sampled_descriptors(dense, x, y):
    if not len(x):
        return np.empty((0, 256), np.float32)
    # Grid-sample coordinates with zero padding and align_corners=False.
    field = dense.transpose(1, 2, 0)
    field /= np.maximum(np.linalg.norm(field, axis=2, keepdims=True), 1e-12)
    height, width = field.shape[:2]
    x, y = (x + 0.5) / STRIDE - 0.5, (y + 0.5) / STRIDE - 0.5
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - left, y - top
    sampled = np.zeros((len(x), field.shape[2]), np.float32)
    for qx, qy, weight in (
        (left, top, (1 - dx) * (1 - dy)),
        (left + 1, top, dx * (1 - dy)),
        (left, top + 1, (1 - dx) * dy),
        (left + 1, top + 1, dx * dy),
    ):
        inside = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
        sampled[inside] += field[qy[inside], qx[inside]] * weight[inside, None]
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValidationError("Selected keypoints have zero-length descriptors")
    return sampled / norms


class SuperPointONNXFeatures(Block):
    type_id, title = "ml.superpoint_onnx_features", "SuperPoint ONNX features"
    category, block_type, subtype = "ML", "ML", "Feature matching"
    preview_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("features", "data"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    project_path_parameters = ("model_path",)
    defaults = {
        **ARTIFACT_DEFAULTS,
        "maximum_edge": 640,
        "minimum_score": 0.005,
        "nms_radius": 4,
        "border": 4,
        "minimum_count": 4,
        "maximum_features": 1024,
        "maximum_image_pixels": 4_194_304,
        "graphics_maximum_features": 256,
        "alpha_policy": "exclude",
    }
    parameter_ranges = {
        **ARTIFACT_RANGES,
        "maximum_edge": NumericRange(32, 2048, whole=True),
        "minimum_score": NumericRange(0, 1),
        "nms_radius": NumericRange(0, 32, whole=True),
        "border": NumericRange(0, 64, whole=True),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_features": NumericRange(1, 4096, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_features": NumericRange(1, 4096, whole=True),
    }
    parameter_choices = {"alpha_policy": ("exclude", "ignore")}
    parameter_help = {
        **ARTIFACT_HELP,
        "maximum_edge": "Downscale only when needed. Pad the network grid to a multiple of eight.",
        "minimum_score": "Retain peak probabilities strictly above this value.",
        "nms_radius": "Square suppression radius in network pixels. Zero disables suppression.",
        "border": "Exclude this many network pixels around the unpadded content.",
        "minimum_count": "Minimum retained keypoints for a passing tool result.",
        "maximum_features": "Keep the strongest eligible points. Ties use network row-major order.",
        "maximum_image_pixels": "Bound source work before conversion. Axes are limited to 16384.",
        "graphics_maximum_features": "Cap result markers independently of the numeric features.",
        "alpha_policy": "Exclude fully transparent source centers or explicitly ignore alpha.",
    }
    input_help = {
        "image": "Gray/color uint8, uint16, or normalized float32. Output keeps source pixels.",
        "mask": "Optional source-grid filter after peak selection.",
        "region": "Optional source-grid region intersected with Mask after peak selection.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model, self._digest = None, None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_artifact(p)
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric settings", name)
        if p["minimum_count"] > p["maximum_features"]:
            raise ValidationError("Minimum count exceeds feature cap", "minimum_count")

    def infer(self, blob):
        raw, receipt = snapshot(self)
        reused = self._model is not None and self._digest == receipt["sha256"]
        try:
            if not reused:
                self._model, _ = load_onnx_snapshot(raw, cv2.dnn.ENGINE_NEW)
                self._digest = receipt["sha256"]
            self._model.setInput(blob)
            logits, descriptors = self._model.forward(["logits", "descriptors"])
            grid = (blob.shape[2] // STRIDE, blob.shape[3] // STRIDE)
            scores = checked_head(logits, (1, 65, *grid), "logits")
            desc = checked_head(descriptors, (1, 256, *grid), "descriptors")
        except Exception:
            self.reset_runtime()
            raise
        return scores, desc, {**receipt, "engine": "new", "native_model_reused": reused}

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        h, w = source.shape[:2]
        if min(h, w) < 1 or max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError("Source exceeds pixel/axis budget", "maximum_image_pixels")
        image_value(source, (np.uint8, np.uint16, np.float32))
        selection = selection_mask(inputs, source)
        blob, (rw, rh) = network_input(source, p["maximum_edge"])
        logits, dense, receipt = self.infer(blob)
        scores = heatmap(logits)
        scores[rh:, :] = 0
        scores[:, rw:] = 0
        peaks = peak_scores(scores, p["nms_radius"])
        y, x = np.where(peaks > p["minimum_score"])
        pad = p["border"]
        inside = (x >= pad) & (y >= pad) & (x < rw - pad) & (y < rh - pad)
        x, y = x[inside], y[inside]
        points = np.column_stack(((x + 0.5) * w / rw - 0.5, (y + 0.5) * h / rh - 0.5))
        xy = np.rint(points).astype(int)
        selected = np.ones(len(x), bool)
        if selection is not None:
            selected &= selection[xy[:, 1], xy[:, 0]] != 0
        if source.ndim == 3 and source.shape[2] == 4 and p["alpha_policy"] == "exclude":
            selected &= source[xy[:, 1], xy[:, 0], 3] > 0
        eligible = np.flatnonzero(selected)
        confidence = scores[y, x]
        rank = np.argsort(-confidence[eligible], kind="stable")
        keep = eligible[rank[: p["maximum_features"]]]
        network_h, network_w = blob.shape[2:]
        context = copy.deepcopy(self.input_payloads.get("image", {}))
        features = {
            "model": MODEL,
            "image_size": [w, h],
            "points": points[keep].tolist(),
            "scores": confidence[keep].tolist(),
            "descriptors": sampled_descriptors(dense, x[keep], y[keep]).tolist(),
            "raw_indices": (y[keep] * network_w + x[keep]).tolist(),
            "coordinate_context": context,
        }
        feature_arrays(features)
        count = len(keep)
        report = {
            **receipt,
            "model": "superpoint_dense_v1",
            "detector_executed": True,
            "network_size": [network_w, network_h],
            "content_size": [rw, rh],
            "peak_count": len(x),
            "eligible_count": len(eligible),
            "retained_count": count,
        }
        outputs = {"image": source, "features": features, "count": count, "report": report}
        passed = count >= p["minimum_count"]
        drawings = {}
        if self.generate_graphics:
            objects = [
                graphic_object("marker", [point], (60, 210, 250))
                for point in features["points"][: p["graphics_maximum_features"]]
            ]
            drawings["SuperPoint features"] = GraphicDrawing(tuple(objects))
        return Result(
            outputs,
            payloads={name: context for name in outputs},
            passed=passed,
            message=f"SuperPoint ONNX: {count}/{len(eligible)} eligible features",
            fail_reasons=[] if passed else ["superpoint.count"],
            drawings=drawings,
        )
