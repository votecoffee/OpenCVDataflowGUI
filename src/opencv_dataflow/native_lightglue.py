"""Native OpenCV ALIKED/ONNX LightGlue with explicit feature and artifact contracts."""

import copy
import hashlib
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import opencv_image
from .local_features import SuperPointLightGlueMatch, feature_arrays
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .result_graphics import review_legend
from .selection import contains, selection_mask

ARTIFACT_DEFAULTS = {"model_path": "", "model_sha256": "", "maximum_model_bytes": 134_217_728}
ARTIFACT_RANGES = {"maximum_model_bytes": NumericRange(1, 536_870_912, whole=True)}
ARTIFACT_HELP = {
    "model_path": "Local ONNX file; never downloaded during execution. Save As rebases the path.",
    "model_sha256": "Optional exact 64-hex SHA-256; checked against the bounded import snapshot.",
    "maximum_model_bytes": (
        "Bound model bytes read before hashing/import; not an internal DNN RAM limit."
    ),
}
FAMILIES = {
    "aliked": ("lightglue_aliked_v1", 128),
    "disk": ("lightglue_disk_v1", 128),
    "superpoint": ("lightglue_superpoint_v1", 256),
}


def validate_artifact(p):
    if not isinstance(p["model_path"], str):
        raise ValidationError("Choose a local ONNX filename", "model_path")
    if not isinstance(p["model_sha256"], str) or not re.fullmatch(
        r"(?:[0-9a-fA-F]{64})?", p["model_sha256"]
    ):
        raise ValidationError("Use empty text or a 64-hex SHA-256", "model_sha256")


def snapshot(block):
    p = block.parameters
    path = local_path(block, p["model_path"])
    with path.open("rb") as stream:
        raw = stream.read(p["maximum_model_bytes"] + 1)
    if not raw or len(raw) > p["maximum_model_bytes"]:
        raise ValidationError(
            "ONNX snapshot is empty or exceeds byte budget", "maximum_model_bytes"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if p["model_sha256"] and digest != p["model_sha256"].lower():
        raise ValidationError("ONNX snapshot SHA-256 mismatch", "model_sha256")
    return raw, {"sha256": digest, "bytes": len(raw), "backend": "opencv", "target": "cpu"}


class NativeALIKEDFeatures(Block):
    type_id, title = "ml.native_aliked_features", "ALIKED ONNX features"
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
    defaults = {
        **ARTIFACT_DEFAULTS,
        "width": 640,
        "height": 640,
        "normalize_descriptors": True,
        "alpha_policy": "exclude",
        "minimum_score": 0.0,
        "minimum_count": 4,
        "maximum_features": 1024,
        "maximum_image_pixels": 4_194_304,
        "maximum_network_pixels": 4_194_304,
        "graphics_maximum_features": 256,
    }
    project_path_parameters = ("model_path",)
    parameter_choices = {"alpha_policy": ("exclude", "ignore")}
    parameter_ranges = {
        **ARTIFACT_RANGES,
        "width": NumericRange(32, 4096, whole=True),
        "height": NumericRange(32, 4096, whole=True),
        "minimum_score": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_features": NumericRange(1, 4096, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_network_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_features": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        **ARTIFACT_HELP,
        "width": "Native stretched RGB network width; must match the selected ONNX model.",
        "height": "Native stretched RGB network height; must match the selected ONNX model.",
        "normalize_descriptors": (
            "Native per-row L2 normalization; disable only for raw downstream consumers."
        ),
        "alpha_policy": "Exclude fully transparent centers or explicitly ignore alpha.",
        "minimum_score": (
            "Inclusive source-center score filter after native extraction; no top-K truncation."
        ),
        "minimum_count": "Minimum retained features for a passing inspection.",
        "maximum_features": (
            "Reject excessive raw rows before serialization; native allocations are "
            "model-dependent."
        ),
        "maximum_image_pixels": "Bound source pixels before conversion; axes are limited to 16384.",
        "maximum_network_pixels": "Bound width times height before native inference.",
        "graphics_maximum_features": "Limit optional markers without changing retained features.",
    }
    input_help = {
        "image": (
            "Source gray/BGR/RGB/alpha image; uint8, uint16 or float32 [0,1]. Output is unchanged."
        ),
        "mask": (
            "Source-center filter, not a CNN attention mask; connected missing values gate work."
        ),
        "region": (
            "Source-grid region intersected with Mask; surrounding pixels remain "
            "visible to the CNN."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model, self._key = None, None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_artifact(p)
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric settings", name)
        if type(p["normalize_descriptors"]) is not bool:
            raise ValidationError("Use a checkbox", "normalize_descriptors")
        if p["minimum_count"] > p["maximum_features"]:
            raise ValidationError("Minimum count exceeds row budget", "minimum_count")
        if p["width"] * p["height"] > p["maximum_network_pixels"]:
            raise ValidationError("Network grid exceeds pixel budget", "maximum_network_pixels")

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image_value(source, (np.uint8, np.uint16, np.float32))
        h, w = source.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError("Source exceeds pixel/axis budget", "maximum_image_pixels")
        selected = selection_mask(inputs, source)
        pixels = opencv_image(source)
        if source.dtype == np.float32:
            if np.any(pixels < 0) or np.any(pixels > 1):
                raise ValidationError("Floating image samples must be in [0,1]")
            pixels = np.rint(pixels * 255).astype(np.uint8)
        elif source.dtype == np.uint16:
            pixels = np.rint(pixels.astype(np.float64) / 257).astype(np.uint8)
        if pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        elif pixels.shape[2] == 4:
            if p["alpha_policy"] == "exclude":
                alpha = (source[..., 3] > 0).astype(np.uint8) * 255
                selected = alpha if selected is None else cv2.bitwise_and(selected, alpha)
            pixels = pixels[..., :3]
        raw, receipt = snapshot(self)
        key = (receipt["sha256"], p["width"], p["height"], p["normalize_descriptors"])
        reused = self._key == key and self._model is not None
        if not reused:
            self.reset_runtime()
            settings = cv2.ALIKED_Params()
            settings.inputSize = (p["width"], p["height"])
            settings.normalizeDescriptors = p["normalize_descriptors"]
            settings.engine = cv2.dnn.ENGINE_NEW
            settings.backend, settings.target = cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
            with tempfile.TemporaryDirectory(prefix="dataflow-aliked-") as directory:
                path = Path(directory) / "snapshot.onnx"
                path.write_bytes(raw)
                self._model = cv2.ALIKED.create(str(path), settings)
            self._key = key
        try:
            keys, descriptors = self._model.detectAndCompute(np.ascontiguousarray(pixels), None)
            count = len(keys)
            if count > p["maximum_features"]:
                raise ValidationError("Native rows exceed feature budget", "maximum_features")
            points = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)
            scores = np.asarray([k.response for k in keys], np.float32)
            descriptors = (
                np.asarray(descriptors, np.float32).reshape(-1, 128)
                if count
                else np.empty((0, 128), np.float32)
            )
            if descriptors.shape != (count, 128) or any(
                not np.isfinite(a).all() for a in (points, scores, descriptors)
            ):
                raise ValidationError("Malformed native ALIKED feature rows")
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValidationError("Native ALIKED scores must be probabilities")
        except Exception:
            self.reset_runtime()
            raise
        keep = [
            i
            for i, (xy, score) in enumerate(zip(points, scores))
            if 0 <= xy[0] <= w - 1
            and 0 <= xy[1] <= h - 1
            and score >= p["minimum_score"]
            and (selected is None or contains(selected, xy))
        ]
        context = copy.deepcopy(self.input_payloads.get("image", {}))
        features = {
            "model": "lightglue_aliked_v1"
            if p["normalize_descriptors"]
            else "native_aliked_raw_v1",
            "image_size": [w, h],
            "points": points[keep].tolist(),
            "descriptors": descriptors[keep].tolist(),
            "scores": scores[keep].tolist(),
            "coordinate_context": context,
            "source_indices": keep,
            "extraction": {**receipt, "engine": "new", "network_size": [p["width"], p["height"]]},
        }
        if p["normalize_descriptors"]:
            feature_arrays(features, model="lightglue_aliked_v1", dimension=128)
        objects = (
            [
                graphic_object("marker", [xy], (60, 210, 250))
                for xy in features["points"][: p["graphics_maximum_features"]]
            ]
            if self.generate_graphics
            else []
        )
        outputs = {
            "image": source,
            "features": features,
            "count": len(keep),
            "report": {
                **receipt,
                "engine": "new",
                "native_model_reused": reused,
                "raw_count": count,
                "retained_count": len(keep),
                "source_indices": keep,
                "graphics_truncated": self.generate_graphics and len(keep) > len(objects),
            },
        }
        passed = len(keep) >= p["minimum_count"]
        return Result(
            outputs,
            payloads={k: context for k in outputs},
            passed=passed,
            message=f"ALIKED ONNX: {len(keep)}/{count} features",
            fail_reasons=[] if passed else ["native_aliked.count"],
            drawings={"ALIKED features": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
        )


class NativeLightGlueMatch(SuperPointLightGlueMatch):
    type_id, title = "ml.native_lightglue", "LightGlue ONNX feature matcher"
    defaults = {
        k: v
        for k, v in SuperPointLightGlueMatch.defaults.items()
        if k
        not in (
            "weights_path",
            "weights_sha256",
            "device",
            "layers",
            "depth_confidence",
            "width_confidence",
            "flash",
            "mixed_precision",
        )
    }
    defaults = {
        **defaults,
        **ARTIFACT_DEFAULTS,
        "feature_family": "aliked",
        "maximum_attention_pairs": 1_048_576,
    }
    outputs = (*SuperPointLightGlueMatch.outputs, Port("report", "data"))
    project_path_parameters = ("model_path",)
    parameter_choices = {"feature_family": tuple(FAMILIES), "geometry": ("ransac", "lmeds", "rho")}
    parameter_ranges = {
        k: v for k, v in SuperPointLightGlueMatch.parameter_ranges.items() if k != "layers"
    }
    parameter_ranges = {
        **parameter_ranges,
        **ARTIFACT_RANGES,
        "maximum_attention_pairs": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        k: v
        for k, v in SuperPointLightGlueMatch.parameter_help.items()
        if k
        not in (
            "weights_path",
            "weights_sha256",
            "device",
            "layers",
            "depth_confidence",
            "width_confidence",
            "flash",
            "mixed_precision",
        )
    }
    parameter_help = {
        **parameter_help,
        **ARTIFACT_HELP,
        "feature_family": (
            "Explicit descriptor family matching the ONNX checkpoint; no family inference."
        ),
        "maximum_attention_pairs": (
            "Bound reference rows times query rows before native attention; not an "
            "internal RAM limit."
        ),
        "filter_threshold": (
            "Inclusive score >= threshold, from 1 - native DMatch.distance. Raw "
            "candidates remain in Report."
        ),
    }
    input_help = {
        "image": "Current query image, returned unchanged with its current ticket and grid.",
        "query_features": (
            "Current normalized ALIKED/DISK/SuperPoint features, with matching image context."
        ),
        "reference_features": (
            "Normalized reference features from the same descriptor family; a "
            "different ticket is permitted."
        ),
    }

    def reset_runtime(self):
        self._matcher, self._key = None, None
        self._native_report = {}

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        Block.validate_parameters(self, parameters)
        p = self.parameters if parameters is None else parameters
        validate_artifact(p)
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric settings", name)

    def feature_arrays(self, features):
        model, dimension = FAMILIES[self.parameters["feature_family"]]
        return feature_arrays(
            features,
            model=model,
            dimension=dimension,
            probability_scores=self.parameters["feature_family"] != "disk",
        )

    def match_features(self, reference, query):
        a, b = self.feature_arrays(reference), self.feature_arrays(query)
        if len(a["keypoints"]) * len(b["keypoints"]) > self.parameters["maximum_attention_pairs"]:
            raise ValidationError(
                "Feature pair exceeds attention work budget", "maximum_attention_pairs"
            )
        raw, receipt = snapshot(self)
        key = (receipt["sha256"], self.parameters["feature_family"])
        reused = self._key == key and self._matcher is not None
        if not reused:
            self.reset_runtime()
            with tempfile.TemporaryDirectory(prefix="dataflow-lightglue-") as directory:
                path = Path(directory) / "snapshot.onnx"
                path.write_bytes(raw)
                self._matcher = cv2.LightGlueMatcher.create(
                    str(path), 0.0, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
                )
            self._key = key
        try:
            # Native query/train correspond to our reference/current query respectively.
            self._matcher.setPairInfo(
                a["keypoints"],
                b["keypoints"],
                tuple(reference["image_size"]),
                tuple(query["image_size"]),
            )
            matches = self._matcher.match(a["descriptors"], b["descriptors"])
            indices = np.asarray([[m.queryIdx, m.trainIdx] for m in matches], np.int64).reshape(
                -1, 2
            )
            scores = np.asarray([1.0 - m.distance for m in matches], np.float64)
            if (
                len(matches) > min(len(a["keypoints"]), len(b["keypoints"]))
                or not np.isfinite(scores).all()
                or np.any(scores < 0)
                or np.any(scores > 1)
                or np.any(indices < 0)
                or np.any(indices[:, 0] >= len(a["keypoints"]))
                or np.any(indices[:, 1] >= len(b["keypoints"]))
                or len(set(indices[:, 0])) != len(indices)
                or len(set(indices[:, 1])) != len(indices)
            ):
                raise ValidationError(
                    "Malformed native LightGlue matches/scores or non-unique pairs"
                )
            keep = scores >= self.parameters["filter_threshold"]
            self._native_report = {
                **receipt,
                "engine": "auto (native factory)",
                "native_model_reused": reused,
                "reference_count": len(a["keypoints"]),
                "query_count": len(b["keypoints"]),
                "raw_candidate_count": len(matches),
                "retained_count": int(keep.sum()),
                "rejected_pairs": indices[~keep].tolist(),
                "rejected_scores": scores[~keep].tolist(),
                "candidate_pairs": indices.tolist(),
                "candidate_scores": scores.tolist(),
                "coordinate_normalization": "native per-axis 2*xy/[width,height]-1",
            }
            return indices[keep], scores[keep]
        except Exception:
            self.reset_runtime()
            raise
        finally:
            if self._matcher is not None:
                self._matcher.clearPairInfo()

    def process(self, inputs):
        self._native_report = {
            "raw_candidate_count": 0,
            "retained_count": 0,
            "native_inference_skipped": "empty feature set",
        }
        result = super().process(inputs)
        result.outputs["report"] = copy.deepcopy(self._native_report)
        result.outputs["matches"]["backend"] = "opencv_onnx"
        result.outputs["matches"]["feature_family"] = self.parameters["feature_family"]
        if self.generate_graphics:
            rejected = [
                inputs["query_features"]["points"][pair[1]]
                for pair in self._native_report.get("rejected_pairs", [])
            ]
            if rejected:
                result.drawings["LightGlue score rejections"] = GraphicDrawing(
                    tuple(graphic_object("marker", [xy], (40, 70, 230)) for xy in rejected)
                )
                overlay = result.outputs["overlay"]
                for xy in rejected:
                    cv2.circle(overlay, tuple(np.rint(xy).astype(int)), 3, (40, 70, 230), 1)
                diagnostic = overlay.copy()
                review_legend(
                    diagnostic,
                    result.outputs["inliers"],
                    result.outputs["count"] - result.outputs["inliers"] + len(rejected),
                )
                result.graphics["LightGlue correspondences"] = diagnostic
        return result


NATIVE_LIGHTGLUE_BLOCKS = (NativeALIKEDFeatures, NativeLightGlueMatch)
