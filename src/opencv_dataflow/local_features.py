"""Explicit local-descriptor tensor packing and optional learned matching."""

import copy
import re

import numpy as np

from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry
from .graphics_layers import GraphicDrawing, graphic_object
from .lightglue_blocks import SIFT_DEFAULTS, LightGlueMatch
from .ml_common import digest_file, file_key, local_path, selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, check_spatial, require_same_frame, validate_payload
from .processing import image_value
from .selection import contains, selection_mask
from .tickets import ticket_key

MODEL = "lightglue_superpoint_v1"


def read_head(inputs, name, limit):
    record = inputs["tensors"].get(name)
    if (
        not isinstance(record, dict)
        or record.get("dtype") != "float32"
        or not isinstance(record.get("values"), list)
        or len(record["values"]) > limit
        or any(type(v) not in (float, int) for v in record["values"])
    ):
        raise ValidationError("Use named float32 heads with bounded flat numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        return selected_tensor(
            inputs["tensors"], name, max_elements=limit, allow_empty=True
        ).astype(np.float64)


def feature_arrays(value, *, model=MODEL, dimension=256, probability_scores=True):
    if not isinstance(value, dict) or value.get("model") != model:
        raise ValidationError(
            f"Connect declared {model} features with the matching descriptor family"
        )
    points, descriptors, scores = (value.get(k) for k in ("points", "descriptors", "scores"))
    if (
        not isinstance(points, list)
        or len(points) > 4096
        or not isinstance(descriptors, list)
        or len(descriptors) != len(points)
        or not isinstance(scores, list)
        or len(scores) != len(points)
        or any(not isinstance(row, list) or len(row) != 2 for row in points)
        or any(not isinstance(row, list) or len(row) != dimension for row in descriptors)
        or any(
            type(v) not in (float, int)
            for rows in (points, descriptors)
            for row in rows
            for v in row
        )
        or any(type(v) not in (float, int) for v in scores)
    ):
        raise ValidationError(f"Bounded XY, {dimension}-descriptor and score rows must correspond")
    try:
        xy = np.asarray(points, np.float32).reshape(-1, 2)
        desc = np.asarray(descriptors, np.float32).reshape(-1, dimension)
        conf = np.asarray(scores, np.float32)
        size = value.get("image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(n) is not int or not 1 <= n <= 16384 for n in size)
            or any(not np.isfinite(v).all() for v in (xy, desc, conf))
            or np.any(xy < 0)
            or np.any(xy > np.array(size) - 1)
            or probability_scores
            and (np.any(conf < 0) or np.any(conf > 1))
            or not np.allclose(np.linalg.norm(desc, axis=1), 1, atol=1e-3, rtol=0)
        ):
            raise ValueError("Invalid coordinates, confidence or non-unit descriptors")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Malformed local features: " + str(exc)) from exc
    return {"keypoints": xy, "descriptors": desc, "image_size": np.array(size, np.float32)}


class PackSuperPointFeatures(Block):
    type_id, title = "ml.pack_superpoint_features", "Pack SuperPoint feature tensors"
    category, block_type, subtype = "ML", "ML", "Feature matching"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("features", "data"),
        Port("indices", "data"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "points_output": "keypoints",
        "descriptors_output": "descriptors",
        "scores_output": "scores",
        "descriptor_layout": "rows",
        "coordinate_units": "network_pixels",
        "coordinate_origin": "pixel_centers",
        "minimum_score": 0.0,
        "normalize_descriptors": True,
        "minimum_count": 4,
        "maximum_features": 1024,
        "maximum_tensor_elements": 2_000_000,
        "maximum_image_pixels": 4_194_304,
        "graphics_maximum_features": 256,
    }
    parameter_choices = {
        "descriptor_layout": ("rows", "channels_first"),
        "coordinate_units": ("network_pixels", "normalized"),
        "coordinate_origin": ("pixel_centers", "pixel_edges"),
    }
    parameter_ranges = {
        "minimum_score": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_features": NumericRange(1, 4096, whole=True),
        "maximum_tensor_elements": NumericRange(1, 2_000_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_features": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "points_output": "Named float32 [K,2] XY tensor, optionally preceded by image batch one.",
        "descriptors_output": "Named float32 [K,256] or [256,K] tensor, optional image batch one.",
        "scores_output": "Float32 [K] or [1,K] probabilities; blank explicitly uses score one.",
        "descriptor_layout": "Descriptor rows or channels first; never inferred from dimensions.",
        "coordinate_units": "Network pixels or fractions multiplied by full network dimensions.",
        "coordinate_origin": "Direct XY denote integer-origin centers or continuous pixel edges.",
        "minimum_score": "Keep scores >= this threshold, then source/ROI filtering.",
        "normalize_descriptors": "L2-normalize nonzero rows; otherwise verify existing unit norms.",
        "minimum_count": "Fewest retained features required for a normal inspection pass.",
        "maximum_features": "Bound raw rows before filtering; never truncate correspondences.",
        "maximum_tensor_elements": "Bound each serialized head before conversion.",
        "maximum_image_pixels": "Bound source pixels; source axes are limited to 16384.",
        "graphics_maximum_features": "Bound optional markers without changing numeric results.",
    }
    input_help = {
        "image": "Original source image, retained with its depth, format and current ticket.",
        "tensors": "Explicit exported SuperPoint heads; this block does not run a detector.",
        "preprocessing": "Matching inference resize/letterbox inverse and source/network sizes.",
        "mask": "Optional source-grid selection; a connected missing event prevents work.",
        "region": "Optional source-grid shape intersected with Mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric feature settings", name)
        for k in ("points_output", "descriptors_output", "scores_output"):
            if not isinstance(p[k], str) or len(p[k]) > 256 or (k != "scores_output" and not p[k]):
                raise ValidationError("Use bounded explicit tensor names", k)
        if p["points_output"] == p["descriptors_output"]:
            raise ValidationError("Point and descriptor heads must differ", "descriptors_output")
        if type(p["normalize_descriptors"]) is not bool:
            raise ValidationError("Descriptor normalization is a checkbox", "normalize_descriptors")
        if p["minimum_count"] > p["maximum_features"]:
            raise ValidationError("Minimum count exceeds row budget", "minimum_count")

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        image_value(image)
        h, w = image.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError(
                "Source exceeds feature pixel/axis budget", "maximum_image_pixels"
            )
        checked = check_context(self, kind="Local feature")
        network, reverse = preprocessing_geometry(inputs["preprocessing"], image)

        def head(name):
            return read_head(inputs, name, p["maximum_tensor_elements"])

        points, desc = head(p["points_output"]), head(p["descriptors_output"])
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if desc.ndim == 3 and desc.shape[0] == 1:
            desc = desc[0]
        if p["descriptor_layout"] == "channels_first":
            desc = desc.T
        if points.ndim != 2 or points.shape[1] != 2 or len(points) > p["maximum_features"]:
            raise ValidationError("Expected bounded XY rows", "maximum_features")
        count = len(points)
        if desc.shape != (count, 256):
            raise ValidationError("SuperPoint descriptors must have one 256-vector per XY row")
        scores = head(p["scores_output"]) if p["scores_output"] else np.ones(count)
        if scores.shape == (1, count):
            scores = scores[0]
        if scores.shape != (count,) or np.any(scores < 0) or np.any(scores > 1):
            raise ValidationError("Expected one probability per feature")
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValidationError("Zero descriptors cannot represent local features")
        if p["normalize_descriptors"]:
            desc = desc / norms
        if p["coordinate_units"] == "normalized":
            points *= network
        if p["coordinate_origin"] == "pixel_centers":
            points += 0.5
        points = points * [reverse[0, 0], reverse[1, 1]] + [
            reverse[0, 2] - 0.5,
            reverse[1, 2] - 0.5,
        ]
        selection = selection_mask(inputs, image)
        keep = [
            i
            for i, xy in enumerate(points)
            if scores[i] >= p["minimum_score"]
            and 0 <= xy[0] <= w - 1
            and 0 <= xy[1] <= h - 1
            and contains(selection, xy)
        ]
        context = self.input_payloads.get("image", {})
        data = {
            "model": MODEL,
            "image_size": [w, h],
            "points": points[keep].tolist(),
            "descriptors": desc[keep].tolist(),
            "scores": scores[keep].tolist(),
            "coordinate_context": copy.deepcopy(context),
            "raw_indices": keep,
        }
        feature_arrays(data)
        objects = (
            [
                graphic_object("marker", [xy], (60, 210, 250))
                for xy in data["points"][: p["graphics_maximum_features"]]
            ]
            if self.generate_graphics
            else []
        )
        outputs = {
            "image": image,
            "features": data,
            "indices": keep,
            "count": len(keep),
            "report": {
                "model": "local_feature_pack_v1",
                "raw_count": count,
                "retained_count": len(keep),
                "context_checks": checked,
                "detector_executed": False,
                "graphics_truncated": self.generate_graphics
                and len(keep) > p["graphics_maximum_features"],
            },
        }
        passed = len(keep) >= p["minimum_count"]
        return Result(
            outputs,
            payloads={k: copy.deepcopy(context) for k in outputs},
            passed=passed,
            message=f"{len(keep)}/{count} local features retained",
            fail_reasons=[] if passed else ["local_features.count"],
            drawings={"Local features": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
        )


class SuperPointLightGlueMatch(LightGlueMatch):
    type_id, title = "ml.lightglue_superpoint", "LightGlue SuperPoint feature matcher"
    descriptor_dimension, add_scale_orientation, feature_family = 256, False, "SuperPoint"
    preview_safe = process_inputs_readonly = True
    inputs = (Port("image"), Port("query_features", "data"), Port("reference_features", "data"))
    outputs = (*LightGlueMatch.outputs, Port("image"))
    defaults = {k: v for k, v in LightGlueMatch.defaults.items() if k not in SIFT_DEFAULTS}
    defaults = {**defaults, "weights_sha256": ""}
    defaults = {**defaults, "maximum_features": 1024, "maximum_image_pixels": 4_194_304}
    project_path_parameters = ("weights_path",)
    parameter_choices = LightGlueMatch.parameter_choices
    parameter_ranges = {
        "layers": NumericRange(1, 9, whole=True),
        "filter_threshold": NumericRange(0, 1),
        "minimum_matches": NumericRange(4, 4096, whole=True),
        "minimum_inliers": NumericRange(4, 4096, whole=True),
        "minimum_inlier_ratio": NumericRange(0, 1),
        "ransac_threshold": NumericRange(0.0001, 1e6),
        "ransac_confidence": NumericRange(0.0001, 0.999999),
        "ransac_iterations": NumericRange(1, 100000, whole=True),
        "maximum_features": NumericRange(4, 4096, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        **LightGlueMatch.parameter_help,
        "weights_path": "Local superpoint_lightglue.pth matcher; no automatic download.",
        "weights_sha256": "Optional 64-character SHA-256 for the local matcher checkpoint.",
        "device": "CPU, explicitly requested CUDA, or Auto; no unavailable CUDA fallback.",
        "filter_threshold": "Learned match confidence threshold in [0,1].",
        "flash": "Use the upstream attention acceleration when available.",
        "mixed_precision": "CUDA mixed precision; CPU requires this disabled.",
        "minimum_matches": "Minimum learned correspondences for a passing geometric fit.",
        "minimum_inliers": "Minimum geometric inliers for a passing fit.",
        "minimum_inlier_ratio": "Required inlier fraction among learned matches.",
        "geometry": "Native robust homography method; one planar registration only.",
        "ransac_threshold": "Geometric reprojection threshold in query source pixels.",
        "ransac_confidence": "Robust-fit confidence, strictly below one.",
        "ransac_iterations": "Maximum native robust-fit iterations.",
        "maximum_features": "Bound each descriptor set before learned attention; no truncation.",
        "maximum_image_pixels": "Bound current query image pixels before diagnostics.",
    }
    input_help = {
        "image": "Current query image for source coordinates, unchanged output and diagnostics.",
        "query_features": "Current-image packed SuperPoint descriptors, not SIFT/ORB.",
        "reference_features": "Packed reference descriptors; may use another acquisition ticket.",
    }

    def reset_runtime(self):
        super().reset_runtime()
        self._digest_key = None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["weights_sha256"], str) or not re.fullmatch(
            r"(?:[0-9a-fA-F]{64})?", p["weights_sha256"]
        ):
            raise ValidationError("Use empty text or a 64-character SHA-256", "weights_sha256")
        for k in ("depth_confidence", "width_confidence"):
            if type(p[k]) not in (float, int) or not (p[k] == -1 or 0 <= p[k] <= 1):
                raise ValidationError("Use -1 or confidence in [0,1]", k)
        for k in ("flash", "mixed_precision"):
            if type(p[k]) is not bool:
                raise ValidationError("Use a checkbox", k)
        if p["mixed_precision"] and p["device"] == "cpu":
            raise ValidationError("Mixed precision requires CUDA", "mixed_precision", "device")

    def load_matcher(self):
        path = local_path(self, self.parameters["weights_path"])
        expected = self.parameters["weights_sha256"].lower()
        key = (*file_key(path), expected)
        if expected and key != self._digest_key:
            if digest_file(path) != expected:
                raise ValidationError("Matcher checkpoint SHA-256 mismatch", "weights_sha256")
            self._digest_key = key
        super().load_matcher()

    def feature_arrays(self, features):
        return feature_arrays(features)

    def extract_features(self, inputs):
        query, reference = inputs["query_features"], inputs["reference_features"]
        image_value(inputs["image"])
        if np.prod(inputs["image"].shape[:2]) > self.parameters["maximum_image_pixels"]:
            raise ValidationError("Query exceeds image budget", "maximum_image_pixels")
        for name, value in (("query_features", query), ("reference_features", reference)):
            arrays = self.feature_arrays(value)
            if len(arrays["keypoints"]) > self.parameters["maximum_features"]:
                raise ValidationError(
                    "Features exceed learned attention row budget", "maximum_features"
                )
            body, wire = value.get("coordinate_context", {}), self.input_payloads.get(name, {})
            validate_payload(body)
            check_spatial(body, value["image_size"])
            contexts = [body, wire] + (
                [self.input_payloads.get("image", {})] if name == "query_features" else []
            )
            require_same_frame(
                [ValuePacket(None, p) for p in contexts], message="Local-feature frame conflict"
            )
            for field in ("ticket", "recipe_ref"):
                known = [p[field] for p in contexts if field in p]
                keys = [ticket_key(v) for v in known] if field == "ticket" else known
                if keys and any(v != keys[0] for v in keys[1:]):
                    raise ValidationError("Local features have conflicting " + field)
        if query["image_size"] != list(inputs["image"].shape[1::-1]):
            raise ValidationError("Query feature grid differs from the current image")
        return query

    def process(self, inputs):
        result = super().process(inputs)
        result.outputs["image"] = inputs["image"]
        objects = []
        if self.generate_graphics:
            matches = result.outputs["matches"]
            objects = [
                graphic_object("marker", [xy], (70, 220, 100) if good else (40, 70, 230))
                for xy, good in zip(matches["query_points"], matches["inliers"])
            ]
        else:
            result.outputs["overlay"] = inputs["image"]
        result.drawings = (
            {"LightGlue matches": GraphicDrawing(tuple(objects))} if self.generate_graphics else {}
        )
        return result


LOCAL_FEATURE_BLOCKS = (PackSuperPointFeatures, SuperPointLightGlueMatch)
