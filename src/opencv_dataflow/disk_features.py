"""Local-weight DISK extraction and matched LightGlue, preserving source image geometry."""

import copy
import re

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import opencv_image
from .local_features import SuperPointLightGlueMatch, feature_arrays
from .ml_common import digest_file, file_key, local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import image_value
from .selection import contains, selection_mask

MODEL = "lightglue_disk_v1"


def disk_arrays(value):
    return feature_arrays(value, model=MODEL, dimension=128, probability_scores=False)


def prepare_image(inputs, p, *, multiple=16, family="DISK"):
    source = inputs["image"]
    # Check geometry before whole-image conversion or allocation.
    h, w = source.shape[:2]
    if not 1 <= min(h, w) or max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
        raise ValidationError(f"Source exceeds {family} image budget", "maximum_image_pixels")
    image_value(source, (np.uint8, np.uint16, np.float32))
    pixels = opencv_image(source).astype(np.float32)
    if source.dtype != np.float32:
        pixels /= np.iinfo(source.dtype).max
    if np.any(pixels < 0) or np.any(pixels > 1):
        raise ValidationError(f"{family} floating-point RGB/gray/alpha samples must be in [0,1]")
    selection = selection_mask(inputs, source)
    if pixels.ndim == 2:
        rgb = np.repeat(pixels[..., None], 3, axis=2)
    else:
        if pixels.shape[2] == 4 and p["alpha_policy"] == "exclude":
            alpha = (pixels[..., 3] > 0).astype(np.uint8) * 255
            selection = alpha if selection is None else cv2.bitwise_and(selection, alpha)
        rgb = pixels[..., :3][..., ::-1]
    factor = min(1.0, p["resize_long_edge"] / max(h, w)) if p["resize_long_edge"] else 1.0
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    padded = (
        [((n + multiple - 1) // multiple) * multiple for n in (nw, nh)]
        if p["pad_if_not_divisible"]
        else [nw, nh]
    )
    if min(padded) < 32 or any(n % multiple for n in padded):
        raise ValidationError(
            f"{family} needs network axes >=32 and multiples of {multiple}; use padding"
        )
    if padded[0] * padded[1] > p["maximum_network_pixels"]:
        raise ValidationError(
            f"Padded {family} network exceeds pixel budget", "maximum_network_pixels"
        )
    if (nw, nh) != (w, h):
        rgb = cv2.resize(
            rgb,
            (nw, nh),
            interpolation={
                "area": cv2.INTER_AREA,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "nearest": cv2.INTER_NEAREST,
            }[p["resize_interpolation"]],
        )
        # Cubic interpolation can overshoot the declared normalized input range.
        rgb = np.clip(rgb, 0, 1)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), selection, [nw / w, nh / h], padded


class DISKFeatures(Block):
    type_id, title = "ml.disk_features", "DISK learned features"
    category, block_type, subtype = "ML", "ML", "Feature matching"
    preview_safe = process_inputs_readonly = True
    feature_family, feature_model, score_semantics = "DISK", MODEL, "raw_not_probability"
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
        "weights_path": "",
        "weights_sha256": "",
        "device": "cpu",
        "max_features": 1024,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
        "resize_long_edge": 0,
        "resize_interpolation": "area",
        "alpha_policy": "exclude",
        "minimum_count": 4,
        "maximum_image_pixels": 4_194_304,
        "maximum_network_pixels": 524_288,
        "graphics_maximum_features": 256,
    }
    project_path_parameters = ("weights_path",)
    parameter_choices = {
        "device": ("cpu", "cuda", "auto"),
        "resize_interpolation": ("area", "linear", "cubic", "nearest"),
        "alpha_policy": ("exclude", "ignore"),
    }
    parameter_ranges = {
        "max_features": NumericRange(1, 4096, whole=True),
        "nms_window_size": NumericRange(1, 31, whole=True, odd=True),
        "detection_threshold": NumericRange(-1e6, 1e6),
        "resize_long_edge": NumericRange(32, 4096, whole=True, zero_allowed=True),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_network_pixels": NumericRange(1024, 2_097_152, whole=True),
        "graphics_maximum_features": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "weights_path": "Local DISK depth/epipolar checkpoint; no automatic download.",
        "weights_sha256": "Optional 64-character SHA-256; actual model digest is always reported.",
        "device": "CPU, explicit CUDA, or Auto; unavailable explicitly requested CUDA rejects.",
        "max_features": "Native whole-image top-N maximum before source/ROI/alpha filtering.",
        "nms_window_size": "Odd native local-maximum window in resized network pixels.",
        "detection_threshold": "Native raw score must be strictly greater; not a probability.",
        "pad_if_not_divisible": "Pad right/bottom to multiples of 16; outputs exclude padding.",
        "resize_long_edge": "Zero keeps source size; otherwise only downscale the longest edge.",
        "resize_interpolation": "OpenCV resize interpolation; cubic overshoot is clipped to [0,1].",
        "alpha_policy": "Exclude zero-alpha centers, or ignore alpha; CNN still sees RGB context.",
        "minimum_count": "Fewest retained features for a pass; zero permits empty capture.",
        "maximum_image_pixels": "Bound source pixels before allocation; axes are at most 16384.",
        "maximum_network_pixels": "Bound padded network pixels before dense descriptor inference.",
        "graphics_maximum_features": "Cap optional markers without truncating numeric output.",
    }
    input_help = {
        "image": "uint8/uint16 or normalized float32 gray/color; original pixels are retained.",
        "mask": "Source-grid center filter after native top-N; missing wired events gate work.",
        "region": "Source-grid shape intersected with Mask; descriptors see surrounding pixels.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model = self._key = self._torch = self._device = self._digest = self._version = None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError(f"Use typed numeric {self.feature_family} settings", name)
        if "pad_if_not_divisible" in p and type(p["pad_if_not_divisible"]) is not bool:
            raise ValidationError("Padding is a checkbox", "pad_if_not_divisible")
        if not isinstance(p["weights_path"], str):
            raise ValidationError("Use a local model path", "weights_path")
        if not isinstance(p["weights_sha256"], str) or not re.fullmatch(
            r"(?:[0-9a-fA-F]{64})?", p["weights_sha256"]
        ):
            raise ValidationError("Use empty text or a 64-character SHA-256", "weights_sha256")
        if p["minimum_count"] > p["max_features"]:
            raise ValidationError(
                "Minimum count exceeds native top-N", "minimum_count", "max_features"
            )

    def model_settings(self):
        return ()

    def build_model(self, kornia, state):
        if not isinstance(state, dict) or not isinstance(state.get("extractor"), dict):
            raise ValidationError("Use a DISK checkpoint with an extractor state dictionary")
        model = kornia.feature.DISK(desc_dim=128)
        model.load_state_dict(state["extractor"], strict=True)
        return model

    def working_image(self, inputs):
        return prepare_image(inputs, self.parameters)

    def native_arrays(self, rgb):
        p = self.parameters
        data = self._model(
            self._torch.from_numpy(rgb)[None].to(self._device),
            n=p["max_features"],
            window_size=p["nms_window_size"],
            score_threshold=p["detection_threshold"],
            pad_if_not_divisible=p["pad_if_not_divisible"],
        )[0]
        return data.keypoints, data.descriptors, data.detection_scores

    def validate_features(self, data):
        return disk_arrays(data)

    def load_model(self):
        p = self.parameters
        path = local_path(self, p["weights_path"])
        identity = file_key(path)
        if identity[1] > 128 * 1024**2:
            raise ValidationError(f"{self.feature_family} checkpoint exceeds 128 MiB")
        key = (*identity, p["weights_sha256"].lower(), p["device"], *self.model_settings())
        if key == self._key:
            return
        digest = digest_file(path)
        if p["weights_sha256"] and digest != p["weights_sha256"].lower():
            raise ValidationError(
                f"{self.feature_family} checkpoint SHA-256 mismatch", "weights_sha256"
            )
        try:
            import kornia
            import torch
        except ImportError as exc:
            raise RuntimeError("Install optional PyTorch and Kornia; see docs/ml.md") from exc
        device = "cuda" if p["device"] == "auto" and torch.cuda.is_available() else p["device"]
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValidationError("Requested CUDA is unavailable", "device")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = self.build_model(kornia, state)
        self._model = model.eval().to(device)
        self._torch, self._device, self._key = torch, device, key
        self._digest, self._version = digest, kornia.__version__

    def capture(self, inputs):
        p, source = self.parameters, inputs["image"]
        rgb, selection, scale, padded = self.working_image(inputs)
        self.load_model()
        with self._torch.inference_mode():
            points, descriptors, confidence = self.native_arrays(rgb)
        xy = points.detach().cpu().numpy().astype(np.float64)
        desc = descriptors.detach().cpu().numpy()
        scores = confidence.detach().cpu().numpy()
        xy = (xy + 0.5) / scale - 0.5
        keep = [
            i
            for i, point in enumerate(xy)
            if contains(selection, point)
            and ("minimum_score" not in p or scores[i] >= p["minimum_score"])
        ]
        context = metadata(source)
        record = {
            "model": self.feature_model,
            "image_size": [source.shape[1], source.shape[0]],
            "points": xy[keep].tolist(),
            "descriptors": desc[keep].tolist(),
            "scores": scores[keep].tolist(),
            "coordinate_context": copy.deepcopy(context),
            "raw_indices": keep,
            "extractor": {
                "family": self.feature_family,
                "weights_sha256": self._digest,
                "backend": "kornia",
                "version": self._version,
                "device": self._device,
            },
        }
        self.validate_features(record)
        report = {
            "model": self.feature_family.lower() + "_extraction_v1",
            "detector_executed": True,
            "raw_count": len(xy),
            "retained_count": len(keep),
            "network_size": list(rgb.shape[2:0:-1]),
            "padded_size": padded,
            "source_to_network_scale": scale,
            "coordinate_origin": "source_pixel_centers",
            "selection_policy": "post_top_n_center_filter",
            "detector_scores": self.score_semantics,
            "weights_sha256": self._digest,
            "device": self._device,
            "kornia": self._version,
        }
        return record, report

    def train(self, inputs):
        """Capture portable reference features; this never fine-tunes network weights."""
        self.validate_parameters()
        return self.capture(inputs)[0]

    def process(self, inputs):
        record, report = self.capture(inputs)
        count = len(record["points"])
        limit = self.parameters["graphics_maximum_features"]
        objects = (
            [graphic_object("marker", [xy], (60, 210, 250)) for xy in record["points"][:limit]]
            if self.generate_graphics
            else []
        )
        report["graphics_truncated"] = self.generate_graphics and count > limit
        outputs = {"image": inputs["image"], "features": record, "count": count, "report": report}
        passed = count >= self.parameters["minimum_count"]
        return Result(
            outputs,
            payloads={k: copy.deepcopy(metadata(inputs["image"])) for k in outputs},
            passed=passed,
            message=f"{self.feature_family}: {count}/{report['raw_count']} features retained",
            fail_reasons=[] if passed else [self.feature_family.lower() + ".features"],
            drawings={self.feature_family + " features": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
        )


class DISKLightGlueMatch(SuperPointLightGlueMatch):
    type_id, title = "ml.lightglue_disk", "LightGlue DISK feature matcher"
    descriptor_dimension, add_scale_orientation, feature_family = 128, False, "DISK"
    parameter_help = {
        **SuperPointLightGlueMatch.parameter_help,
        "weights_path": "Local disk_lightglue.pth matcher, not SIFT or SuperPoint weights.",
    }
    input_help = {
        **SuperPointLightGlueMatch.input_help,
        "query_features": "Current-image DISK features with 128-value learned descriptors.",
        "reference_features": "Captured DISK reference; may use another acquisition ticket.",
    }

    def feature_arrays(self, features):
        return disk_arrays(features)


DISK_BLOCKS = (DISKFeatures, DISKLightGlueMatch)
