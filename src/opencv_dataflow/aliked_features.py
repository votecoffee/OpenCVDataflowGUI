"""Local-weight ALIKED extraction using the shared image, feature and matcher contracts."""

from .disk_features import DISKFeatures, prepare_image
from .local_features import SuperPointLightGlueMatch, feature_arrays
from .parameter_validation import NumericRange, ValidationError

MODEL = "lightglue_aliked_v1"


def aliked_arrays(data):
    return feature_arrays(data, model=MODEL, dimension=128)


class ALIKEDFeatures(DISKFeatures):
    type_id, title = "ml.aliked_features", "ALIKED learned features"
    feature_family, feature_model, score_semantics = "ALIKED", MODEL, "probabilities"
    defaults = {
        k: v
        for k, v in DISKFeatures.defaults.items()
        if k not in ("pad_if_not_divisible", "nms_window_size")
    }
    defaults = {
        **defaults,
        "model_name": "aliked-n16",
        "detection_mode": "top_k",
        "detection_threshold": 0.2,
        "nms_radius": 2,
        "minimum_score": 0.0,
    }
    parameter_choices = {
        **DISKFeatures.parameter_choices,
        "model_name": ("aliked-n16", "aliked-n16rot", "aliked-n32"),
        "detection_mode": ("top_k", "threshold"),
    }
    parameter_ranges = {
        k: v for k, v in DISKFeatures.parameter_ranges.items() if k != "nms_window_size"
    }
    parameter_ranges = {
        **parameter_ranges,
        "detection_threshold": NumericRange(0, 1),
        "nms_radius": NumericRange(1, 8, whole=True),
        "minimum_score": NumericRange(0, 1),
    }
    parameter_help = {
        k: v
        for k, v in DISKFeatures.parameter_help.items()
        if k not in ("pad_if_not_divisible", "nms_window_size")
    }
    parameter_help = {
        **parameter_help,
        "weights_path": "Local official 128-descriptor ALIKED checkpoint; no runtime download.",
        "model_name": "128-descriptor architecture; t16's 64 descriptors do not fit this matcher.",
        "detection_mode": "Top-N or threshold mode; the latter can fall back to mean score.",
        "detection_threshold": "Native positive threshold-mode peak cutoff; unused in top-N mode.",
        "nms_radius": "Positive native NMS/soft subpixel radius in resized network pixels.",
        "minimum_score": "Inclusive hard score floor after detection, independent of fallback.",
        "maximum_network_pixels": "Bound replicate-padded network pixels (multiples of 32).",
    }
    input_help = {
        **DISKFeatures.input_help,
        "mask": "Source-grid center filter after native detection; missing events gate work.",
        "region": "Source-grid shape/Mask intersection; CNN still sees surrounding pixels.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["detection_mode"] == "threshold" and p["detection_threshold"] <= 0:
            raise ValidationError(
                "Threshold mode requires a positive cutoff", "detection_mode", "detection_threshold"
            )

    def model_settings(self):
        p = self.parameters
        return (
            p["model_name"],
            p["max_features"],
            p["detection_mode"],
            p["detection_threshold"],
            p["nms_radius"],
        )

    def build_model(self, kornia, state):
        p = self.parameters
        try:
            kind = kornia.feature.ALIKED
        except AttributeError as exc:
            raise RuntimeError(
                "Install a Kornia build with ALIKED support (verified with 0.9.0)"
            ) from exc
        model = kind(
            model_name=p["model_name"],
            max_num_keypoints=p["max_features"],
            detection_threshold=p["detection_threshold"]
            if p["detection_mode"] == "threshold"
            else -1.0,
            nms_radius=p["nms_radius"],
        )
        if not isinstance(state, dict):
            raise ValidationError("Use an ALIKED state dictionary")
        incompatible = model.load_state_dict(state, strict=False)
        # The upstream weights predate this deterministic, non-learned radius grid.
        if set(incompatible.missing_keys) - {"dkd.hw_grid"} or incompatible.unexpected_keys:
            raise ValidationError("ALIKED checkpoint lacks required model parameters")
        return model

    def working_image(self, inputs):
        return prepare_image(
            inputs, {**self.parameters, "pad_if_not_divisible": True}, multiple=32, family="ALIKED"
        )

    def native_arrays(self, rgb):
        data = self._model(self._torch.from_numpy(rgb)[None].to(self._device))[0]
        return data.keypoints, data.descriptors, data.keypoint_scores

    def validate_features(self, data):
        return aliked_arrays(data)

    def capture(self, inputs):
        record, report = super().capture(inputs)
        p = self.parameters
        details = {
            k: p[k]
            for k in (
                "model_name",
                "detection_mode",
                "detection_threshold",
                "nms_radius",
                "max_features",
                "minimum_score",
            )
        }
        record["extractor"]["settings"] = details
        report.update(details)
        report["threshold_fallback_policy"] = "native_per_image_mean"
        report["selection_policy"] = "post_native_cap_and_score_center_filter"
        return record, report


class ALIKEDLightGlueMatch(SuperPointLightGlueMatch):
    type_id, title = "ml.lightglue_aliked", "LightGlue ALIKED feature matcher"
    descriptor_dimension, add_scale_orientation, feature_family = 128, False, "ALIKED"
    parameter_help = {
        **SuperPointLightGlueMatch.parameter_help,
        "weights_path": "Local aliked_lightglue.pth matcher; other descriptor families reject.",
    }
    input_help = {
        **SuperPointLightGlueMatch.input_help,
        "query_features": "Current-image ALIKED features with 128-value descriptors.",
        "reference_features": "Captured ALIKED reference; may use a different acquisition ticket.",
    }

    def feature_arrays(self, features):
        return aliked_arrays(features)


ALIKED_BLOCKS = (ALIKEDFeatures, ALIKEDLightGlueMatch)
