"""OpenCV background subtractors with bounded, editor-owned sequence state."""

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import finite, image_value, integer
from .selection import selection_mask
from .sequence import SEQUENCE_DEFAULTS, SEQUENCE_HELP, NativeSequenceBlock, SequenceStep

BACKGROUND_DEFAULTS = {
    **SEQUENCE_DEFAULTS,
    "warmup": 20,
    "max_bytes_mb": 256.0,
    "history": 500,
    "learning_rate": -1.0,
    "detect_shadows": True,
    "shadow_threshold": 0.5,
    "shadow_value": 127,
    "include_shadows": False,
}
BACKGROUND_HELP = {
    **SEQUENCE_HELP,
    "learning_rate": "-1 selects OpenCV's automatic rate. Manual rates are in [1e-8, 1).",
    "shadow_value": "Value in the raw mask for detected shadows. The shadows port is binary.",
    "include_shadows": "Include shadows in the main binary foreground mask.",
    "max_bytes_mb": "Retained CPU model budget. Processing images require additional memory.",
}


class NativeBackgroundSequence(NativeSequenceBlock):
    """An opaque OpenCV model can move between editor runs without being serialized."""

    inputs = (
        *NativeSequenceBlock.inputs,
        Port("known_foreground", required=False, visual_type="mask"),
    )
    data_outputs = (
        Port("mask"),
        Port("raw_mask", visual_type="mask"),
        Port("shadows", visual_type="mask"),
        Port("background"),
        Port("foreground"),
    )
    defaults = BACKGROUND_DEFAULTS
    parameter_help = BACKGROUND_HELP
    image_depths = (np.uint8,)

    def learning_rate(self, inputs):
        rate = finite(self.parameters["learning_rate"], "Learning rate")
        if rate != -1 and not 1e-8 <= rate < 1:
            raise ValueError("Learning rate must be -1 or in [1e-8, 1)")
        return rate

    def new_model(self):
        raise NotImplementedError

    def step(self, inputs, state, mask):
        source, p = inputs["image"], self.parameters
        image = image_value(source, self.image_depths, canonical=True)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype == np.float32 and ((image < 0) | (image > 255)).any():
            raise ValueError("Floating-point MOG2 frames must use the [0, 255] intensity scale")
        rate = self.learning_rate(inputs)
        known = selection_mask({"image": source, "mask": inputs.get("known_foreground")}, image)
        if "native" not in state:
            if rate == 0:
                return SequenceStep(
                    dict.fromkeys(self.flow_primary),
                    ready=False,
                    learned=False,
                    details={"reason": "learning_disabled"},
                )
            self.reserve_native_state(state, self.estimate_state_bytes(source))
            model = self.new_model()
            shadow_threshold = finite(p["shadow_threshold"], "Shadow threshold", 0)
            if shadow_threshold > 1:
                raise ValueError("Shadow threshold must be at most 1")
            model.setShadowThreshold(shadow_threshold)
            model.setShadowValue(integer(p["shadow_value"], "Shadow value", 1, 254))
            state["native"] = model
        model = state["native"]
        if known is None:
            raw = model.apply(image, learningRate=rate)
        else:
            raw = model.apply(image, knownForegroundMask=known, learningRate=rate)
        if mask is not None:
            raw[mask == 0] = 0
        shadows = raw == p["shadow_value"] if p["detect_shadows"] else np.zeros(raw.shape, bool)
        visible = (raw == 255) | (shadows if p["include_shadows"] else False)
        foreground = np.asarray(source).copy()
        foreground[~visible] = 0
        background = model.getBackgroundImage()
        encoding = "gray" if image.ndim == 2 else "bgr"
        binary = ImageArray(visible.astype(np.uint8) * 255, "gray")
        return SequenceStep(
            {
                "mask": binary,
                "raw_mask": ImageArray(raw, "gray"),
                "shadows": ImageArray(shadows.astype(np.uint8) * 255, "gray"),
                "background": ImageArray(background, encoding) if background is not None else None,
                "foreground": ImageArray(foreground, image_format(source)),
            },
            ready=background is not None,
            learned=rate != 0 and (known is None or not known.all()),
            details={
                "foreground_fraction": float(visible.sum() / max(1, cv2.countNonZero(mask)))
                if mask is not None
                else float(visible.mean()),
                "native_model_bytes": state["native_bytes"],
            },
            graphics={"background subtraction": binary},
        )


class MOG2Sequence(NativeBackgroundSequence):
    type_id, title = "sequence.mog2", "MOG2 background subtraction"
    inputs = (*NativeBackgroundSequence.inputs, Port("learn", "data", required=False))
    image_depths = (np.uint8, np.float32)
    defaults = {
        **BACKGROUND_DEFAULTS,
        "warmup": 5,
        "variance_threshold": 16.0,
        "new_mixture_threshold": 9.0,
        "mixtures": 5,
        "background_ratio": 0.9,
        "initial_variance": 15.0,
        "minimum_variance": 4.0,
        "maximum_variance": 75.0,
        "complexity_reduction": 0.05,
    }
    parameter_help = {
        **BACKGROUND_HELP,
        "variance_threshold": "Squared Mahalanobis distance threshold for background matching.",
        "new_mixture_threshold": "Squared distance threshold for generating a new mixture.",
        "complexity_reduction": "Evidence threshold for retaining a mixture. Zero disables it.",
    }

    def estimate_state_bytes(self, image):
        mixtures = integer(self.parameters["mixtures"], "Mixtures", 1, 32)
        channels = 1 if image.ndim == 2 else 3
        return image.shape[0] * image.shape[1] * (4 * mixtures * (channels + 2) + 1) + 4096

    def learning_rate(self, inputs):
        rate = super().learning_rate(inputs)
        learn = inputs.get("learn", True)
        if type(learn) is not bool:
            raise ValueError("Learn input must be boolean")
        return rate if learn else 0.0

    def new_model(self):
        p = self.parameters
        history = integer(p["history"], "History", 1, 1_000_000)
        threshold = finite(p["variance_threshold"], "Variance threshold", 1e-6)
        minimum, initial, maximum = (
            finite(p[k], k, 1e-6)
            for k in ("minimum_variance", "initial_variance", "maximum_variance")
        )
        ratio = finite(p["background_ratio"], "Background ratio", 1e-6)
        reduction = finite(p["complexity_reduction"], "Complexity reduction", 0)
        if not minimum <= initial <= maximum or ratio > 1 or reduction > 1:
            raise ValueError("Invalid mixture variance, background ratio or complexity reduction")
        model = cv2.createBackgroundSubtractorMOG2(history, threshold, p["detect_shadows"])
        model.setNMixtures(integer(p["mixtures"], "Mixtures", 1, 32))
        model.setVarThresholdGen(finite(p["new_mixture_threshold"], "New mixture threshold", 1e-6))
        model.setBackgroundRatio(ratio)
        model.setVarInit(initial)
        model.setVarMin(minimum)
        model.setVarMax(maximum)
        model.setComplexityReductionThreshold(reduction)
        return model


class KNNBackgroundSequence(NativeBackgroundSequence):
    type_id, title = "sequence.knn_background", "KNN background subtraction"
    defaults = {
        **BACKGROUND_DEFAULTS,
        "squared_distance_threshold": 400.0,
        "samples": 7,
        "neighbors": 2,
    }
    parameter_help = {
        **BACKGROUND_HELP,
        "squared_distance_threshold": "Squared distance used to decide if a sample matches.",
        "neighbors": "Minimum matching samples required to classify a pixel as background.",
    }

    def estimate_state_bytes(self, image):
        samples = integer(self.parameters["samples"], "Samples", 1, 64)
        channels = 1 if image.ndim == 2 else 3
        return image.shape[0] * image.shape[1] * (samples * 3 * (channels + 1) + 6) + 4096

    def new_model(self):
        p = self.parameters
        history = integer(p["history"], "History", 1, 1_000_000)
        threshold = finite(p["squared_distance_threshold"], "Squared distance threshold", 1e-6)
        samples = integer(p["samples"], "Samples", 1, 64)
        model = cv2.createBackgroundSubtractorKNN(history, threshold, p["detect_shadows"])
        model.setNSamples(samples)
        model.setkNNSamples(integer(p["neighbors"], "Neighbors", 1, samples * 3))
        return model


SAMPLE_DEFAULTS = {
    **SEQUENCE_DEFAULTS,
    "warmup": 20,
    "max_bytes_mb": 256.0,
    "camera_motion": "none",
    "samples": 20,
    "noise_background": 0.0004,
    "noise_foreground": 0.0008,
}
SAMPLE_RANGES = {
    "samples": NumericRange(2, 1023, whole=True),
    "noise_background": NumericRange(0, 0.499999),
    "noise_foreground": NumericRange(0, 0.499999),
}
SAMPLE_HELP = {
    **SEQUENCE_HELP,
    "camera_motion": "Lucas–Kanade compensates between textured frames on a fixed-size grid.",
    "samples": "Native samples per pixel in each of two retained model buffers.",
    "noise_background": "Remove small foreground components below this fraction of frame area.",
    "noise_foreground": "Fill small background components below this fraction of frame area.",
    "max_bytes_mb": "Estimated retained CPU model budget. Processing uses extra buffers.",
}


class SampleBackgroundSequence(NativeSequenceBlock):
    """Shared lifecycle for the contrib sample models and their binary foreground outputs."""

    subtype = "Foreground segmentation"
    defaults = SAMPLE_DEFAULTS
    parameter_choices = {
        **NativeSequenceBlock.parameter_choices,
        "camera_motion": ("none", "lucas_kanade"),
    }
    parameter_ranges = SAMPLE_RANGES
    parameter_help = SAMPLE_HELP
    input_help = {
        "image": "8-bit or normalized Float32 grayscale or color frames.",
        "mask": "Restrict foreground outputs. Model updates still use the full frame.",
        "region": "Restrict foreground outputs. Model updates still use the full frame.",
    }
    data_outputs = (
        Port("mask", visual_type="mask"),
        Port("foreground"),
        Port("background"),
    )
    process_inputs_readonly = True

    def estimate_state_bytes(self, image):
        samples = integer(self.parameters["samples"], "Samples", 2, 1023)
        # Native retains two sample buffers, even with motion compensation off.
        return image.shape[0] * image.shape[1] * (2 * samples * self.sample_bytes + 32) + 4096

    def native_options(self):
        p = self.parameters
        bgsegm = cv2.bgsegm
        motion = (
            bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_LK
            if p["camera_motion"] == "lucas_kanade"
            else bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE
        )
        return {
            "mc": motion,
            "nSamples": int(p["samples"]),
            "noiseRemovalThresholdFacBG": float(p["noise_background"]),
            "noiseRemovalThresholdFacFG": float(p["noise_foreground"]),
        }

    def new_model(self):
        raise NotImplementedError

    def step(self, inputs, state, mask):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.float32), canonical=True)
        minimum = 25 if self.parameters["camera_motion"] == "lucas_kanade" else 2
        if min(image.shape[:2]) < minimum:
            raise ValidationError(f"Frame must be at least {minimum} pixels on each axis")
        if image.dtype == np.float32 and ((image < 0) | (image > 1)).any():
            raise ValidationError("Float32 frames must use normalized [0, 1] intensities")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if "native" not in state:
            self.reserve_native_state(state, self.estimate_state_bytes(source))
            state["native"] = self.new_model()
        model = state["native"]
        rate = float(self.parameters.get("color_learning_rate", -1))
        binary = model.apply(image, learningRate=rate)
        if mask is not None:
            binary[mask == 0] = 0
        foreground = np.asarray(source).copy()
        foreground[binary == 0] = 0
        background = model.getBackgroundImage()
        if image.ndim == 2:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        return SequenceStep(
            {
                "mask": ImageArray(binary, "gray"),
                "foreground": ImageArray(foreground, image_format(source)),
                "background": ImageArray(background, "gray" if image.ndim == 2 else "bgr"),
            },
            details={
                "foreground_fraction": cv2.countNonZero(binary)
                / max(1, binary.size if mask is None else cv2.countNonZero(mask)),
                "native_model_bytes": state["native_bytes"],
            },
        )


class GSOCBackgroundSequence(SampleBackgroundSequence):
    type_id, title = "sequence.gsoc_background", "GSoC background subtraction"
    sample_bytes = 32
    defaults = {
        **SAMPLE_DEFAULTS,
        "replace_rate": 0.003,
        "propagation_rate": 0.01,
        "hits_threshold": 32,
        "alpha": 0.01,
        "beta": 0.0022,
        "blink_decay": 0.1,
        "blink_multiplier": 0.1,
        "color_learning_rate": 0.1,
    }
    parameter_ranges = {
        **SAMPLE_RANGES,
        "replace_rate": NumericRange(0, 1),
        "propagation_rate": NumericRange(0, 1),
        "hits_threshold": NumericRange(0, 2**31 - 1, whole=True),
        "alpha": NumericRange(0, 100),
        "beta": NumericRange(0, 3),
        "blink_decay": NumericRange(1e-6, 0.999999),
        "blink_multiplier": NumericRange(0, 1),
        "color_learning_rate": NumericRange(0, 1),
    }
    parameter_help = {
        **SAMPLE_HELP,
        "replace_rate": "Chance of adding an unmatched color to the sample bank.",
        "propagation_rate": "Chance of copying a trusted sample to adjacent pixels.",
        "hits_threshold": "Matches needed before a sample is eligible for propagation.",
        "alpha": "Weight on the recent squared color error in the matching threshold.",
        "beta": "Additive matching threshold in normalized squared color units.",
        "blink_decay": "Decay factor for the foreground toggling estimate.",
        "blink_multiplier": "Strength of adaptation where pixels repeatedly toggle.",
        "color_learning_rate": "Blend rate for matched colors and error. Other updates continue.",
    }

    def new_model(self):
        p = self.parameters
        return cv2.bgsegm.createBackgroundSubtractorGSOC(
            **self.native_options(),
            replaceRate=float(p["replace_rate"]),
            propagationRate=float(p["propagation_rate"]),
            hitsThreshold=int(p["hits_threshold"]),
            alpha=float(p["alpha"]),
            beta=float(p["beta"]),
            blinkingSupressionDecay=float(p["blink_decay"]),
            blinkingSupressionMultiplier=float(p["blink_multiplier"]),
        )


class LSBPBackgroundSequence(SampleBackgroundSequence):
    type_id, title = "sequence.lsbp_background", "LSBP background subtraction"
    sample_bytes = 20
    defaults = {
        **SAMPLE_DEFAULTS,
        "descriptor_radius": 16,
        "update_minimum": 2.0,
        "update_maximum": 32.0,
        "update_increase": 1.0,
        "update_decrease": 0.05,
        "distance_scale": 10.0,
        "distance_adaptation": 0.005,
        "descriptor_threshold": 8,
        "minimum_matches": 2,
    }
    parameter_ranges = {
        **SAMPLE_RANGES,
        "descriptor_radius": NumericRange(1, 4096, whole=True),
        "update_minimum": NumericRange(1e-6, 1e6),
        "update_maximum": NumericRange(1e-6, 1e6),
        "update_increase": NumericRange(0, 1e6),
        "update_decrease": NumericRange(0, 1e6),
        "distance_scale": NumericRange(0, 1e6),
        "distance_adaptation": NumericRange(0, 1),
        "descriptor_threshold": NumericRange(1, 33, whole=True),
        "minimum_matches": NumericRange(1, 1023, whole=True),
    }
    parameter_help = {
        **SAMPLE_HELP,
        "descriptor_radius": "Pixel radius of the local texture samples.",
        "update_minimum": "Lower bound for the reciprocal update probability.",
        "update_maximum": "Upper bound for the reciprocal update probability.",
        "update_increase": "Step toward slower updates on foreground pixels.",
        "update_decrease": "Step toward faster updates on background pixels.",
        "distance_scale": "Scale the mean sample error to target the adaptive color threshold.",
        "distance_adaptation": "Fractional adjustment of the adaptive color threshold.",
        "descriptor_threshold": "Exclusive Hamming distance limit for the 32-bit texture codes.",
        "minimum_matches": "Minimum matching samples required for a background pixel.",
    }

    def new_model(self):
        p = self.parameters
        if np.float32(p["update_minimum"]) >= np.float32(p["update_maximum"]):
            raise ValidationError(
                "Update minimum must be below update maximum", "update_minimum", "update_maximum"
            )
        if p["minimum_matches"] > p["samples"]:
            raise ValidationError("Minimum matches cannot exceed samples", "minimum_matches")
        return cv2.bgsegm.createBackgroundSubtractorLSBP(
            **self.native_options(),
            LSBPRadius=int(p["descriptor_radius"]),
            Tlower=float(p["update_minimum"]),
            Tupper=float(p["update_maximum"]),
            Tinc=float(p["update_increase"]),
            Tdec=float(p["update_decrease"]),
            Rscale=float(p["distance_scale"]),
            Rincdec=float(p["distance_adaptation"]),
            LSBPthreshold=int(p["descriptor_threshold"]),
            minCount=int(p["minimum_matches"]),
        )


NATIVE_BACKGROUND_BLOCKS = (
    MOG2Sequence,
    KNNBackgroundSequence,
    GSOCBackgroundSequence,
    LSBPBackgroundSequence,
)
