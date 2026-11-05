"""OpenCV background subtractors with bounded, editor-owned sequence state."""

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, image_format
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


NATIVE_BACKGROUND_BLOCKS = (MOG2Sequence, KNNBackgroundSequence)
