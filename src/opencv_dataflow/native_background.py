"""OpenCV background subtractors with bounded, editor-owned sequence state."""

import cv2
import numpy as np

from .background_reference import identifiers, load_reference, match_reference, reference_path
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
    "reference_path": "",
    "reference_sha256": "",
    "reference_camera_id": "",
    "reference_calibration_id": "",
    "reference_replay_passes": 32,
    "reference_replay_rate": 0.5,
    "maximum_reference_work": 64_000_000,
    "reference_update": "frozen",
}
BACKGROUND_HELP = {
    **SEQUENCE_HELP,
    "learning_rate": "-1 selects OpenCV's automatic rate. Manual rates are in [1e-8, 1).",
    "shadow_value": "Value in the raw mask for detected shadows. The shadows port is binary.",
    "include_shadows": "Include shadows in the main binary foreground mask.",
    "max_bytes_mb": "Retained CPU model budget. Processing images require additional memory.",
    "reference_path": (
        "Optional sealed clean-background JSON. Restored through bounded reference replay."
    ),
    "reference_sha256": (
        "Optional exact file SHA256. Reset forces a fresh read even with unchanged file metadata."
    ),
    "reference_camera_id": "Must match the saved reference's stable camera/setup ID.",
    "reference_calibration_id": (
        "Must match the saved reference's calibration revision, including empty values."
    ),
    "reference_replay_passes": (
        "Clean-reference updates before the first live image; native background probe must pass."
    ),
    "reference_replay_rate": (
        "Native learning rate used only during clean-reference replay, below one."
    ),
    "maximum_reference_work": (
        "Maximum reference pixels times (replay passes + one verification probe)."
    ),
    "reference_update": (
        "Frozen disables live learning; Guarded requires a known-foreground mask on "
        "every activation."
    ),
}


class NativeBackgroundSequence(NativeSequenceBlock):
    """An opaque OpenCV model can move between editor runs without being serialized."""

    inputs = (
        *NativeSequenceBlock.inputs,
        Port("known_foreground", required=False, required_when_connected=True, visual_type="mask"),
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
    parameter_choices = {
        **NativeSequenceBlock.parameter_choices,
        "reference_update": ("frozen", "guarded"),
    }
    parameter_ranges = {
        "reference_replay_passes": NumericRange(1, 1024, whole=True),
        "reference_replay_rate": NumericRange(1e-8, 0.999999),
        "maximum_reference_work": NumericRange(1, 512_000_000, whole=True),
    }
    project_asset_parameters = (*NativeSequenceBlock.project_asset_parameters, "reference_path")
    input_help = {
        **NativeSequenceBlock.input_help,
        "known_foreground": (
            "Current-grid uint8 object mask. With a restored reference, Guarded requires "
            "this mask and replaces marked samples with trusted background before learning."
        ),
    }
    image_depths = (np.uint8,)
    reference_rebuild_frozen = False

    def reset_runtime(self):
        super().reset_runtime()
        self._reference_cache = self._active_reference = None

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["reference_path"]:
            identifiers(
                p["reference_camera_id"],
                p["reference_calibration_id"],
                fields=("reference_camera_id", "reference_calibration_id"),
            )
        digest = p["reference_sha256"]
        if (
            not isinstance(digest, str)
            or digest
            and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest))
        ):
            raise ValidationError(
                "Reference SHA256 must be empty or 64 hex digits", "reference_sha256"
            )

    def extra_context(self, inputs):
        p = self.parameters
        self._active_reference = None
        if not p["reference_path"]:
            self._reference_cache = None
            return None
        path = reference_path(self, "reference_path")
        info = path.stat()
        key = (
            str(path),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            p["reference_sha256"],
        )
        if (
            inputs.get("reset", False)
            or self._reference_cache is None
            or self._reference_cache[0] != key
        ):
            self._reference_cache = (key, load_reference(path, p["reference_sha256"]))
        entry = self._reference_cache[1]
        record, reference, _ = entry
        match_reference(
            record,
            reference,
            inputs["image"],
            self.input_payloads["image"],
            p["reference_camera_id"],
            p["reference_calibration_id"],
        )
        self._active_reference = entry  # One immutable file snapshot per activation.
        return record["id"]

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
        reference_entry = self._active_reference
        if reference_entry is not None and p["reference_update"] == "guarded" and known is None:
            raise ValidationError(
                "Guarded reference learning requires Known foreground on every activation"
            )
        rebuild_query = bool(
            reference_entry is not None
            and self.reference_rebuild_frozen
            and (p["reference_update"] == "frozen" or rate == 0)
        )
        if rebuild_query:
            # This wheel's KNN updates samples even at learningRate=0. Never
            # retain a queried frozen model; release it before reserving again.
            state.pop("native", None)
            state.pop("native_bytes", None)
        if "native" not in state:
            if rate == 0 and reference_entry is None:
                return SequenceStep(
                    dict.fromkeys(self.flow_primary),
                    ready=False,
                    learned=False,
                    details={"reason": "learning_disabled"},
                )
            seed = None
            if reference_entry is not None:
                record, saved, digest = reference_entry
                seed = image_value(saved, self.image_depths, canonical=True)
                passes = int(p["reference_replay_passes"])
                if seed.shape[0] * seed.shape[1] * (passes + 1) > p["maximum_reference_work"]:
                    raise ValidationError(
                        "Reference replay exceeds Maximum reference work", "maximum_reference_work"
                    )
            self.reserve_native_state(
                state, self.estimate_state_bytes(source) + (seed.nbytes if seed is not None else 0)
            )
            model = self.new_model()
            shadow_threshold = finite(p["shadow_threshold"], "Shadow threshold", 0)
            if shadow_threshold > 1:
                raise ValueError("Shadow threshold must be at most 1")
            model.setShadowThreshold(shadow_threshold)
            model.setShadowValue(integer(p["shadow_value"], "Shadow value", 1, 254))
            if seed is not None:
                for _ in range(passes):
                    model.apply(seed, learningRate=float(p["reference_replay_rate"]))
                probe_rate = (
                    float(p["reference_replay_rate"]) if self.reference_rebuild_frozen else 0
                )
                probe = model.apply(seed, learningRate=probe_rate)
                restored = model.getBackgroundImage()
                if probe.any() or restored is None or not np.array_equal(restored, seed):
                    raise ValidationError(
                        "Reference replay is not ready; increase replay passes or review "
                        "reference/model settings",
                        "reference_replay_passes",
                    )
                state.update(seeded=True, reference_id=record["id"], reference_replay_passes=passes)
            state["native"] = model
        model = state["native"]
        if reference_entry is not None and p["reference_update"] == "frozen":
            rate = 0
        protected_background = model.getBackgroundImage() if reference_entry is not None else None
        working = image
        if reference_entry is not None and known is not None and known.any():
            # Some native knownForegroundMask implementations mark FG only
            # after updating samples. Do not feed object values to learning.
            working = image.copy()
            working[known != 0] = protected_background[known != 0]
        query_rate = float(p["reference_replay_rate"]) if rebuild_query else rate
        if known is None:
            raw = model.apply(working, learningRate=query_rate)
        else:
            raw = model.apply(working, knownForegroundMask=known, learningRate=query_rate)
        if mask is not None:
            raw[mask == 0] = 0
        shadows = raw == p["shadow_value"] if p["detect_shadows"] else np.zeros(raw.shape, bool)
        visible = (raw == 255) | (shadows if p["include_shadows"] else False)
        foreground = np.asarray(source).copy()
        foreground[~visible] = 0
        background = protected_background if rebuild_query else model.getBackgroundImage()
        if rebuild_query:
            state.pop("native", None)
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
                "background_reference": {
                    "id": state["reference_id"],
                    "file_sha256": reference_entry[2],
                    "replay_passes": state["reference_replay_passes"],
                    "update_policy": p["reference_update"],
                    "replay_verified": True,
                    "frozen_query_rebuild": rebuild_query,
                }
                if reference_entry is not None
                else None,
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
    reference_rebuild_frozen = True
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
