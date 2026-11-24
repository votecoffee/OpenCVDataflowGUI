"""Bounded OpenCV MOG, GMG, and CNT models for continuous inspection streams."""

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing
from .images import ImageArray, image_format
from .mask_boundary import outlines
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import finite, gray_image, image_value
from .result_graphics import ACCEPTED
from .sequence import SEQUENCE_DEFAULTS, SEQUENCE_HELP, NativeSequenceBlock, SequenceStep


class AdaptiveForeground(NativeSequenceBlock):
    subtype = "Foreground segmentation"
    process_inputs_readonly = True
    payload_reference = "image"
    data_outputs = (Port("image"), Port("mask", visual_type="mask"), Port("foreground"))
    defaults = {**SEQUENCE_DEFAULTS, "max_bytes_mb": 256.0}
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "warmup": "Minimum frames before publishing a mask. Use a clean scene during startup.",
        "max_bytes_mb": "Estimated retained model budget across streams, in MiB.",
    }
    input_help = {
        "image": "Consecutive frames on one camera grid, starting with a clean scene.",
        "mask": "Limit the published foreground; learning still uses the full image.",
        "region": "Intersect this region with the optional mask on the source grid.",
        "reset": "Clear this stream's model before processing the current frame.",
        "stream": "Optional key to keep independent camera models.",
    }
    output_help = {
        "image": "Source image with optional foreground outlines.",
        "mask": "Source-grid binary uint8 foreground mask.",
        "foreground": "Masked source image in its original encoding and depth.",
        "status": "Warmup, foreground counts, model memory estimate, and restart state.",
    }
    image_depths = (np.uint8,)
    has_background = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "mask", "foreground", "ready"]

    def validate_sequence_image(self, image):
        image_value(image, self.image_depths)
        if not image.size or image.shape[0] * image.shape[1] > 16_777_216:
            raise ValidationError("Adaptive foreground supports up to 16,777,216 image pixels")

    def working_image(self, source):
        image = image_value(source, self.image_depths, canonical=True)
        return image[..., :3] if image.ndim == 3 else image

    def model_factory(self, name):
        factory = getattr(getattr(cv2, "bgsegm", None), name, None)
        if factory is None:
            raise ValidationError(f"{self.title} requires OpenCV contrib bgsegm")
        return factory

    def minimum_frames(self):
        return 1

    def native_mask(self, model, image, seen, required):
        return model.apply(image)

    def step(self, inputs, state, mask):
        source = inputs["image"]
        image = self.working_image(source)
        video = self.input_payloads["image"].get("video_frame", {})
        index = video.get("frame_index") if isinstance(video, dict) else None
        index = index if type(index) is int else None
        previous = state.get("media_index")
        restarted = index is not None and previous is not None and index <= previous
        if restarted:
            state.pop("native", None)
            state.pop("native_bytes", None)
            state["learning"] = 0
        state["media_index"] = index
        if "native" not in state:
            self.reserve_native_state(state, self.estimate_state_bytes(source))
            state["native"] = self.new_model()
        model = state["native"]
        seen = state["learning"] + 1
        required = max(self.minimum_frames(), self.parameters["warmup"])
        binary = self.native_mask(model, image, seen, required)
        if binary is None or binary.shape != source.shape[:2] or binary.dtype != np.uint8:
            raise ValidationError("OpenCV did not produce a source-grid foreground mask")

        ready = seen >= required
        outputs = dict.fromkeys(self.flow_primary, NO_VALUE)
        count = None
        if ready:
            if mask is not None:
                binary = cv2.bitwise_and(binary, mask)
            count = cv2.countNonZero(binary)
            foreground = np.asarray(source).copy()
            foreground[binary == 0] = 0
            outputs.update(
                image=source,
                mask=ImageArray(binary, "gray"),
                foreground=ImageArray(foreground, image_format(source)),
            )
            if self.has_background:
                outputs["background"] = ImageArray(model.getBackgroundImage(), "gray")
        return SequenceStep(
            outputs,
            ready=ready,
            details={
                "model_frames": seen,
                "required_model_frames": required,
                "foreground_pixels": count,
                "native_model_bytes": state["native_bytes"],
                "frame_index": index,
                "model_restart": "video position restarted" if restarted else None,
            },
        )

    def process(self, inputs):
        result = super().process(inputs)
        mask = result.outputs["mask"]
        if mask is not NO_VALUE:
            for name in ("mask", "background"):
                if name in result.payloads:
                    result.payloads[name].pop("alpha_representation", None)
            if self.generate_graphics and result.outputs["status"]["foreground_pixels"]:
                result.drawings["adaptive foreground"] = GraphicDrawing(
                    outlines(mask, ACCEPTED, 2048)
                )
        return result


class MOGForeground(AdaptiveForeground):
    type_id, title = "sequence.mog_background", "MOG background subtraction"
    description = "Model a clean scene with per-pixel mixtures of Gaussian color distributions."
    defaults = {
        **AdaptiveForeground.defaults,
        "history": 200,
        "mixtures": 5,
        "background_ratio": 0.7,
        "noise_sigma": 0.0,
        "learning_rate": -1.0,
    }
    parameter_ranges = {
        **AdaptiveForeground.parameter_ranges,
        "history": NumericRange(1, 1000000, whole=True),
        "mixtures": NumericRange(1, 8, whole=True),
        "background_ratio": NumericRange(1e-6, 1),
        "noise_sigma": NumericRange(0, 255),
        "learning_rate": NumericRange(-1, 0.999999),
    }
    parameter_help = {
        **AdaptiveForeground.parameter_help,
        "history": "Frame history for automatic learning.",
        "mixtures": "Maximum Gaussian components per pixel.",
        "background_ratio": "Minimum cumulative weight of the background components.",
        "noise_sigma": "Background noise standard deviation. Zero selects the native default.",
        "learning_rate": "Use -1 for automatic learning or 0 to freeze after the first frame.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if -1 < p["learning_rate"] < 0:
            raise ValidationError("Learning rate must be -1 or at least zero", "learning_rate")

    def new_model(self):
        p = self.parameters
        return self.model_factory("createBackgroundSubtractorMOG")(
            int(p["history"]),
            int(p["mixtures"]),
            float(p["background_ratio"]),
            float(p["noise_sigma"]),
        )

    def estimate_state_bytes(self, image):
        channels = 1 if image.ndim == 2 else 3
        return (
            image.shape[0] * image.shape[1] * int(self.parameters["mixtures"]) * (8 + 8 * channels)
        )

    def native_mask(self, model, image, seen, required):
        return model.apply(image, learningRate=float(self.parameters["learning_rate"]))


class GMGForeground(AdaptiveForeground):
    type_id, title = "sequence.gmg_background", "GMG background subtraction"
    description = "Estimate foreground probability from a per-pixel history of quantized colors."
    image_depths = (np.uint8, np.uint16, np.float32, np.float64)
    defaults = {
        **AdaptiveForeground.defaults,
        "initialization_frames": 120,
        "decision_threshold": 0.8,
        "max_features": 64,
        "learning_rate": 0.025,
        "quantization_levels": 16,
        "background_prior": 0.8,
        "smoothing_size": 7,
        "update_background": True,
        "intensity_range": "native",
        "minimum_intensity": 0.0,
        "maximum_intensity": 1.0,
    }
    parameter_choices = {
        **AdaptiveForeground.parameter_choices,
        "intensity_range": ("native", "custom"),
    }
    parameter_ranges = {
        **AdaptiveForeground.parameter_ranges,
        "initialization_frames": NumericRange(1, 9998, whole=True),
        "decision_threshold": NumericRange(0, 1),
        "max_features": NumericRange(1, 256, whole=True),
        "learning_rate": NumericRange(1e-8, 1),
        "quantization_levels": NumericRange(1, 255, whole=True),
        "background_prior": NumericRange(1e-6, 0.999999),
        "smoothing_size": NumericRange(0, 31, whole=True),
    }
    parameter_help = {
        **AdaptiveForeground.parameter_help,
        "initialization_frames": "Native training frames, followed by a clean priming frame.",
        "decision_threshold": "Foreground probability threshold.",
        "max_features": "Maximum quantized colors per pixel. Higher values use more memory.",
        "learning_rate": "Weight of a new sample during live histogram updates.",
        "quantization_levels": "Number of levels per channel over the selected intensity range.",
        "background_prior": "Prior probability that a pixel belongs to the background.",
        "smoothing_size": "Odd median kernel width. Zero disables smoothing.",
        "update_background": "Keep adapting after warmup. Disable to hold the learned scene.",
        "intensity_range": "Native uses the full unsigned range or [0, 1] for float images.",
        "minimum_intensity": "Lower bound of the custom range in canonical image units.",
        "maximum_intensity": "Upper bound of the custom range in canonical image units.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        size = int(p["smoothing_size"])
        if size and size % 2 == 0:
            raise ValidationError("Smoothing size must be zero or odd", "smoothing_size")
        if p["intensity_range"] == "custom":
            low = finite(p["minimum_intensity"], "Minimum intensity")
            high = finite(p["maximum_intensity"], "Maximum intensity")
            if not 0 < high - low <= np.finfo(np.float64).max / 255:
                raise ValidationError(
                    "Custom intensity limits must define a positive finite quantization range",
                    "minimum_intensity",
                    "maximum_intensity",
                )

    def minimum_frames(self):
        return int(self.parameters["initialization_frames"]) + 2

    def new_model(self):
        p = self.parameters
        model = self.model_factory("createBackgroundSubtractorGMG")(
            int(p["initialization_frames"]), float(p["decision_threshold"])
        )
        model.setMaxFeatures(int(p["max_features"]))
        model.setDefaultLearningRate(float(p["learning_rate"]))
        model.setQuantizationLevels(int(p["quantization_levels"]))
        model.setBackgroundPrior(float(p["background_prior"]))
        model.setSmoothingRadius(int(p["smoothing_size"]))
        if p["intensity_range"] == "custom":
            model.setMinVal(float(p["minimum_intensity"]))
            model.setMaxVal(float(p["maximum_intensity"]))
        return model

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * (4 + 8 * int(self.parameters["max_features"]))

    def native_mask(self, model, image, seen, required):
        p = self.parameters
        if p["intensity_range"] == "custom":
            low, high = float(p["minimum_intensity"]), float(p["maximum_intensity"])
        else:
            low = 0.0
            high = 1.0 if image.dtype.kind == "f" else float(np.iinfo(image.dtype).max)
        if float(image.min()) < low or float(image.max()) > high:
            raise ValidationError("GMG image samples must fit the selected intensity range")
        # OpenCV 5.0.0 exposes its first valid color counts after the first
        # post-initialization update. Use a clean frame for that update.
        model.setUpdateBackgroundModel(p["update_background"] or seen < required)
        return model.apply(image)


class CNTForeground(AdaptiveForeground):
    type_id, title = "sequence.cnt_background", "CNT background subtraction"
    description = "Find foreground by counting consecutive stable grayscale pixel values."
    has_background = True
    data_outputs = (*AdaptiveForeground.data_outputs, Port("background"))
    defaults = {
        **AdaptiveForeground.defaults,
        "min_stability": 15,
        "max_stability": 900,
        "use_history": True,
        "history_scale": 1.0,
        "parallel": True,
    }
    parameter_ranges = {
        **AdaptiveForeground.parameter_ranges,
        "min_stability": NumericRange(1, 9999, whole=True),
        "max_stability": NumericRange(2, 1000000, whole=True),
        "history_scale": NumericRange(1e-6, 1),
    }
    parameter_help = {
        **AdaptiveForeground.parameter_help,
        "min_stability": "Consecutive stable frames before classifying a pixel as background.",
        "max_stability": "Maximum stability count of a pixel's historical background value.",
        "use_history": "Use accumulated pixel history as well as adjacent-frame stability.",
        "history_scale": "Fraction of maximum stability used in history mode.",
        "parallel": "Process image rows in parallel within OpenCV.",
    }
    output_help = {
        **AdaptiveForeground.output_help,
        "background": "Native grayscale estimate of the learned scene.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["min_stability"] >= p["max_stability"]:
            raise ValidationError(
                "Maximum stability must exceed minimum stability", "min_stability", "max_stability"
            )
        if p["use_history"] and int(p["max_stability"] * p["history_scale"]) <= p["min_stability"]:
            raise ValidationError(
                "Scaled history must exceed minimum stability", "history_scale", "min_stability"
            )

    def minimum_frames(self):
        return int(self.parameters["min_stability"]) + int(self.parameters["use_history"])

    def new_model(self):
        p = self.parameters
        return self.model_factory("createBackgroundSubtractorCNT")(
            int(p["min_stability"]), p["use_history"], int(p["max_stability"]), p["parallel"]
        )

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * 18

    def working_image(self, source):
        return gray_image(source, self.image_depths)

    def native_mask(self, model, image, seen, required):
        scale = float(self.parameters["history_scale"])
        return model.apply(image, learningRate=-1.0 if scale == 1.0 else scale)


ADAPTIVE_FOREGROUND_BLOCKS = (MOGForeground, GMGForeground, CNTForeground)
