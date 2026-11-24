"""Stateful source-grid detail and motion channels from OpenCV Retina."""

import cv2
import numpy as np

from .alpha_composition import split_layer, unassociate
from .core import Port
from .images import ImageArray, image_format
from .native_runtime import cpu_execution
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .sequence import SEQUENCE_DEFAULTS, SEQUENCE_HELP, NativeSequenceBlock, SequenceStep

_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_MOSAICS = {
    "bayer": "RETINA_COLOR_BAYER",
    "diagonal": "RETINA_COLOR_DIAGONAL",
    "random": "RETINA_COLOR_RANDOM",
}
_PARVO = {
    "photoreceptor_sensitivity": (0.75, 0, 1, "Compression strength for local light adaptation."),
    "photoreceptor_time": (0.9, 0, 128, "Photoreceptor temporal smoothing, in frames."),
    "photoreceptor_space": (0.53, 0.01, 256, "Photoreceptor spatial smoothing, in pixels."),
    "horizontal_gain": (0.01, 0, 1, "Amount of broad luminance retained in the detail channel."),
    "horizontal_time": (0.5, 0, 128, "Horizontal-cell temporal smoothing, in frames."),
    "horizontal_space": (7.0, 0.01, 256, "Horizontal-cell spatial smoothing, in pixels."),
    "detail_sensitivity": (0.75, 0, 1, "Local contrast compression in the detail channel."),
}
_MAGNO = {
    "parasol_gain": (0.0, 0, 1, "Gain of the parasol-cell local adaptation filter."),
    "parasol_time": (0.0, 0, 128, "Parasol-cell temporal smoothing, in frames."),
    "parasol_space": (7.0, 0.01, 256, "Parasol-cell spatial smoothing, in pixels."),
    "amacrine_time": (2.0, 0.01, 128, "Motion high-pass time constant, in frames."),
    "motion_sensitivity": (0.95, 0, 1, "Local contrast compression in the motion channel."),
    "motion_integration_time": (0.0, 0, 128, "Local motion averaging time, in frames."),
    "motion_integration_space": (7.0, 0.01, 256, "Local motion averaging radius, in pixels."),
}


def retina_input(image, color, payload):
    encoding = image_format(image)
    if payload.get("alpha_representation") == "premultiplied":
        rgb, alpha, _ = split_layer(image, "premultiplied", "Retina image")
        signal = unassociate(rgb, alpha)[..., ::-1].astype(np.float32)
        if signal.min() < 0 or signal.max() > 1:
            raise ValidationError("Retina requires normalized straight colors in [0, 1]")
        signal *= np.float32(255)
    else:
        if image.dtype == np.float64 and encoding in ("hsv", "lab"):
            image = ImageArray(image.astype(np.float32), encoding)
        native = image_value(image, _DEPTHS, canonical=True)
        signal = (native[..., :3] if native.ndim == 3 else native).astype(np.float32)
        if native.dtype == np.uint16:
            signal *= np.float32(255 / 65535)
        elif native.dtype.kind == "f":
            if signal.min() < 0 or signal.max() > 1:
                raise ValidationError("Retina requires floating-point colors in [0, 1]")
            signal *= np.float32(255)
    if color and signal.ndim == 2:
        signal = cv2.cvtColor(signal, cv2.COLOR_GRAY2BGR)
    elif not color and signal.ndim == 3:
        signal = cv2.cvtColor(signal, cv2.COLOR_BGR2GRAY)
    # Prevent nonfinite adaptation in the native model on zero-light frames.
    return np.maximum(signal, np.float32(1e-4), out=signal)


def retina_model(size, color, p):
    module = getattr(cv2, "bioinspired", None)
    create = getattr(module, "Retina_create", None)
    if create is None:
        raise ValidationError("Retina preprocessing requires OpenCV contrib bioinspired")
    with cpu_execution():
        model = create(size, color, getattr(module, _MOSAICS[p["color_sampling"]]))
    model.setupOPLandIPLParvoChannel(
        color, p["normalize_detail"], *(float(p[key]) for key in _PARVO)
    )
    model.setupIPLMagnoChannel(p["normalize_motion"], *(float(p[key]) for key in _MAGNO))
    model.setColorSaturation(p["saturate_colors"], float(p["color_saturation"]))
    # Native motion filtering uses intermediate signals computed by Parvo.
    model.activateMovingContoursProcessing(p["channels"] != "detail")
    return model


def native_channel(buffer, shape, label, color=False):
    height, width = shape
    count = height * width * (3 if color else 1)
    if (
        buffer is None
        or buffer.size != count
        or buffer.dtype != np.float32
        or not np.isfinite(buffer).all()
    ):
        raise ValidationError(f"OpenCV Retina returned an invalid {label} response")
    if not color:
        return buffer.reshape(height, width)
    return buffer.reshape(3, height, width).transpose(1, 2, 0)


class RetinaPreprocessing(NativeSequenceBlock):
    type_id, title = "sequence.retina", "Retina contrast and motion"
    subtype = "Temporal preprocessing"
    description = "Adapt to luminance and reveal detail and temporal motion in consecutive frames."
    process_inputs_readonly = True
    requires_opencl_context = True
    payload_reference = "image"
    data_outputs = (
        Port("detail", required=False),
        Port("motion", required=False),
        Port("detail_raw", required=False),
        Port("motion_raw", required=False),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 6,
        "max_bytes_mb": 256.0,
        "channels": "both",
        "color_mode": "auto",
        "color_sampling": "bayer",
        "normalize_detail": True,
        "normalize_motion": False,
        "saturate_colors": False,
        "color_saturation": 4.0,
        **{key: spec[0] for key, spec in (_PARVO | _MAGNO).items()},
    }
    parameter_choices = {
        **NativeSequenceBlock.parameter_choices,
        "channels": ("both", "detail", "motion"),
        "color_mode": ("auto", "gray", "color"),
        "color_sampling": tuple(_MOSAICS),
    }
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0),
        "color_saturation": NumericRange(0.01, 32),
        **{key: NumericRange(spec[1], spec[2]) for key, spec in (_PARVO | _MAGNO).items()},
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "max_bytes_mb": "Budget for estimated retained Retina models across streams.",
        "channels": "Publish detail (Parvo), motion (Magno), or both.",
        "color_mode": "Follow the source layout or force grayscale or color detail.",
        "color_sampling": "Native color mosaic. A random mosaic varies between model instances.",
        "normalize_detail": "Apply native per-frame scaling to the detail response.",
        "normalize_motion": "Apply per-frame scaling. Turn off to compare magnitudes over time.",
        "saturate_colors": "Enable the native color saturation adjustment.",
        "color_saturation": "Strength of the optional native color saturation.",
        **{key: spec[3] for key, spec in (_PARVO | _MAGNO).items()},
    }
    input_help = {
        "image": "Consecutive color or grayscale frames on one camera grid.",
        "mask": "Limit published pixels while the model adapts to the full frame.",
        "region": "Intersect the source-grid region with the optional mask.",
        "reset": "Recreate the native model for the current stream.",
        "stream": "Optional stream ID for independent temporal models.",
    }
    output_help = {
        "detail": "Parvo contrast and contour response clipped to the byte range.",
        "motion": "Magno temporal response as a byte image.",
        "detail_raw": "Float32 Parvo response before byte conversion.",
        "motion_raw": "Float32 Magno response before byte conversion.",
        "status": "Readiness, motion response statistics and available video position.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["detail", "motion", "ready", "status"]
        self.preview_output = "detail"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("normalize_detail", "normalize_motion", "saturate_colors"):
            if type(p[name]) is not bool:
                raise ValidationError(
                    f"{name.replace('_', ' ').capitalize()} must be boolean", name
                )

    def validate_sequence_image(self, image):
        image_value(image, _DEPTHS)
        height, width = image.shape[:2]
        if min(height, width) < 16 or max(height, width) > 4096 or height * width > 1_048_576:
            raise ValidationError("Retina supports 16–4096 pixels per axis and 1,048,576 total")

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * 512

    def step(self, inputs, state, mask):
        source, p = inputs["image"], self.parameters
        payload = self.input_payloads["image"]
        height, width = source.shape[:2]
        mode = p["color_mode"]
        color = mode == "color" or (mode == "auto" and source.ndim == 3)
        encoding = (
            "rgb"
            if color and image_format(source) in ("rgb", "rgba")
            else ("bgr" if color else "gray")
        )
        video = payload.get("video_frame", {})
        video = video if isinstance(video, dict) else {}
        index = video.get("frame_index")
        index = index if type(index) is int else None
        previous = state.get("media_index")
        restarted = index is not None and previous is not None and index <= previous
        if restarted:
            state.pop("native", None)
            state.pop("native_bytes", None)
            state["learning"] = 0
        state["media_index"] = index
        signal = retina_input(source, color, payload)
        if "native" not in state:
            self.reserve_native_state(state, self.estimate_state_bytes(source))
            state["native"] = retina_model((width, height), color, p)
        model = state["native"]
        model.run(signal)

        outputs = dict.fromkeys(self.flow_primary, NO_VALUE)
        ready = state["learning"] + 1 >= p["warmup"]
        selected = mask != 0 if mask is not None else None
        count = int(selected.sum()) if selected is not None else width * height
        details = {
            "frame_index": index,
            "timestamp_ms": video.get("timestamp_ms"),
            "model_restart": "video position restarted" if restarted else None,
            "selected_pixels": count,
            "motion_normalized": p["normalize_motion"],
        }
        show_detail = "detail" in self.exposed_outputs or "detail_raw" in self.exposed_outputs
        if ready and p["channels"] != "motion" and show_detail:
            detail = native_channel(model.getParvoRAW(), (height, width), "detail", color)
            if color:
                detail = np.ascontiguousarray(detail[..., ::-1] if encoding == "bgr" else detail)
            self.channel_outputs(outputs, "detail", detail, encoding, selected)
        if ready and p["channels"] != "detail":
            motion = native_channel(model.getMagnoRAW(), (height, width), "motion")
            values = motion if selected is None else motion[selected]
            details["motion_mean"] = float(values.mean()) if count else None
            details["motion_peak"] = float(values.max()) if count else None
            self.channel_outputs(outputs, "motion", motion, "gray", selected)
        return SequenceStep(outputs, ready=ready, details=details)

    def channel_outputs(self, outputs, name, signal, encoding, selection):
        if selection is not None:
            signal[~selection] = 0
        if name in self.exposed_outputs:
            display = np.rint(np.clip(signal, 0, 255)).astype(np.uint8)
            outputs[name] = ImageArray(display, encoding)
        if f"{name}_raw" in self.exposed_outputs:
            outputs[f"{name}_raw"] = ImageArray(signal, encoding)

    def process(self, inputs):
        result = super().process(inputs)
        for name in self.flow_primary:
            if name in result.payloads:
                result.payloads[name].pop("alpha_representation", None)
                result.payloads[name].pop("raw_sensor", None)
        return result
