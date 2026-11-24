"""Ticketed Bin Wang and retina transient-area motion masks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing
from .images import ImageArray
from .mask_boundary import outlines
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .result_graphics import ACCEPTED
from .sequence import SEQUENCE_DEFAULTS, SEQUENCE_HELP, NativeSequenceBlock, SequenceStep

METHODS = ("bin wang", "retina transient")
MAX_PIXELS = 1_048_576
# At more than 200 foreground updates, the native neighborhood check splits
# the full model buffers for every candidate pixel. Relearn before that step.
PROMOTION_GUARD = 190
TRANSIENT_FILTERS = {
    "transient_threshold": ("thresholdON", 100.0),
    "local_temporal": ("localEnergy_temporalConstant", 0.5),
    "local_spatial": ("localEnergy_spatialConstant", 5.0),
    "neighborhood_temporal": ("neighborhoodEnergy_temporalConstant", 1.0),
    "neighborhood_spatial": ("neighborhoodEnergy_spatialConstant", 15.0),
    "context_temporal": ("contextEnergy_temporalConstant", 1.0),
    "context_spatial": ("contextEnergy_spatialConstant", 75.0),
}


def bin_wang_size(image):
    height, width = image.shape[:2]
    return width + (-width % 4), height + (-height % 4)


def native_bin_wang(size):
    create = getattr(getattr(cv2, "saliency", None), "MotionSaliencyBinWangApr2014_create", None)
    if create is None:
        raise ValidationError("Bin Wang motion saliency requires OpenCV contrib saliency")
    model = create()
    model.setImagesize(*size)
    if not model.init():
        raise ValidationError("OpenCV could not initialize Bin Wang motion saliency")
    return model


def native_retina(size, parameters):
    module = getattr(cv2, "bioinspired", None)
    retina_create = getattr(module, "Retina_create", None)
    segment_create = getattr(module, "TransientAreasSegmentationModule_create", None)
    if retina_create is None or segment_create is None:
        raise ValidationError("Retina transient segmentation requires OpenCV contrib bioinspired")
    retina = retina_create(size, False)
    retina.setupIPLMagnoChannel(normaliseOutput=False)
    segmenter = segment_create(size)
    if any(float(parameters[name]) != default for name, (_, default) in TRANSIENT_FILTERS.items()):
        xml = Element("opencv_storage")
        setup = SubElement(xml, "SegmentationModuleSetup")
        for name, (native_name, _) in TRANSIENT_FILTERS.items():
            SubElement(setup, native_name).text = repr(float(parameters[name]))
        SubElement(setup, "thresholdOFF").text = "100.0"
        with TemporaryDirectory(prefix="ocv-transient-") as directory:
            path = Path(directory) / "setup.xml"
            path.write_bytes(tostring(xml, encoding="utf-8", xml_declaration=True))
            segmenter.setup(str(path), False)
    return retina, segmenter


class MotionSaliency(NativeSequenceBlock):
    type_id, title, subtype = "sequence.motion_saliency", "Motion saliency", "Motion saliency"
    description = "Find moving foreground or transient areas across consecutive image frames."
    data_outputs = (Port("image"), Port("mask", visual_type="region"))
    process_inputs_readonly = True
    payload_reference = "image"
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 60,
        "method": METHODS[0],
        **{name: default for name, (_, default) in TRANSIENT_FILTERS.items()},
    }
    parameter_choices = {**NativeSequenceBlock.parameter_choices, "method": METHODS}
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0),
        **{
            name: NumericRange(0.01, 32) for name in TRANSIENT_FILTERS if name.endswith("_temporal")
        },
        **{name: NumericRange(1, 256) for name in TRANSIENT_FILTERS if name.endswith("_spatial")},
        "transient_threshold": NumericRange(0.1, 100000),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "method": "Bin Wang learns a background. Retina transient finds local changes over time.",
        "warmup": "Minimum frames before publishing. Bin Wang also waits to establish background.",
        "max_bytes_mb": "Budget for estimated retained native models across streams.",
        "transient_threshold": "Local motion energy threshold in Retina transient mode.",
        "local_temporal": "Local motion filter temporal constant in Retina transient mode.",
        "local_spatial": "Local motion filter spatial constant in Retina transient mode.",
        "neighborhood_temporal": "Neighborhood temporal constant in Retina transient mode.",
        "neighborhood_spatial": "Neighborhood spatial constant in Retina transient mode.",
        "context_temporal": "Wide-area context temporal constant in Retina transient mode.",
        "context_spatial": "Wide-area context spatial constant in Retina transient mode.",
    }
    input_help = {
        "image": "Consecutive color or grayscale frames from the same camera and pixel grid.",
        "mask": "Limit the published motion mask while the model processes the full image.",
        "region": "Limit motion to this region, intersected with the optional mask.",
        "reset": "Clear this stream's model before processing the current frame.",
        "stream": "Optional stream ID to maintain separate models for different cameras.",
    }
    output_help = {
        "image": "Current source image once ready, with optional motion outlines.",
        "mask": "Source-grid uint8 motion mask emitted once the sequence is ready.",
        "status": "Readiness, model restarts, selected motion count, and available video time.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "mask", "ready", "status"]

    def validate_sequence_image(self, image):
        height, width = image.shape[:2]
        if min(height, width) < 16 or max(height, width) > 4096 or width * height > MAX_PIXELS:
            raise ValidationError(
                "Motion saliency supports 16–4096 pixels per axis and up to 1,048,576 pixels"
            )

    def estimate_state_bytes(self, image):
        if self.parameters["method"] == METHODS[0]:
            width, height = bin_wang_size(image)
            return width * height * 64
        return image.shape[0] * image.shape[1] * 256

    def bin_wang(self, gray, state):
        height, width = gray.shape
        if "native" not in state:
            self.reserve_native_state(state, self.estimate_state_bytes(gray))
            padded_width, padded_height = bin_wang_size(gray)
            state["native"] = (
                native_bin_wang((padded_width, padded_height)),
                np.zeros((padded_height, padded_width), np.uint8),
            )
        model, updates = state["native"]
        pad_y, pad_x = updates.shape[0] - height, updates.shape[1] - width
        padded = cv2.copyMakeBorder(gray, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
        ok, native = model.computeSaliency(padded)
        if not ok or native is None or native.shape != updates.shape or native.dtype != np.uint8:
            raise ValidationError("OpenCV did not produce a valid Bin Wang motion mask")
        foreground = native != 0
        updates += foreground
        if int(updates.max()) >= PROMOTION_GUARD:
            state["pending_restart"] = "foreground persistence limit"
        motion = foreground[:height, :width]
        state["background_established"] = state.get("background_established", False) or (
            2 * np.count_nonzero(motion) < motion.size
        )
        return motion, state["background_established"]

    def retina_transient(self, gray, state):
        height, width = gray.shape
        if "native" not in state:
            self.reserve_native_state(state, self.estimate_state_bytes(gray))
            state["native"] = native_retina((width, height), self.parameters)
        retina, segmenter = state["native"]
        retina.run(gray)
        response = retina.getMagnoRAW().reshape(height, width)
        if not np.isfinite(response).all():
            raise ValidationError("OpenCV retina produced nonfinite motion energy")
        segmenter.run(response)
        native = segmenter.getSegmentationPicture()
        if native is None or native.shape != gray.shape or native.dtype != np.uint8:
            raise ValidationError("OpenCV did not produce a valid transient mask")
        return native != 0, True

    def step(self, inputs, state, mask):
        source = inputs["image"]
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        method = self.parameters["method"]
        video = self.input_payloads["image"].get("video_frame", {})
        video = video if isinstance(video, dict) else {}
        index = video.get("frame_index")
        index = index if type(index) is int else None
        previous = state.get("media_index")
        restart = state.pop("pending_restart", None)
        if index is not None and previous is not None and index <= previous:
            restart = "video position restarted"
        if restart is not None:
            for key in ("native", "native_bytes", "background_established"):
                state.pop(key, None)
            state["learning"] = 0
        state["media_index"] = index

        detect = self.bin_wang if method == METHODS[0] else self.retina_transient
        motion, established = detect(gray, state)
        if mask is not None:
            motion = motion & (mask != 0)
        return SequenceStep(
            {
                "image": source,
                "mask": ImageArray(np.where(motion, np.uint8(255), np.uint8(0)), "gray"),
            },
            ready=established,
            details={
                "method": method,
                "motion_pixels": int(np.count_nonzero(motion)),
                "frame_index": index,
                "timestamp_ms": video.get("timestamp_ms"),
                "model_restart": restart,
            },
        )

    def process(self, inputs):
        result = super().process(inputs)
        mask = result.outputs["mask"]
        if mask is not NO_VALUE:
            result.payloads["mask"].pop("alpha_representation", None)
            if self.generate_graphics and result.outputs["status"]["motion_pixels"]:
                result.drawings["motion saliency"] = GraphicDrawing(outlines(mask, ACCEPTED, 2048))
        return result
