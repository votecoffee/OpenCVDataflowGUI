"""Native extremal text components and horizontal OCR regions."""

import copy
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import format_image, opencv_image
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import contains, selection_mask

_SOURCE_REVISION = "755e50675d97db9b7d449d8bd6b09888646f6c6e"
_MODEL_ROOT = Path(__file__).parent / "models" / "text_er"
_MODEL_DIGESTS = {
    "NM1": "75e67b14a84b7a4982ec63aae42eb2ade9aec4cc94e72e1af5e2863d227794ce",
    "NM2": "5e712fefc183584a9ac51544a291a53e070ba2ad785370168e774a905e2220a3",
}
_MAX_MODEL_BYTES = 2 * 1024 * 1024
_CHANNELS = {
    "gray": ("gray",),
    "rgb_lightness_gradient": ("red", "green", "blue", "lightness", "gradient"),
    "hsv_gradient": ("hue", "saturation", "value", "gradient"),
}


def _component_channels(api, rgb, mode, polarity):
    if mode == "gray":
        planes = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),)
    else:
        flag = (
            api.ERFILTER_NM_RGBLGrad
            if mode == "rgb_lightness_gradient"
            else api.ERFILTER_NM_IHSGrad
        )
        planes = api.computeNMChannels(rgb, _mode=flag)
    for name, plane in zip(_CHANNELS[mode], planes, strict=True):
        if polarity != "bright":
            yield name, "dark", plane
        if polarity != "dark":
            yield name, "bright", cv2.bitwise_not(plane)


class TextExtremalRegions(ImageTool):
    type_id, title = "cv.text_extremal_regions", "Detect extremal text regions"
    category, block_type, subtype = "Inspection", "Inspection", "Text inspection"
    description = "Find character candidates or horizontal text regions for a separate OCR block."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("regions", "data"),
        Port("count", "number"),
        Port("report", "data"),
    )
    defaults = {
        "mode": "horizontal_groups",
        "component_channels": "gray",
        "polarity": "both",
        "nm1_model_path": "",
        "nm2_model_path": "",
        "threshold_delta": 8,
        "minimum_area_fraction": 0.00025,
        "maximum_area_fraction": 0.13,
        "nm1_probability": 0.4,
        "nm1_nonmax": True,
        "nm1_probability_difference": 0.1,
        "nm2_probability": 0.3,
        "minimum_count": 1,
        "maximum_regions": 256,
    }
    parameter_choices = {
        "mode": ("horizontal_groups", "components"),
        "component_channels": tuple(_CHANNELS),
        "polarity": ("both", "dark", "bright"),
    }
    parameter_ranges = {
        "threshold_delta": NumericRange(1, 128, whole=True),
        "minimum_area_fraction": NumericRange(0, 1),
        "maximum_area_fraction": NumericRange(0, 1),
        "nm1_probability": NumericRange(0, 1),
        "nm1_probability_difference": NumericRange(0, 1),
        "nm2_probability": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 256, whole=True),
        "maximum_regions": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        "mode": "Group horizontal text in both gray polarities, or return individual components.",
        "component_channels": "Channels used only in components mode.",
        "polarity": "Contrast polarity used only in components mode.",
        "nm1_model_path": "Local first-stage Boost XML/YAML. Empty uses the shipped model.",
        "nm2_model_path": "Local second-stage Boost XML/YAML. Empty uses the shipped model.",
        "threshold_delta": "Intensity step used when extracting the first-stage component tree.",
        "minimum_area_fraction": "Minimum component area as a fraction of image area.",
        "maximum_area_fraction": "Maximum component area as a fraction of image area.",
        "nm1_probability": "Minimum first-stage estimated character probability.",
        "nm1_nonmax": "Suppress branch probabilities that are not local maxima.",
        "nm1_probability_difference": "Minimum difference between branch probability extrema.",
        "nm2_probability": "Minimum second-stage estimated character probability.",
        "minimum_count": "Fail when fewer OCR regions remain after selection. Zero allows none.",
        "maximum_regions": "Keep up to this many regions. Truncation marks the result as failed.",
    }
    input_help = {
        "image": "Gray or color. Converted to 8-bit RGB for the native detector.",
        "mask": "Retain boxes whose centers lie within the mask.",
        "region": "Retain boxes whose centers lie inside this region and any wired mask.",
    }
    output_help = {
        "regions": "Source-grid OCR rectangles. Connect directly to an OCR recognition block.",
        "report": "Native candidate count, filtering information, and classifier SHA-256 hashes.",
    }
    project_path_parameters = ("nm1_model_path", "nm2_model_path")
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_area_fraction"] >= p["maximum_area_fraction"]:
            raise ValidationError(
                "Minimum area must be smaller than maximum area",
                "minimum_area_fraction",
                "maximum_area_fraction",
            )
        if p["minimum_count"] > p["maximum_regions"]:
            raise ValidationError(
                "Minimum count must fit within the region limit", "minimum_count", "maximum_regions"
            )
        if type(p["nm1_nonmax"]) is not bool:
            raise ValidationError("First-stage suppression must be on or off", "nm1_nonmax")
        for field in self.project_path_parameters:
            if not isinstance(p[field], str):
                raise ValidationError("Choose a local classifier filename", field)

    def _make_filter(self, factory, stage, field, args):
        configured = self.parameters[field].strip()
        if not configured:
            path = _MODEL_ROOT / f"trained_classifier{stage}.xml"
        else:
            path = local_path(self, configured)
        if not path.is_file() or path.suffix.lower() not in (".xml", ".yml", ".yaml"):
            raise ValidationError("Use an existing OpenCV text classifier XML or YAML file", field)
        with path.open("rb") as stream:
            raw = stream.read(_MAX_MODEL_BYTES + 1)
        if not raw or len(raw) > _MAX_MODEL_BYTES:
            raise ValidationError("Text classifiers must contain up to 2 MiB", field)
        digest = hashlib.sha256(raw).hexdigest()
        if not configured and digest != _MODEL_DIGESTS[stage]:
            raise ValidationError("Bundled text classifier SHA-256 mismatch", field)
        info = {"sha256": digest, "source": "local_file" if configured else _SOURCE_REVISION}
        # Load the exact snapshot whose digest will be reported.
        with tempfile.NamedTemporaryFile(suffix=path.suffix, delete_on_close=False) as frozen:
            frozen.write(raw)
            frozen.close()
            if configured:
                classifier = cv2.ml.Boost_load(frozen.name)
                if (
                    classifier.empty()
                    or not classifier.isTrained()
                    or classifier.getVarCount() != (4 if stage == "NM1" else 7)
                ):
                    raise ValidationError(
                        "Model does not match the expected text classifier stage", field
                    )
            return factory(frozen.name, *args), info

    def process(self, inputs):
        source = inputs["image"]
        if min(source.shape[:2]) < 12 or source.shape[0] * source.shape[1] > 2_100_000:
            raise ValidationError(
                "Text region detection requires at least 12×12 and up to 2.1M pixels"
            )
        pixels = image_value(source, (np.uint8, np.uint16, np.float32))
        if pixels.dtype == np.float32 and (pixels.min() < 0 or pixels.max() > 1):
            raise ValidationError("Float text images must use intensity values in [0, 1]")
        api = getattr(cv2, "text", None)
        if api is None or not hasattr(api, "detectRegions"):
            raise ValidationError("Extremal text regions require OpenCV contrib text")
        selection = selection_mask(inputs, source)
        rgb = format_image(opencv_image(source), "rgb", "uint8")
        p = self.parameters
        nm1, info1 = self._make_filter(
            api.createERFilterNM1,
            "NM1",
            "nm1_model_path",
            (
                int(p["threshold_delta"]),
                float(p["minimum_area_fraction"]),
                float(p["maximum_area_fraction"]),
                float(p["nm1_probability"]),
                p["nm1_nonmax"],
                float(p["nm1_probability_difference"]),
            ),
        )
        nm2, info2 = self._make_filter(
            api.createERFilterNM2, "NM2", "nm2_model_path", (float(p["nm2_probability"]),)
        )
        raw = []
        if p["mode"] == "horizontal_groups":
            boxes = api.detectRegions(rgb, nm1, nm2, method=api.ERGROUPING_ORIENTATION_HORIZ)
            raw.extend((tuple(map(int, box)), {}) for box in boxes)
        else:
            for name, polarity, plane in _component_channels(
                api, rgb, p["component_channels"], p["polarity"]
            ):
                for contour in api.detectRegions(plane, nm1, nm2):
                    raw.append((cv2.boundingRect(contour), {"channel": name, "polarity": polarity}))
        return self.publish(source, selection, raw, {"NM1": info1, "NM2": info2})

    def publish(self, source, selection, raw, models):
        height, width = source.shape[:2]
        items = {}
        for bounds, extra in raw:
            x, y, w, h = bounds
            if not (w > 1 and h > 1 and 0 <= x <= width - w and 0 <= y <= height - h):
                continue
            center = (x + (w - 1) / 2, y + (h - 1) / 2)
            if bounds not in items and contains(selection, center):
                items[bounds] = {
                    "bounds": list(bounds),
                    "center": list(center),
                    "corners": [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
                    **extra,
                }
        p = self.parameters
        ordered = sorted(items.values(), key=lambda item: (item["bounds"][1], item["bounds"][0]))
        regions = ordered[: int(p["maximum_regions"])]
        for index, item in enumerate(regions):
            item["id"] = index
        truncated = len(ordered) > len(regions)
        reasons = []
        if len(regions) < int(p["minimum_count"]):
            reasons.append({"code": "text_er.count", "message": "Fewer regions than requested"})
        if truncated:
            reasons.append(
                {"code": "text_er.truncated", "message": "Text regions exceed the output limit"}
            )
        color = (70, 220, 140) if not reasons else (60, 80, 240)
        context = self.input_payloads.get("image", {})
        return Result(
            {
                "image": source,
                "regions": {
                    "model": "ocr_regions_v1",
                    "items": regions,
                    "image_size": [width, height],
                    "context": copy.deepcopy(context),
                },
                "count": len(regions),
                "report": {
                    "mode": p["mode"],
                    "native_candidates": len(raw),
                    "selected_regions": len(ordered),
                    "truncated": truncated,
                    "models": models,
                },
            },
            payloads={"regions": copy.deepcopy(context)},
            passed=not reasons,
            fail_reasons=reasons,
            drawings={
                "Text regions": GraphicDrawing(
                    tuple(graphic_box(*item["bounds"], color) for item in regions)
                )
            }
            if self.generate_graphics and regions
            else {},
            message=f"{len(regions)} text regions",
        )
