"""Native static saliency maps and deterministic source-grid foreground masks."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle
from .images import ImageArray
from .mask_boundary import outlines
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask

FACTORIES = {
    "spectral residual": "StaticSaliencySpectralResidual_create",
    "fine-grained": "StaticSaliencyFineGrained_create",
}
MAX_PIXELS = 4_194_304


def threshold_scores(scores, parameters):
    mode = parameters["threshold_mode"]
    if mode == "otsu":
        gray = np.rint(scores * 255).astype(np.uint8).reshape(-1, 1)
        cutoff, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cutoff / 255, binary.ravel()
    cutoff = (
        float(np.percentile(scores, float(parameters["percentile"])))
        if mode == "percentile"
        else float(parameters["threshold"])
    )
    return cutoff, np.where((scores > 0) & (scores >= cutoff), np.uint8(255), np.uint8(0))


class StaticSaliencyMap(ImageTool):
    type_id, title, subtype = "cv.static_saliency", "Static saliency map", "Static saliency"
    description = "Find visually prominent regions with spectral residual or fine-grained saliency."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("saliency"),
        Port("mask", visual_type="region"),
        Port("report", "data"),
    )
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "method": "spectral residual",
        "spectral_width": 64,
        "spectral_height": 64,
        "threshold_mode": "otsu",
        "threshold": 0.5,
        "percentile": 95.0,
    }
    parameter_choices = {
        "method": tuple(FACTORIES),
        "threshold_mode": ("otsu", "fixed", "percentile"),
    }
    parameter_ranges = {
        "spectral_width": NumericRange(8, 512, whole=True),
        "spectral_height": NumericRange(8, 512, whole=True),
        "threshold": NumericRange(0, 1),
        "percentile": NumericRange(0, 100),
    }
    parameter_help = {
        "method": "Spectral residual uses frequency contrast. Fine-grained uses local contrast.",
        "spectral_width": "Working DFT grid width in pixels for spectral residual.",
        "spectral_height": "Working DFT grid height in pixels for spectral residual.",
        "threshold_mode": "Otsu uses selected 8-bit scores. Fixed and percentile use float scores.",
        "threshold": "Inclusive positive-score cutoff in [0,1] for fixed mode.",
        "percentile": "Cutoff percentile among selected pixels. Positive ties at the cutoff count.",
    }
    input_help = {
        "image": "Color or grayscale converted to uint8 luminance for comparison.",
        "mask": "Restrict published scores and threshold statistics after full-image analysis.",
        "region": "Output selection on the original image grid, intersected with any mask.",
    }
    output_help = {
        "image": "Original input with optional foreground outlines and a peak marker for review.",
        "saliency": "Normalized float32 saliency in [0,1]. Scores outside the selection are zero.",
        "mask": "Source-grid uint8 mask. White pixels meet the configured threshold.",
        "report": "Threshold, peak location, and foreground coverage in the selected pixels.",
    }

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 8 or max(height, width) > 8192 or height * width > MAX_PIXELS:
            raise ValidationError(
                "Static saliency needs 8–8192 pixels per axis and at most 4.19M pixels"
            )
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        selection = selection_mask(inputs, gray)
        include = selection != 0 if selection is not None else None
        selected_count = int(np.count_nonzero(include)) if include is not None else gray.size
        method = p["method"]
        create = getattr(getattr(cv2, "saliency", None), FACTORIES[method], None)
        if create is None:
            raise ValidationError("Static saliency requires the OpenCV contrib saliency module")

        response = np.zeros(gray.shape, np.float32)
        mask = np.zeros(gray.shape, np.uint8)
        cutoff, peak = None, None
        status = "empty selection"
        if selected_count:
            minimum, maximum, _, _ = cv2.minMaxLoc(gray)
            status = "uniform luminance"
            if minimum < maximum:
                detector = create()
                if method == "spectral residual":
                    detector.setImageWidth(int(p["spectral_width"]))
                    detector.setImageHeight(int(p["spectral_height"]))
                ok, native = detector.computeSaliency(gray)
                if (
                    not ok
                    or native is None
                    or native.shape != gray.shape
                    or not np.isfinite(native).all()
                ):
                    raise ValidationError("OpenCV did not produce a valid full-size saliency map")
                response = np.clip(native, 0, 1).astype(np.float32, copy=False)
                if include is not None:
                    response[~include] = 0
                _, peak_score, _, peak_point = cv2.minMaxLoc(response, mask=selection)
                status = "no positive response"
                if peak_score > 0:
                    status = "ok"
                    peak = {"point": list(peak_point), "score": peak_score}
                    if include is None:
                        cutoff, binary = threshold_scores(response.ravel(), p)
                        mask = binary.reshape(gray.shape)
                    else:
                        cutoff, mask[include] = threshold_scores(response[include], p)

        count = cv2.countNonZero(mask)
        report = {
            "model": "static_saliency_v1",
            "method": method,
            "status": status,
            "threshold_mode": p["threshold_mode"],
            "threshold": cutoff,
            "peak": peak,
            "selected_pixels": selected_count,
            "foreground_pixels": count,
            "foreground_fraction": count / selected_count if selected_count else 0.0,
        }
        drawings = {}
        if self.generate_graphics and peak is not None:
            if count:
                drawings["salient areas"] = GraphicDrawing(outlines(mask, ACCEPTED, 2048))
            drawings["saliency peak"] = GraphicDrawing(
                (graphic_circle(*peak["point"], 5, (60, 220, 255)),)
            )
        context = {
            key: value
            for key, value in self.input_payloads["image"].items()
            if key != "alpha_representation"
        }
        return Result(
            {
                "image": source,
                "saliency": ImageArray(response, "gray"),
                "mask": ImageArray(mask, "gray"),
                "report": report,
            },
            payloads={port: context for port in ("saliency", "mask", "report")},
            drawings=drawings,
            message=f"{method.capitalize()} · {count} foreground pixels"
            if status == "ok"
            else status,
        )
