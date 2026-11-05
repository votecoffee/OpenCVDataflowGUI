"""Decode 2D and linear codes with OpenCV and retain their image-space geometry."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange
from .payloads import NO_VALUE, image_with_payload
from .processing import finite, gray_image
from .selection import SELECTION_INPUTS, selection_mask


def barcode_image(inputs):
    gray = gray_image(inputs["image"], (np.uint8,))
    mask = selection_mask(inputs, gray)
    if mask is None:
        return gray, None
    return (np.where(mask, gray, np.uint8(255)) if mask.any() else None), mask


def code_record(value, symbology, corners):
    corners = np.asarray(corners)
    return {
        "text": value,
        "symbology": symbology,
        "corners": corners.tolist(),
        "center": corners.mean(axis=0).tolist(),
    }


class CodeReader(Block):
    category, block_type, subtype = "Inspection", "Inspection", "Code reading"
    inputs = SELECTION_INPUTS
    outputs = (Port("codes", "data"), Port("count", "number"), Port("overlay", required=False))
    defaults = {"minimum_reads": 1, "draw_overlay": True}
    parameter_ranges = {"minimum_reads": NumericRange(0, 4095, whole=True)}

    def process(self, inputs):
        search, _ = barcode_image(inputs)
        codes = []
        if search is not None:
            decoded, symbologies, points = self.decode(search)
            if points is not None:
                for value, symbology, corners in zip(
                    decoded, symbologies, np.asarray(points).reshape(-1, 4, 2), strict=True
                ):
                    if value:
                        codes.append(code_record(value, symbology, corners))
        return self.code_result(inputs["image"], codes)

    def code_result(self, source, codes, **extra):
        minimum = int(self.parameters["minimum_reads"])
        overlay = overlay_image(source) if self.parameters["draw_overlay"] else None
        if overlay is not None:
            for code in codes:
                corners = np.rint(code["corners"]).astype(np.int32)
                cv2.polylines(overlay, [corners], True, (60, 230, 120), 2)
        count = len(codes)
        outputs = {
            "codes": codes,
            "count": count,
            "overlay": (
                image_with_payload(overlay, self.input_payloads.get("image", {}))
                if overlay is not None
                else NO_VALUE
            ),
            **extra,
        }
        payloads = {name: self.input_payloads.get("image", {}) for name in outputs}
        return Result(
            outputs,
            graphics={"decoded codes": overlay} if overlay is not None else {},
            drawings={"decoded codes": GraphicDrawing(tuple(
                graphic_object("polyline", code["corners"], (60, 230, 120), closed=True)
                for code in codes
            ))} if self.generate_graphics and codes else {},
            passed=count >= minimum,
            message=f"Decoded {count} codes (minimum {minimum})",
            payloads=payloads,
            fail_reasons=[] if count >= minimum else ["code_read_count_below_minimum"],
        )


class ReadQRCode(CodeReader):
    type_id, title = "inspect.read_qr", "Read QR codes"
    defaults = {
        **CodeReader.defaults,
        "mode": "multi",
        "eps_x": 0.2,
        "eps_y": 0.1,
        "use_alignment_markers": True,
    }
    parameter_choices = {"mode": ("multi", "single", "curved")}
    parameter_ranges = {
        **CodeReader.parameter_ranges,
        "eps_x": NumericRange(0),
        "eps_y": NumericRange(0),
    }

    def decode(self, image):
        detector = cv2.QRCodeDetector()
        detector.setEpsX(float(self.parameters["eps_x"]))
        detector.setEpsY(float(self.parameters["eps_y"]))
        detector.setUseAlignmentMarkers(self.parameters["use_alignment_markers"])
        mode = self.parameters["mode"]
        if mode == "multi":
            _, texts, points, _ = detector.detectAndDecodeMulti(image)
        else:
            decode = {
                "single": detector.detectAndDecode,
                "curved": detector.detectAndDecodeCurved,
            }[mode]
            value, points, _ = decode(image)
            texts = (value,)
        return texts, ("QR_CODE",) * len(texts), points


class ReadLinearBarcode(CodeReader):
    type_id, title = "inspect.read_barcode", "Read linear barcodes"
    defaults = {
        **CodeReader.defaults,
        "detector_scales": [0.01, 0.03, 0.06, 0.08],
        "downsampling_threshold": 512.0,
        "gradient_threshold": 64.0,
        "super_resolution_model": "",
        "super_resolution_weights": "",
    }

    def decode(self, image):
        model = self.parameters["super_resolution_model"]
        weights = self.parameters["super_resolution_weights"]
        if bool(model) != bool(weights):
            raise ValueError("Provide both barcode super-resolution model files")
        detector = (
            cv2.barcode.BarcodeDetector(model, weights)
            if model
            else (cv2.barcode.BarcodeDetector())
        )
        scales = [
            finite(value, "Detector scale", 0.000001)
            for value in self.parameters["detector_scales"]
        ]
        if not scales or any(value > 1 for value in scales):
            raise ValueError("Detector scales must be a nonempty list of fractions in (0, 1]")
        detector.setDetectorScales(scales)
        detector.setDownsamplingThreshold(
            finite(self.parameters["downsampling_threshold"], "Downsampling threshold", 1)
        )
        detector.setGradientThreshold(
            finite(self.parameters["gradient_threshold"], "Gradient threshold", 0)
        )
        _, texts, symbologies, points = detector.detectAndDecodeWithType(image)
        return texts, symbologies, points


CODE_READER_BLOCKS = (ReadQRCode, ReadLinearBarcode)
