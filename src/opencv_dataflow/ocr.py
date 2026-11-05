"""Text-region extraction and CTC recognition with explicit model contracts."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .dnn_blocks import ENGINES
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import file_key, local_path, selected_tensor
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import finite, gray_image, image_value, integer
from .registration import matrix_value
from .selection import SELECTION_INPUTS, contains, selection_mask


def unit_value(value, label):
    value = finite(value, label, 0)
    if value > 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return value


def text_rectangle(corners):
    corners = np.asarray(corners, np.float32)
    if (
        corners.shape != (4, 2)
        or not np.isfinite(corners).all()
        or np.max(np.abs(corners)) > 1_000_000
        or not cv2.isContourConvex(corners)
        or cv2.contourArea(corners) < 1
    ):
        raise ValueError("Text region needs four cyclically ordered corners enclosing an area")
    return corners


class TextRegions(Block):
    type_id, title = "ml.text_regions", "Text regions from probability map"
    category, block_type, subtype = "ML", "ML", "Text inspection"
    inputs = (*SELECTION_INPUTS, Port("tensors", "data"), Port("preprocessing", "data"))
    outputs = (Port("regions", "data"), Port("count", "number"), Port("overlay", required=False))
    defaults = {
        "output_name": "",
        "pixel_threshold": 0.3,
        "minimum_confidence": 0.5,
        "minimum_area": 8.0,
        "padding_ratio": 1.1,
        "maximum_regions": 64,
        "draw_overlay": True,
    }
    parameter_help = {
        "output_name": "Probability-map tensor name. Empty selects the first output.",
        "minimum_confidence": "Minimum mean probability inside each detected contour.",
        "minimum_area": "Minimum contour area in probability-map pixels.",
        "padding_ratio": "Multiply the detected rectangle dimensions by this ratio.",
    }

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        require_same_frame(
            (
                source,
                ValuePacket(None, self.input_payloads["tensors"]),
                ValuePacket(None, self.input_payloads["preprocessing"]),
            )
        )
        probability = selected_tensor(inputs["tensors"], p["output_name"])
        while probability.ndim > 2 and probability.shape[0] == 1:
            probability = probability[0]
        if (
            probability.ndim != 2
            or min(probability.shape) < 2
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError("Text detection requires a single 2D probability map in [0, 1]")
        geometry = inputs["preprocessing"]
        height, width = source.shape[:2]
        if not isinstance(geometry, dict) or geometry.get("input_size") != [width, height]:
            raise ValueError("Connect preprocessing geometry for the same original image")
        net_width, net_height = (
            integer(v, "Network dimension", 1, 4096) for v in geometry["network_size"]
        )
        inverse = matrix_value(geometry["network_to_input"], (2, 3))
        threshold = unit_value(p["pixel_threshold"], "Pixel threshold")
        minimum = unit_value(p["minimum_confidence"], "Minimum confidence")
        area_minimum = finite(p["minimum_area"], "Minimum area", 0)
        padding = finite(p["padding_ratio"], "Padding ratio", 1)
        if padding > 4:
            raise ValueError("Text region padding ratio must be at most 4")
        limit = integer(p["maximum_regions"], "Maximum regions", 1, 256)
        selection = selection_mask(inputs, source)
        binary = cv2.compare(probability, threshold, cv2.CMP_GT)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        scale = np.array([net_width / probability.shape[1], net_height / probability.shape[0]])
        for contour in contours:
            if cv2.contourArea(contour) < area_minimum:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
            confidence = cv2.mean(probability[y : y + h, x : x + w], mask=local)[0]
            if confidence < minimum:
                continue
            center, (w, h), angle = cv2.minAreaRect(contour)
            if min(w, h) < 1:
                continue
            corners = cv2.boxPoints((center, (w * padding, h * padding), angle))
            if cv2.contourArea(corners, oriented=True) < 0:
                corners = corners[::-1]
            edges = np.roll(corners, -1, axis=0) - corners
            i = max(range(2), key=lambda j: (np.linalg.norm(edges[j]), abs(edges[j, 0])))
            dx, dy = edges[i]
            if dx < -1e-5 or (abs(dx) <= 1e-5 and dy < 0):
                i += 2
            corners = np.roll(corners, -i, axis=0)
            mapped = cv2.transform(((corners + 0.5) * scale - 0.5)[None], inverse)[0]
            center = mapped.mean(axis=0)
            if not (0 <= center[0] <= width - 1 and 0 <= center[1] <= height - 1):
                continue
            if contains(selection, center):
                corners = text_rectangle(mapped)
                regions.append(
                    {
                        "corners": corners.tolist(),
                        "center": center.tolist(),
                        "confidence": confidence,
                    }
                )
        regions.sort(key=lambda item: (-item["confidence"], *item["center"][::-1]))
        regions = sorted(regions[:limit], key=lambda item: item["center"][::-1])
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        if overlay is not NO_VALUE:
            for region in regions:
                outline = np.rint(region["corners"]).astype(np.int32)
                cv2.polylines(overlay, [outline], True, (70, 220, 140), 2)
        return Result(
            {
                "regions": {"model": "ocr_regions_v1", "items": regions},
                "count": len(regions),
                "overlay": overlay,
            },
            graphics={"text regions": overlay} if overlay is not NO_VALUE else {},
            drawings={"text regions": GraphicDrawing(tuple(
                graphic_object("polyline", region["corners"], (70, 220, 140), closed=True)
                for region in regions
            ))} if self.generate_graphics and regions else {},
            message=f"{len(regions)} text regions",
        )


class RecognizeText(Block):
    type_id, title = "ml.recognize_text", "Recognize text (CTC ONNX)"
    category, block_type, subtype = "ML", "ML", "Text inspection"
    inputs = (*SELECTION_INPUTS, Port("regions", "data", required=False))
    outputs = (Port("text", "data"), Port("reads", "data"), Port("count", "number"))
    defaults = {
        "model_path": "",
        "vocabulary_path": "",
        "vocabulary": [],
        "width": 100,
        "height": 32,
        "color_format": "gray",
        "scale": 1 / 255,
        "mean": [0.0],
        "crop": False,
        "engine": "auto",
        "output_name": "",
        "decoder": "CTC-greedy",
        "beam_size": 10,
        "vocabulary_prune": 0,
        "background": 255,
        "minimum_reads": 1,
    }
    parameter_choices = {
        "color_format": ("gray", "bgr", "rgb"),
        "engine": tuple(ENGINES),
        "decoder": ("CTC-greedy", "CTC-prefix-beam-search"),
    }
    parameter_help = {
        "vocabulary_path": "UTF-8 token file, one token per line in model order, excluding blank.",
        "vocabulary": "Inline alternative to the token file. CTC blank is implicit at index 0.",
        "output_name": "Optional explicit CTC tensor output name.",
        "vocabulary_prune": "Maximum tokens per beam step. Zero disables pruning.",
        "background": "Fill masked or out-of-image pixels with this uint8 intensity.",
        "minimum_reads": "Minimum number of regions that produce nonempty text.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model, self._key = None, None
        self._tokens, self._token_key = None, None

    close = reset_runtime

    def vocabulary(self):
        p = self.parameters
        tokens = p["vocabulary"]
        if p["vocabulary_path"]:
            if tokens:
                raise ValueError("Choose an inline vocabulary or a vocabulary file")
            path = local_path(self, p["vocabulary_path"])
            key = file_key(path)
            if key != self._token_key:
                if key[1] > 1_000_000:
                    raise ValueError("OCR vocabulary file exceeds 1 MB")
                self._tokens = path.read_text(encoding="utf-8-sig").splitlines()
                self._token_key = key
            tokens = self._tokens
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 10_000
            or any(not isinstance(token, str) or not 1 <= len(token) <= 64 for token in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError("Vocabulary must contain 1–10000 unique nonempty tokens")
        return tokens

    def recognizer(self):
        p = self.parameters
        path = local_path(self, p["model_path"])
        key = (*file_key(path), p["engine"], p["output_name"])
        if key != self._key:
            net = cv2.dnn.readNetFromONNX(
                np.frombuffer(path.read_bytes(), np.uint8), ENGINES[p["engine"]]
            )
            model = cv2.dnn.TextRecognitionModel(net)
            if p["output_name"]:
                model.setOutputNames([p["output_name"]])
            self._model, self._key = model, key
        return self._model

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        if p["color_format"] == "gray":
            image = gray_image(source, (np.uint8,))
        else:
            image = image_value(source, (np.uint8,), canonical=True)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        width, height = (integer(p[k], k, 1, 4096) for k in ("width", "height"))
        if width * height > 4_194_304:
            raise ValueError("OCR input is limited to four million pixels")
        background = integer(p["background"], "Background", 0, 255)
        mean = p["mean"]
        if not isinstance(mean, list) or len(mean) != (1 if image.ndim == 2 else 3):
            raise ValueError("OCR mean requires one value per model input channel")
        mean = tuple(finite(value, "Mean") for value in mean)
        minimum = integer(p["minimum_reads"], "Minimum reads", 0, 256)
        selection = selection_mask(inputs, image)
        if selection is not None:
            image = image.copy()
            image[selection == 0] = background
        regions = inputs.get("regions")
        if regions is None:
            if selection is None:
                h, w = image.shape[:2]
                roi = [(image, {"bounds": [0, 0, w, h]})]
            elif selection.any():
                x, y, w, h = cv2.boundingRect(selection)
                roi = [(image[y : y + h, x : x + w], {"bounds": [x, y, w, h]})]
            else:
                roi = []
        else:
            require_same_frame((source, ValuePacket(None, self.input_payloads["regions"])))
            if (
                not isinstance(regions, dict)
                or regions.get("model") != "ocr_regions_v1"
                or not isinstance(regions.get("items"), list)
                or len(regions["items"]) > 256
            ):
                raise ValueError("Connect up to 256 OCR text regions")
            destination = np.float32(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            )
            if min(width, height) < 2:
                raise ValueError("Rectified OCR regions require model axes of at least two pixels")
            roi = []
            for region in regions["items"]:
                corners = text_rectangle(region["corners"])
                if contains(selection, corners.mean(axis=0)):
                    warp = cv2.getPerspectiveTransform(corners, destination)
                    roi.append((warp, region))
        reads = []
        if roi:
            vocabulary = self.vocabulary()
            model = self.recognizer()
            model.setInputParams(
                finite(p["scale"], "Scale"),
                (width, height),
                mean,
                p["color_format"] == "rgb",
                p["crop"],
            )
            model.setVocabulary(vocabulary)
            model.setDecodeType(p["decoder"])
            if p["decoder"] == "CTC-prefix-beam-search":
                model.setDecodeOptsCTCPrefixBeamSearch(
                    integer(p["beam_size"], "Beam size", 1, 256),
                    integer(p["vocabulary_prune"], "Vocabulary pruning", 0, len(vocabulary) + 1),
                )
            for tile, region in roi:
                if regions is not None:
                    tile = cv2.warpPerspective(
                        image, tile, (width, height), borderValue=(background,) * 3
                    )
                text = model.recognize(tile)
                reads.append({**region, "text": text})
        texts = [entry["text"] for entry in reads if entry["text"].strip()]
        passed = len(texts) >= minimum
        return Result(
            {"text": "\n".join(texts), "reads": reads, "count": len(texts)},
            passed=passed,
            message=f"Read {len(texts)} of {len(roi)} text regions",
            fail_reasons=[] if passed else ["ocr_read_count_below_minimum"],
        )


OCR_BLOCKS = (TextRegions, RecognizeText)
