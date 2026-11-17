"""Text-region extraction and CTC recognition with explicit model contracts."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry
from .dnn_blocks import ENGINES
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import file_key, local_path, selected_tensor
from .onnx_loading import load_onnx_snapshot
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import finite, gray_image, image_value, integer
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
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("tensors", "data"), Port("preprocessing", "data"))
    outputs = (
        Port("regions", "data"),
        Port("count", "number"),
        Port("overlay", required=False),
        Port("image"),
        Port("report", "data"),
    )
    defaults = {
        "output_name": "",
        "pixel_threshold": 0.3,
        "minimum_confidence": 0.5,
        "minimum_area": 8.0,
        "padding_ratio": 1.1,
        "maximum_regions": 64,
        "draw_overlay": True,
        "expansion": "rectangle_scale",
        "unclip_ratio": 2.0,
        "minimum_side": 1.0,
        "border_policy": "keep",
        "minimum_count": 0,
        "maximum_count": 0,
        "fail_on_truncation": True,
        "maximum_candidates": 2000,
        "maximum_tensor_elements": 4_194_304,
        "maximum_image_pixels": 4_194_304,
    }
    parameter_choices = {
        "expansion": ("rectangle_scale", "db_unclip"),
        "border_policy": ("keep", "discard"),
    }
    parameter_ranges = {
        "pixel_threshold": NumericRange(0, 1),
        "minimum_confidence": NumericRange(0, 1),
        "minimum_area": NumericRange(0, 16_777_216),
        "padding_ratio": NumericRange(1, 4),
        "unclip_ratio": NumericRange(0, 10),
        "minimum_side": NumericRange(1, 4096),
        "maximum_regions": NumericRange(1, 256, whole=True),
        "minimum_count": NumericRange(0, 256, whole=True),
        "maximum_count": NumericRange(0, 256, whole=True),
        "maximum_candidates": NumericRange(1, 20_000, whole=True),
        "maximum_tensor_elements": NumericRange(4, 16_777_216, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        "output_name": "Probability-map tensor name. Empty selects the first output.",
        "pixel_threshold": "Strictly greater probability starts an external contour.",
        "minimum_confidence": "Minimum mean probability inside each filled contour, inclusive.",
        "minimum_area": "Minimum contour area in probability-map pixels, inclusive.",
        "minimum_side": "Minimum unexpanded rectangle side in probability-map pixels.",
        "expansion": "Scale the rectangle, or add the DB area/perimeter offset.",
        "padding_ratio": "Rectangle scale only; multiply both detected rectangle dimensions.",
        "unclip_ratio": "DB only: add 2 x area x ratio / perimeter to both dimensions.",
        "border_policy": "Keep expanded corners outside the image, or discard the whole region.",
        "maximum_regions": "Retain highest-confidence regions, then order by center y,x.",
        "minimum_count": "Minimum retained regions to pass; zero permits an empty result.",
        "maximum_count": "Maximum retained count; zero disables this limit.",
        "fail_on_truncation": "Fail if accepted regions are truncated; retain current results.",
        "maximum_candidates": "Reject more external contours before scoring.",
        "maximum_tensor_elements": "Bound map samples before conversion; maximum axis is 4096.",
        "maximum_image_pixels": "Source-grid pixel limit; maximum source axis is 16384.",
        "draw_overlay": "Optional raster review; global/per-block graphics settings also apply.",
    }
    input_help = {
        "image": "Original image, returned unchanged; match inference ticket and frame.",
        "tensors": "One named exclusive float32 probability map, HW or leading singleton axes.",
        "preprocessing": "Matching positive resize/letterbox geometry from DNN inference.",
        "mask": "Optional source-grid selection; region centers must lie inside it.",
        "region": "Optional source-grid region; intersects Mask, applied by region center.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["output_name"], str) or len(p["output_name"]) > 1024:
            raise ValidationError("Use bounded tensor-name text", "output_name")
        for key in ("draw_overlay", "fail_on_truncation"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric text-region controls", key)
        if p["maximum_count"] and p["minimum_count"] > p["maximum_count"]:
            raise ValidationError("Count bounds are reversed", "minimum_count", "maximum_count")

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        height, width = source.shape[:2]
        if max(height, width) > 16384 or height * width > p["maximum_image_pixels"]:
            raise ValidationError("Source exceeds the pixel/axis budget", "maximum_image_pixels")
        checked = check_context(self, kind="Text detection")
        frames = [
            self.input_payloads.get(name, {}).get("spatial", {})
            for name in ("image", "tensors", "preprocessing")
        ]
        known = [item for item in frames if item.get("root_id") is not None]
        if known and any(item.get("root_size") != known[0].get("root_size") for item in known):
            raise ValidationError("Text detection inputs disagree on spatial root dimensions")
        size, inverse = preprocessing_geometry(inputs["preprocessing"], image_value(source))
        try:
            record = (
                inputs["tensors"][p["output_name"]]
                if p["output_name"]
                else next(iter(inputs["tensors"].values()))
            )
            if record.get("dtype") != "float32" or not isinstance(record.get("values"), list):
                raise ValidationError("Use a dequantized float32 tensor_v1 probability map")
            if len(record["values"]) > p["maximum_tensor_elements"]:
                raise ValidationError("Map exceeds tensor sample budget", "maximum_tensor_elements")
            if any(type(value) not in (int, float) for value in record["values"]):
                raise ValidationError("Probability-map samples must be flat numbers")
            probability = selected_tensor(
                inputs["tensors"], p["output_name"], max_elements=p["maximum_tensor_elements"]
            )
        except ValidationError:
            raise
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        while probability.ndim > 2 and probability.shape[0] == 1:
            probability = probability[0]
        if (
            probability.ndim != 2
            or min(probability.shape) < 2
            or max(probability.shape) > 4096
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValidationError("Text detection requires a bounded 2D probability map in [0,1]")
        selection = selection_mask(inputs, source)
        binary = cv2.compare(probability, p["pixel_threshold"], cv2.CMP_GT)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > p["maximum_candidates"]:
            raise ValidationError("Map exceeds external contour budget", "maximum_candidates")
        regions = []
        rejected = dict(area=0, confidence=0, side=0, center=0, border=0, selection=0)
        scale = np.array([size[0] / probability.shape[1], size[1] / probability.shape[0]])
        for contour in contours:
            if cv2.contourArea(contour) < p["minimum_area"]:
                rejected["area"] += 1
                continue
            x, y, w, h = cv2.boundingRect(contour)
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
            confidence = float(cv2.mean(probability[y : y + h, x : x + w], mask=local)[0])
            if confidence < p["minimum_confidence"]:
                rejected["confidence"] += 1
                continue
            center, (w, h), angle = cv2.minAreaRect(contour)
            if min(w, h) < p["minimum_side"]:
                rejected["side"] += 1
                continue
            if p["expansion"] == "db_unclip":
                # Offset the fitted rectangle in head pixels, before inverse mapping.
                distance = w * h * p["unclip_ratio"] / (2 * (w + h))
                sides = (w + 2 * distance, h + 2 * distance)
            else:
                sides = (w * p["padding_ratio"], h * p["padding_ratio"])
            corners = cv2.boxPoints((center, sides, angle))
            if cv2.contourArea(corners, oriented=True) < 0:
                corners = corners[::-1]
            edges = np.roll(corners, -1, axis=0) - corners
            i = max(range(2), key=lambda j: (np.linalg.norm(edges[j]), abs(edges[j, 0])))
            dx, dy = edges[i]
            if dx < -1e-5 or (abs(dx) <= 1e-5 and dy < 0):
                i += 2
            corners = np.roll(corners, -i, axis=0)
            # Preprocessing uses pixel EDGES. Map head centers through edges, then
            # return to source centers; subtracting .5 before inverse scaling is wrong.
            mapped = cv2.transform(((corners + 0.5) * scale)[None], inverse[:2])[0] - 0.5
            center = mapped.mean(axis=0)
            if not (0 <= center[0] <= width - 1 and 0 <= center[1] <= height - 1):
                rejected["center"] += 1
                continue
            if p["border_policy"] == "discard" and (
                np.any(mapped < 0)
                or np.any(mapped[:, 0] > width - 1)
                or np.any(mapped[:, 1] > height - 1)
            ):
                rejected["border"] += 1
                continue
            if not contains(selection, center):
                rejected["selection"] += 1
                continue
            corners = text_rectangle(mapped)
            regions.append(
                {"corners": corners.tolist(), "center": center.tolist(), "confidence": confidence}
            )
        accepted = len(regions)
        regions.sort(key=lambda item: (-item["confidence"], *item["center"][::-1]))
        regions = sorted(regions[: p["maximum_regions"]], key=lambda item: item["center"][::-1])
        truncated = accepted > len(regions)
        count_ok = len(regions) >= p["minimum_count"] and (
            not p["maximum_count"] or len(regions) <= p["maximum_count"]
        )
        reasons = []
        if not count_ok:
            reasons.append(
                {"code": "text_regions.count", "message": "Text-region count outside limits"}
            )
        if truncated and p["fail_on_truncation"]:
            reasons.append(
                {"code": "text_regions.truncated", "message": "Accepted text regions truncated"}
            )
        passed = not reasons
        color = (70, 220, 140) if passed else (60, 80, 240)
        overlay = (
            overlay_image(source) if p["draw_overlay"] and self.generate_graphics else NO_VALUE
        )
        if overlay is not NO_VALUE:
            for region in regions:
                cv2.polylines(
                    overlay, [np.rint(region["corners"]).astype(np.int32)], True, color, 2
                )
        context = self.input_payloads.get("image", {})
        outputs = {
            "regions": {
                "model": "ocr_regions_v1",
                "items": regions,
                "image_size": [width, height],
                "context": copy.deepcopy(context),
            },
            "count": len(regions),
            "overlay": overlay,
            "image": source,
            "report": {
                "model": "text_regions_decode_v1",
                "map_size": list(probability.shape[::-1]),
                "source_size": [width, height],
                "raw_contours": len(contours),
                "rejected": rejected,
                "accepted_before_limit": accepted,
                "truncated": truncated,
                "returned_regions": len(regions),
                "expansion": p["expansion"],
                "context_checks": checked,
                "grid_convention": "head_centers_via_network_edges_to_source_centers",
            },
        }
        return Result(
            outputs,
            payloads={
                key: copy.deepcopy(context)
                for key, value in outputs.items()
                if value is not NO_VALUE
            },
            passed=passed,
            score=max((r["confidence"] for r in regions), default=0),
            fail_reasons=reasons,
            graphics={"text regions": overlay} if overlay is not NO_VALUE else {},
            drawings={
                "text regions": GraphicDrawing(
                    tuple(
                        graphic_object("polyline", region["corners"], color, closed=True)
                        for region in regions
                    )
                )
            }
            if self.generate_graphics and regions
            else {},
            message=f"{len(regions)} text regions"
            + ("; accepted regions truncated" if truncated else ""),
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
            net, _ = load_onnx_snapshot(path.read_bytes(), ENGINES[p["engine"]])
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
