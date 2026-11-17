"""Ordered source-region OCR with genuine bounded native batches and explicit CTC scores."""

import copy
import hashlib
import json
import re

import cv2
import numpy as np

from .batch_inference import BatchImageInference
from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .ml_common import local_path, selected_tensor
from .ocr import text_rectangle
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, moved_payload, require_same_frame
from .processing import image_value
from .recipes import unique_ref
from .selection import SELECTION_INPUTS, contains, selection_mask
from .tickets import ticket_key


def vocabulary(parameters, block):
    """Read one bounded vocabulary snapshot per activation, including its exact identity."""
    tokens = parameters["vocabulary"]
    raw = None
    if parameters["vocabulary_path"]:
        if tokens:
            raise ValidationError("Choose inline tokens or a vocabulary file", "vocabulary")
        path = local_path(block, parameters["vocabulary_path"])
        with path.open("rb") as stream:
            raw = stream.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValidationError("Vocabulary exceeds 1 MB", "vocabulary_path")
        if hashlib.sha256(raw).hexdigest() != parameters["vocabulary_sha256"]:
            raise ValidationError("Vocabulary file digest differs", "vocabulary_sha256")
        tokens = raw.decode("utf-8-sig").splitlines()
    if (
        not isinstance(tokens, list)
        or not 1 <= len(tokens) <= 10_000
        or any(not isinstance(t, str) or not 1 <= len(t) <= 64 for t in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValidationError(
            "Use 1..10000 unique nonempty tokens, excluding CTC blank", "vocabulary"
        )
    identity = hashlib.sha256(
        raw
        if raw is not None
        else json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return list(tokens), identity


def greedy_ctc(values, tokens, *, blank, activation, maximum_steps):
    """Collapse the best path; score is geometric mean of emitted run peak probabilities."""
    if activation not in ("softmax", "probability"):
        raise ValidationError("Choose softmax logits or normalized probabilities")
    if values.ndim != 2 or values.shape[1] != len(tokens) + 1:
        raise ValidationError("CTC classes must equal vocabulary length plus one blank")
    if not 1 <= values.shape[0] <= maximum_steps:
        raise ValidationError("CTC time steps exceed the configured limit", "maximum_steps")
    if not 0 <= blank < values.shape[1] or not np.isfinite(values).all():
        raise ValidationError("CTC blank index or finite samples are invalid", "blank_index")
    if activation == "softmax":
        scores = values.astype(np.float64) - values.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=1, keepdims=True)
    else:
        scores = values
        if (
            np.any(scores < 0)
            or np.any(scores > 1)
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-4, rtol=0)
        ):
            raise ValidationError("CTC probabilities must be in [0,1] and sum to one")
    path = scores.argmax(axis=1)
    emitted = []
    start = 0
    while start < len(path):
        end = start + 1
        while end < len(path) and path[end] == path[start]:
            end += 1
        index = int(path[start])
        if index != blank:
            token = tokens[index if index < blank else index - 1]
            emitted.append(
                {
                    "token": token,
                    "class_id": index,
                    "start_step": start,
                    "stop_step": end,
                    "confidence": float(scores[start:end, index].max()),
                }
            )
        start = end
    text = "".join(item["token"] for item in emitted)
    confidence = (
        float(np.exp(np.mean(np.log([item["confidence"] for item in emitted]))))
        if emitted
        else None
    )
    return text, confidence, emitted


class RecognizeTextBatch(BatchImageInference):
    type_id, title = "ml.recognize_text_batch", "Recognize text regions in batches"
    subtype = "Text inspection"
    # One source image produces one aggregate judgment; region rows retain their own verdicts.
    preserve_image_sets = False
    outcome_passthrough = False
    payload_reference = "image"
    inputs = (*SELECTION_INPUTS, Port("regions", "data", required=False))
    outputs = (
        Port("image"),
        Port("text", "text"),
        Port("reads", "data"),
        Port("count", "integer"),
        Port("report", "data"),
        Port("receipt", "data", required=False),
    )
    project_asset_parameters = (*BatchImageInference.project_asset_parameters, "vocabulary_path")
    defaults = {
        **copy.deepcopy(BatchImageInference.defaults),
        "input_name": "input",
        "width": 100,
        "height": 32,
        "color_format": "gray",
        "mean": [127.5],
        "std": [1.0],
        "scale": 1 / 127.5,
        "input_shape": [-1, 1, 32, 100],
        "spatial_mode": "configured",
        "ctc_layout": "TNC",
        "score_activation": "softmax",
        "blank_index": 0,
        "quadrilaterals": [],
        "vocabulary": [],
        "vocabulary_path": "",
        "vocabulary_sha256": "",
        "minimum_confidence": 0.0,
        "minimum_reads": 1,
        "aggregate": "all",
        "background": 255.0,
        "maximum_steps": 1024,
    }
    parameter_choices = {
        **BatchImageInference.parameter_choices,
        "spatial_mode": ("configured",),
        "ctc_layout": ("TNC", "NTC"),
        "score_activation": ("softmax", "probability"),
        "aggregate": ("all", "minimum_reads"),
    }
    parameter_ranges = {
        **BatchImageInference.parameter_ranges,
        "width": NumericRange(2, 4096, whole=True),
        "height": NumericRange(2, 4096, whole=True),
        "blank_index": NumericRange(0, 10000, whole=True),
        "minimum_confidence": NumericRange(0, 1),
        "minimum_reads": NumericRange(0, 64, whole=True),
        "background": NumericRange(0, 65535),
        "maximum_steps": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        **BatchImageInference.parameter_help,
        "quadrilaterals": "Source quads without Regions; empty uses image/selection bounds.",
        "ctc_layout": "Exact native CTC axes: time/batch/classes or batch/time/classes.",
        "score_activation": "Softmax for logits; Probability checks normalized class scores.",
        "blank_index": "CTC blank class; vocabulary lists all other classes in their native order.",
        "vocabulary": "Unique ordered tokens, excluding blank; multi-character tokens are allowed.",
        "vocabulary_path": "Alternative project-relative UTF-8 token lines; requires its SHA256.",
        "vocabulary_sha256": "File SHA256; inline tokens are sealed with parameters.",
        "minimum_confidence": "Geometric mean of emitted run peaks; not a CTC posterior.",
        "minimum_reads": "Minimum nonblank reads meeting the confidence limit.",
        "aggregate": "Require all selected rows to pass, or only Minimum reads.",
        "background": "Intensity outside selection and at borders, in source units.",
        "maximum_steps": "Bound time steps before greedy path decoding.",
        "maximum_images": "Maximum supplied regions, including those skipped by source ROI.",
        "batch_size": "Native crop rows per call; model must support declared batch dimensions.",
    }
    input_help = {
        "image": "One source acquisition; output unchanged with format/depth/ticket/frame.",
        "regions": "Source ocr_regions_v1; input order and IDs stay intact.",
        "mask": "Mask filters centers and fills excluded pixels with Background.",
        "region": "Source selection intersected with Mask; independent of the text quadrilaterals.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("vocabulary_path", "vocabulary_sha256"):
            if not isinstance(p[key], str) or len(p[key]) > 4096:
                raise ValidationError("Use bounded vocabulary text", key)
        if p["vocabulary_path"] and not re.fullmatch(r"[0-9a-f]{64}", p["vocabulary_sha256"]):
            raise ValidationError("Set the file's lowercase SHA256", "vocabulary_sha256")
        if (
            not isinstance(p["quadrilaterals"], list)
            or len(p["quadrilaterals"]) > p["maximum_images"]
        ):
            raise ValidationError(
                "Use a bounded JSON list of source quadrilaterals", "quadrilaterals"
            )
        for corners in p["quadrilaterals"]:
            text_rectangle(corners)
        if not isinstance(p["vocabulary"], list):
            raise ValidationError("Vocabulary must be a JSON list", "vocabulary")

    def canonical_outputs(self, arrays, submitted):
        if len(arrays) != 1:
            raise ValidationError("Select exactly one native CTC output", "output_names")
        name, values = next(iter(arrays.items()))
        if values.ndim != 3:
            raise ValidationError(
                "CTC output requires three explicit TNC or NTC axes", "ctc_layout"
            )
        if self.parameters["ctc_layout"] == "TNC":
            values = values.swapaxes(0, 1)
        if not 1 <= values.shape[1] <= self.parameters["maximum_steps"]:
            raise ValidationError("CTC time steps exceed the configured limit", "maximum_steps")
        return super().canonical_outputs({name: values}, submitted)

    def prepare_rows(self, inputs):
        p, source = self.parameters, inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        h, w = image.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError("Source image exceeds its pixel budget", "maximum_image_pixels")
        context = self.input_payloads.get("image", {})
        selection = selection_mask(inputs, source)
        if np.issubdtype(image.dtype, np.integer) and (
            p["background"] > np.iinfo(image.dtype).max or p["background"] % 1
        ):
            raise ValidationError("Background must fit the source integer depth", "background")
        if selection is not None:
            image = image.copy()
            image[selection == 0] = p["background"]
        regions = inputs.get("regions")
        if regions is None:
            if p["quadrilaterals"]:
                rows = [{"corners": corners} for corners in p["quadrilaterals"]]
            elif selection is None or selection.any():
                x, y, rw, rh = (0, 0, w, h) if selection is None else cv2.boundingRect(selection)
                rows = [
                    {
                        "corners": [
                            [x, y],
                            [x + rw - 1, y],
                            [x + rw - 1, y + rh - 1],
                            [x, y + rh - 1],
                        ]
                    }
                ]
            else:
                rows = []
        else:
            if (
                not isinstance(regions, dict)
                or regions.get("model") != "ocr_regions_v1"
                or not isinstance(regions.get("items"), list)
            ):
                raise ValidationError("Connect ocr_regions_v1 source quadrilaterals")
            rows = regions["items"]
            contexts = [context, self.input_payloads.get("regions", {}), regions.get("context", {})]
            require_same_frame([ValuePacket(None, c) for c in contexts])
            tickets = [c["ticket"] for c in contexts if "ticket" in c]
            if tickets and any(ticket_key(t) != ticket_key(tickets[0]) for t in tickets):
                raise ValidationError("OCR regions belong to another ticket")
            unique_ref([ValuePacket(None, c) for c in contexts])
            roots = [c["spatial"] for c in contexts if c.get("spatial", {}).get("root_id")]
            if roots and any(r["root_size"] != roots[0]["root_size"] for r in roots):
                raise ValidationError("OCR region root dimensions disagree")
            if "image_size" in regions and regions["image_size"] != [w, h]:
                raise ValidationError("OCR region image size disagrees with source")
        if len(rows) > p["maximum_images"]:
            raise ValidationError("Too many OCR regions", "maximum_images")
        width, height = p["width"], p["height"]
        if len(rows) * width * height > p["maximum_source_pixels"]:
            raise ValidationError(
                "Rectified crops exceed the combined pixel budget", "maximum_source_pixels"
            )
        destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        packets, labels, reads = [], [], []
        for index, region in enumerate(rows):
            if not isinstance(region, dict):
                raise ValidationError("Every OCR region must be a mapping")
            corners = text_rectangle(region.get("corners"))
            row = {
                "index": index,
                "region": copy.deepcopy(region),
                "text": "",
                "confidence": None,
                "status": "skipped",
                "passed": None,
                "characters": [],
                "context": copy.deepcopy(context),
            }
            reads.append(row)
            center = corners.mean(axis=0)
            if not (0 <= center[0] < w and 0 <= center[1] < h) or not contains(selection, center):
                row["skip_reason"] = "outside_source_or_selection"
                continue
            warp = cv2.getPerspectiveTransform(corners, destination)
            crop = cv2.warpPerspective(
                image, warp, (width, height), borderValue=(p["background"],) * 4
            )
            payload = moved_payload(context, np.linalg.inv(warp), [width, height])
            row["crop_context"] = copy.deepcopy(payload)
            packets.append(ValuePacket(ImageArray(crop, image_format(source)), payload))
            labels.append(str(index))
        return packets, labels, reads

    def process(self, inputs):
        self.begin_inference()
        try:
            packets, labels, reads = self.prepare_rows(inputs)
            tokens, digest = (None, None)
            info = {"native_calls": [], "padding_rows": 0, "image_count": 0}
            if packets:
                tokens, digest = vocabulary(self.parameters, self)
                result = self.infer_group({"images": FlowGroup(tuple(packets), tuple(labels))})
                info = result.outputs["info"]
                for packet, label in zip(result.outputs["results"].items, labels, strict=True):
                    row = reads[int(label)]
                    values = selected_tensor(
                        packet.value["tensors"], max_elements=self.parameters["max_output_elements"]
                    )
                    text, confidence, characters = greedy_ctc(
                        values[0],
                        tokens,
                        blank=self.parameters["blank_index"],
                        activation=self.parameters["score_activation"],
                        maximum_steps=self.parameters["maximum_steps"],
                    )
                    status = (
                        "blank"
                        if not text.strip()
                        else "low_confidence"
                        if confidence < self.parameters["minimum_confidence"]
                        else "read"
                    )
                    row["fail_reasons"] = (
                        []
                        if status == "read"
                        else [
                            {
                                "code": "ocr.blank" if status == "blank" else "ocr.confidence",
                                "message": "Blank text"
                                if status == "blank"
                                else "Read confidence below limit",
                            }
                        ]
                    )
                    native_receipt = packet.value.get("receipt")
                    if isinstance(native_receipt, dict):
                        row["native_call_index"] = native_receipt["native_call_index"]
                        row["native_row_index"] = native_receipt["native_row_index"]
                    row.update(
                        text=text,
                        confidence=confidence,
                        characters=characters,
                        status=status,
                        passed=status == "read",
                    )
            return self.finish(inputs, reads, info, digest)
        except Exception:
            self.reset_runtime()
            raise

    def finish(self, inputs, reads, info, vocabulary_digest):
        p = self.parameters
        selected = [row for row in reads if row["status"] != "skipped"]
        accepted = [row for row in selected if row["passed"]]
        failed = [row for row in selected if not row["passed"]]
        reasons = []
        if len(accepted) < p["minimum_reads"]:
            reasons.append({"code": "ocr.minimum_reads", "message": "Too few accepted OCR reads"})
        if failed and p["aggregate"] == "all":
            reasons.append(
                {
                    "code": "ocr.failed_regions",
                    "message": "Blank or low-confidence regions",
                }
            )
        passed = not reasons
        report = {
            "model": "ocr_batch_report_v1",
            "ctc_layout": p["ctc_layout"],
            "canonical_row_layout": "NTC",
            "decoder": "CTC-greedy",
            "confidence_semantics": "geometric_mean_emitted_run_peak_probability",
            "vocabulary_sha256": vocabulary_digest,
            "input_regions": len(reads),
            "selected_regions": len(selected),
            "accepted_reads": len(accepted),
            "blank_reads": sum(row["status"] == "blank" for row in selected),
            "failed_region_indices": [r["index"] for r in failed],
            "low_confidence_reads": sum(row["status"] == "low_confidence" for row in selected),
            "skipped_regions": len(reads) - len(selected),
            "batch": info,
        }
        drawings = {}
        if self.generate_graphics:
            objects = []
            for row in selected:
                corners = row["region"]["corners"]
                color = (70, 220, 140) if row["passed"] else (60, 80, 240)
                objects.append(graphic_object("polyline", corners, color, closed=True))
                objects.append(
                    graphic_object("label", [corners[0]], color, text=row["text"] or "<blank>")
                )
            if objects:
                drawings["OCR reads"] = GraphicDrawing(tuple(objects))
        context = self.input_payloads.get("image", {})
        outputs = {
            "image": inputs["image"],
            "text": "\n".join(row["text"] for row in accepted),
            "reads": reads,
            "count": len(accepted),
            "report": report,
        }
        return Result(
            outputs,
            payloads={key: copy.deepcopy(context) for key in outputs},
            passed=passed,
            fail_reasons=reasons,
            score=float(np.mean([r["confidence"] for r in accepted])) if accepted else None,
            drawings=drawings,
            message=f"{len(accepted)} accepted of {len(selected)} selected regions",
        )
