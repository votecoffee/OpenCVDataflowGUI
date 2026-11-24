"""Tesseract TSV recognition for source-grid OCR regions."""

import copy
import csv
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import format_image, opencv_image
from .ml_common import local_path
from .ocr import text_rectangle
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import ImageTool, image_value
from .recipes import unique_ref
from .selection import contains, selection_mask
from .tickets import ticket_key

_LAYOUTS = {
    "automatic": 3,
    "column": 4,
    "block": 6,
    "line": 7,
    "word": 8,
    "character": 10,
    "sparse": 11,
    "raw_line": 13,
}
_ENGINES = {"lstm": 1, "legacy": 0, "combined": 2, "automatic": 3}
_TSV_FIELDS = ("level", "left", "top", "width", "height", "conf", "text")


def _executable():
    configured = os.environ.get("OPENCV_DATAFLOW_TESSERACT", "")
    executable = "tesseract.exe" if os.name == "nt" else "tesseract"
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute() and candidate.name.lower() == executable and candidate.is_file():
            return candidate
        raise ValidationError("Set OPENCV_DATAFLOW_TESSERACT to an absolute Tesseract executable")
    if os.name == "nt" and Path(sys.prefix).name == ".venv":
        runtime = Path(sys.prefix).parent / ".local/tesseract/.pixi/envs/default/Library/bin"
        if (runtime / executable).is_file():
            return runtime / executable
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = Path(entry.strip('"'))
        if directory.is_absolute() and (directory / executable).is_file():
            return directory / executable
    raise ValidationError("Install Tesseract or set OPENCV_DATAFLOW_TESSERACT")


def _tsv_words(data, shape, minimum_confidence):
    try:
        rows = csv.DictReader(
            io.StringIO(data.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE
        )
        if any(name not in (rows.fieldnames or ()) for name in _TSV_FIELDS):
            raise ValueError("Missing Tesseract TSV columns")
        for row in rows:
            if row["level"] != "5" or not (row["text"] or "").strip():
                continue
            conf = float(row["conf"])
            x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
            if (
                not math.isfinite(conf)
                or not 0 <= conf <= 100
                or not (w > 0 and h > 0 and 0 <= x <= shape[1] - w and 0 <= y <= shape[0] - h)
            ):
                raise ValueError("Invalid Tesseract word confidence or geometry")
            if conf > minimum_confidence:
                yield row["text"].strip(), conf / 100, (x, y, w, h)
    except (ValueError, UnicodeError, TypeError, csv.Error) as exc:
        raise ValidationError("Tesseract returned malformed word results") from exc


class TesseractOCR(ImageTool):
    type_id, title = "cv.tesseract_ocr", "Recognize text (Tesseract)"
    category, block_type, subtype = "Inspection", "Inspection", "Text inspection"
    description = "Recognize words and return normalized confidences and source-grid boxes."
    inputs = (
        Port("image"),
        Port("regions", "data", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("text", "text"),
        Port("words", "data"),
        Port("report", "data"),
    )
    defaults = {
        "language": "eng",
        "tessdata_path": "",
        "language_sha256": {},
        "layout": "block",
        "engine": "lstm",
        "whitelist": "",
        "minimum_confidence": 30,
        "padding": 8,
        "timeout_seconds": 12.0,
    }
    parameter_choices = {"layout": tuple(_LAYOUTS), "engine": tuple(_ENGINES)}
    parameter_ranges = {
        "minimum_confidence": NumericRange(0, 100, whole=True),
        "padding": NumericRange(0, 128, whole=True),
        "timeout_seconds": NumericRange(0.1, 120),
    }
    parameter_help = {
        "language": "Tesseract language code or language codes joined with +.",
        "tessdata_path": "Traineddata folder. Relative paths are based on the saved project.",
        "language_sha256": "Optional map from language code to exact traineddata SHA-256.",
        "layout": "Tesseract page segmentation mode for each crop.",
        "engine": "Tesseract recognition engine. Fast traineddata uses LSTM.",
        "whitelist": "Optional literal characters allowed by the recognizer.",
        "minimum_confidence": "Retain words with confidence above this percentage.",
        "padding": "White pixels around each crop before recognition.",
        "timeout_seconds": "Total time allowed for all recognition subprocesses in one activation.",
    }
    input_help = {
        "image": "Gray or color, converted to 8-bit gray for recognition.",
        "regions": "Optional same-ticket OCR quadrilaterals. Each enclosing rectangle is cropped.",
        "mask": "Paint excluded source pixels white and select OCR crop centers.",
        "region": "Restrict recognition to a source region or its intersection with the mask.",
    }
    output_help = {
        "text": "Accepted words in crop and reading order.",
        "words": "Accepted text, confidence in [0,1], and source-grid bounds.",
        "report": "Crop count, settings, engine and configured language file digests.",
    }
    project_path_parameters = ("tessdata_path",)
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        language = p["language"]
        if not isinstance(language, str) or not re.fullmatch(
            r"[A-Za-z0-9_]{1,24}(?:\+[A-Za-z0-9_]{1,24}){0,3}", language
        ):
            raise ValidationError(
                "Use up to four Tesseract language codes joined with +", "language"
            )
        if not isinstance(p["tessdata_path"], str):
            raise ValidationError("Choose a local tessdata folder", "tessdata_path")
        pins = p["language_sha256"]
        if not isinstance(pins, dict) or any(
            key not in language.split("+")
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-fA-F]{64}", digest)
            for key, digest in pins.items()
        ):
            raise ValidationError(
                "Map configured language codes to 64-hex hashes", "language_sha256"
            )
        if pins and not p["tessdata_path"].strip():
            raise ValidationError("Choose the tessdata folder for pinned data", "tessdata_path")
        whitelist = p["whitelist"]
        if (
            not isinstance(whitelist, str)
            or len(whitelist) > 256
            or any(ord(c) < 32 for c in whitelist)
        ):
            raise ValidationError("Use up to 256 printable whitelist characters", "whitelist")

    def _language_files(self):
        p = self.parameters
        value = p["tessdata_path"]
        if not value.strip():
            return None, {}
        folder = local_path(self, value)
        hashes = {}
        for code in p["language"].split("+"):
            model = folder / f"{code}.traineddata"
            if not model.is_file() or not 0 < model.stat().st_size <= 128 * 1024 * 1024:
                raise ValidationError(
                    f"Missing or invalid language data for {code}", "tessdata_path"
                )
            with model.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if p["language_sha256"].get(code, digest).lower() != digest:
                raise ValidationError(
                    f"Language data SHA-256 differs for {code}", "language_sha256"
                )
            hashes[code] = digest
        return folder, hashes

    def _crops(self, inputs, selection):
        h, w = inputs["image"].shape[:2]
        regions = inputs.get("regions")
        if regions is None:
            rect = (0, 0, w, h) if selection is None else cv2.boundingRect(selection)
            return [rect] if min(rect[2:]) > 0 else []
        if (
            not isinstance(regions, dict)
            or regions.get("model") != "ocr_regions_v1"
            or not isinstance(regions.get("items"), list)
            or len(regions["items"]) > 32
            or regions.get("image_size", [w, h]) != [w, h]
        ):
            raise ValidationError("Connect up to 32 OCR regions on the source image grid")
        contexts = [
            self.input_payloads["image"],
            self.input_payloads["regions"],
            regions.get("context", {}),
        ]
        packets = [ValuePacket(None, context) for context in contexts]
        require_same_frame(packets)
        unique_ref(packets)
        if len({ticket_key(context["ticket"]) for context in contexts if "ticket" in context}) > 1:
            raise ValidationError("OCR regions belong to another ticket")
        crops = []
        for row in regions["items"]:
            if not isinstance(row, dict):
                raise ValidationError("OCR region entries must contain corners")
            corners = text_rectangle(row.get("corners"))
            if not contains(selection, corners.mean(axis=0)):
                continue
            x, y, rw, rh = cv2.boundingRect(corners)
            left, top = max(0, x), max(0, y)
            right, bottom = min(w, x + rw), min(h, y + rh)
            if right > left and bottom > top:
                crops.append((left, top, right - left, bottom - top))
        return crops

    def _recognize(self, gray, crops, command):
        p = self.parameters
        pad = int(p["padding"])
        deadline = time.monotonic() + float(p["timeout_seconds"])
        words = []
        for index, (x, y, w, h) in enumerate(crops):
            data = cv2.copyMakeBorder(
                gray[y : y + h, x : x + w], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
            )
            ok, png = cv2.imencode(".png", data)
            if not ok:
                raise ValidationError("Could not encode the OCR crop")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValidationError("Tesseract exceeded the OCR time limit", "timeout_seconds")
            try:
                result = subprocess.run(
                    command,
                    input=png.tobytes(),
                    capture_output=True,
                    timeout=remaining,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                raise ValidationError(
                    "Tesseract exceeded the OCR time limit", "timeout_seconds"
                ) from exc
            if result.returncode:
                detail = result.stderr.decode("utf-8", errors="replace").strip()[:768]
                raise ValidationError(f"Tesseract exited with code {result.returncode}. {detail}")
            for text, confidence, (left, top, rw, rh) in _tsv_words(
                result.stdout, data.shape, p["minimum_confidence"]
            ):
                lo_x, lo_y = max(0, left - pad), max(0, top - pad)
                hi_x, hi_y = min(w, left + rw - pad), min(h, top + rh - pad)
                if hi_x <= lo_x or hi_y <= lo_y:
                    continue
                words.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "crop_index": index,
                        "bounds": [x + lo_x, y + lo_y, hi_x - lo_x, hi_y - lo_y],
                    }
                )
        return words

    def process(self, inputs):
        source = inputs["image"]
        pixels = image_value(source, (np.uint8, np.uint16, np.float32))
        if not pixels.size or pixels.shape[0] * pixels.shape[1] > 8_400_000:
            raise ValidationError("OCR requires a nonempty image of at most 8.4 million pixels")
        if pixels.dtype == np.float32 and (pixels.min() < 0 or pixels.max() > 1):
            raise ValidationError("Float OCR images must use intensity values in [0, 1]")
        selection = selection_mask(inputs, source)
        crops = self._crops(inputs, selection)
        p = self.parameters
        pad = int(p["padding"])
        if sum((w + 2 * pad) * (h + 2 * pad) for _, _, w, h in crops) > 16_000_000:
            raise ValidationError("OCR crops exceed the combined pixel budget")
        folder, hashes = self._language_files()
        words = []
        if crops:
            command = [
                str(_executable()),
                "stdin",
                "stdout",
                "-l",
                p["language"],
                "--psm",
                str(_LAYOUTS[p["layout"]]),
                "--oem",
                str(_ENGINES[p["engine"]]),
                "-c",
                "tessedit_create_tsv=1",
            ]
            if folder is not None:
                command.extend(("--tessdata-dir", str(folder)))
            if p["whitelist"]:
                command.extend(("-c", f"tessedit_char_whitelist={p['whitelist']}"))
            gray = format_image(opencv_image(source), "gray", "uint8")
            if selection is not None:
                gray[selection == 0] = 255
            words = self._recognize(gray, crops, command)
            if hashes and hashes != self._language_files()[1]:
                raise ValidationError(
                    "Tesseract language data changed during recognition", "tessdata_path"
                )
        objects = []
        if self.generate_graphics:
            color = (70, 220, 140)
            for item in words:
                x, y, w, h = item["bounds"]
                objects.append(graphic_box(x, y, w, h, color))
                objects.append(
                    graphic_object("text", [(x, max(0, y - 4))], color, text=item["text"])
                )
        context = self.input_payloads.get("image", {})
        return Result(
            {
                "image": source,
                "text": " ".join(item["text"] for item in words),
                "words": {"model": "tesseract_words_v1", "items": words},
                "report": {
                    "backend": "tesseract_cli",
                    "language": p["language"],
                    "language_sha256": hashes,
                    "crops": len(crops),
                    "accepted_words": len(words),
                },
            },
            payloads={"words": copy.deepcopy(context)},
            passed=bool(words),
            message=f"{len(words)} recognized words",
            fail_reasons=[]
            if words
            else [{"code": "tesseract.no_words", "message": "No accepted words"}],
            drawings={"Recognized words": GraphicDrawing(tuple(objects))} if objects else {},
        )
