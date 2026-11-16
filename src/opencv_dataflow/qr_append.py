"""Atomic structured-append QR recovery from a scene or a bounded image group."""

import copy
import hashlib
from collections import Counter
from dataclasses import replace

import numpy as np

from .code_readers import ReadQRCode, barcode_image, code_graphics, code_record
from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .processing import image_value
from .qr_append_header import structured_header
from .qr_detector import make_detector, validate_qr


def assemble(fragments, expected_parts=0, expected_sha256="", maximum_bytes=65536):
    reasons = []
    valid = [f for f in fragments if f.get("header") is not None]
    if not fragments:
        reasons.append("no_qr_fragments")
    if len(valid) != len(fragments):
        reasons.append("unverified_qr_fragment")
    counts = {f["header"]["count"] for f in valid}
    parities = {f["header"]["parity"] for f in valid}
    if len(counts) > 1 or len(parities) > 1:
        reasons.append("mismatched_qr_fragment_set")
    required = next(iter(counts)) if len(counts) == 1 else expected_parts
    seen = Counter(f["header"]["index"] for f in valid)
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    missing = sorted(set(range(required)) - seen.keys())
    if duplicates:
        reasons.append("duplicate_qr_fragment_index")
    if missing or (required and len(valid) < required):
        reasons.append("missing_qr_fragments")
    if expected_parts and counts != {expected_parts}:
        reasons.append("unexpected_qr_fragment_count")
    raw = b""
    if not reasons and valid:
        raw = b"".join(
            bytes.fromhex(f["bytes_hex"]) for f in sorted(valid, key=lambda f: f["header"]["index"])
        )
        if len(raw) > maximum_bytes:
            raise ValidationError(
                "Assembled QR payload exceeds byte limit", "maximum_payload_bytes"
            )
        parity = 0
        for value in raw:
            parity ^= value
        if parities != {parity}:
            reasons.append("qr_payload_parity_mismatch")
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
            reasons.append("qr_expected_digest_mismatch")
    status = {
        "complete": not reasons and bool(valid),
        "candidate_count": len(fragments),
        "verified_fragments": len(valid),
        "declared_counts": sorted(counts),
        "declared_parities": sorted(parities),
        "missing_indices": missing,
        "duplicate_indices": duplicates,
        "reasons": reasons,
        "integrity": "raw QR codeword syndromes, unique complete indices, count and XOR parity",
        "expected_digest_checked": bool(expected_sha256),
    }
    return (raw if status["complete"] else None), status


class ReadQRAppend(Block):
    type_id, title = "inspect.read_qr_append", "Read structured-append QR"
    category, block_type, subtype = "Inspection", "Inspection", "Code reading"
    preview_safe = parallel_safe = process_inputs_readonly = preserve_image_sets = True
    composes_result_image = True
    inputs = (
        Port("image", required=False),
        Port("images", "any", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("text", "text", required=False),
        Port("bytes_hex", "text", required=False),
        Port("complete", "boolean"),
        Port("fragments", "data"),
        Port("status", "data"),
        Port("overlay", required=False),
    )
    defaults = {key: value for key, value in ReadQRCode.defaults.items() if key != "minimum_reads"}
    defaults = {
        **defaults,
        "backend": "aruco",
        "expected_parts": 0,
        "text_encoding": "ascii",
        "expected_sha256": "",
        "maximum_images": 16,
        "maximum_symbols": 16,
        "maximum_total_pixels": 33554432,
        "maximum_payload_bytes": 65536,
    }
    parameter_choices = {
        **ReadQRCode.parameter_choices,
        "mode": ("multi",),
        "text_encoding": ("ascii", "utf8", "hex"),
    }
    parameter_ranges = {
        key: value for key, value in ReadQRCode.parameter_ranges.items() if key != "minimum_reads"
    }
    parameter_ranges = {
        **parameter_ranges,
        "expected_parts": NumericRange(0, 16, whole=True),
        "maximum_images": NumericRange(1, 16, whole=True),
        "maximum_symbols": NumericRange(2, 16, whole=True),
        "maximum_total_pixels": NumericRange(441, 67108864, whole=True),
        "maximum_payload_bytes": NumericRange(1, 65536, whole=True),
    }
    parameter_help = {
        **ReadQRCode.parameter_help,
        "mode": "Multi detection; decode each candidate individually before assembly.",
        "expected_parts": "Zero accepts any complete 2..16-part set; otherwise require this count.",
        "text_encoding": "Decode the assembled bytes strictly; hex supports binary payloads.",
        "expected_sha256": "Optional 64-hex payload digest; XOR parity is not a unique identity.",
        "maximum_images": "Atomic group entry limit; no cross-run accumulation.",
        "maximum_symbols": "Candidate cap checked before per-symbol native decoding.",
        "maximum_total_pixels": "Combined image-group budget checked before any native detection.",
        "maximum_payload_bytes": "Total decoded fragment bytes, including rejected duplicates.",
        "draw_overlay": "Single image: fragment outlines/labels. Groups keep per-member geometry.",
    }
    input_help = {
        "image": "One camera scene or contact sheet; use exactly one of image/images.",
        "images": "FlowGroup of 1..16 images; order and labels do not identify fragments.",
        "mask": "Single-image search selection only; retain complete symbols and quiet zones.",
        "region": "Single-image region only; intersect with mask in original coordinates.",
    }

    def validate_parameters(self, parameters=None):
        # Reuse the exact reader controls without inheriting its ordinary-read result policy.
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_qr(p)
        if p["expected_parts"] == 1:
            raise ValidationError("Use zero or 2..16 expected parts", "expected_parts")
        if p["expected_parts"] > p["maximum_symbols"]:
            raise ValidationError(
                "Expected parts exceed candidate limit", "expected_parts", "maximum_symbols"
            )
        value = p["expected_sha256"]
        if not isinstance(value, str) or (
            value and (len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value))
        ):
            raise ValidationError(
                "Expected SHA256 must be empty or 64 hex digits", "expected_sha256"
            )

    def process(self, inputs):
        p = self.parameters
        if ("image" in inputs) == ("images" in inputs):
            raise ValidationError("Connect exactly one of image or images")
        grouped = "images" in inputs
        if grouped:
            group = inputs["images"]
            if not isinstance(group, FlowGroup) or len(group.items) > p["maximum_images"]:
                raise ValidationError("Images must be a bounded FlowGroup", "maximum_images")
            if "mask" in inputs or "region" in inputs:
                raise ValidationError("Apply selections per member upstream of a group")
            items = [
                (packet.value, packet.payload, group.labels[i])
                for i, packet in enumerate(group.items)
            ]
        else:
            items = [(inputs["image"], self.input_payloads.get("image", {}), "image")]
        total = 0
        for image, _, _ in items:
            if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
                raise ValidationError("Every group entry must be an image")
            image_value(image, (np.uint8,))
            if not image.size:
                raise ValidationError("QR images must not be empty")
            pixels = image.shape[0] * image.shape[1]
            if pixels > p["maximum_pixels"]:
                raise ValidationError("QR image exceeds pixel limit", "maximum_pixels")
            total += pixels
        if total > p["maximum_total_pixels"]:
            raise ValidationError(
                "QR image group exceeds total pixel limit", "maximum_total_pixels"
            )
        work, candidates, detector = [], 0, make_detector(p)
        for member, (image, payload, label) in enumerate(items):
            search, _ = barcode_image({"image": image} if grouped else inputs)
            if search is None:
                continue
            found, points = detector.detectMulti(search)
            if not found or points is None:
                continue
            corners = np.asarray(points).reshape(-1, 4, 2)
            candidates += len(corners)
            if candidates > p["maximum_symbols"]:
                raise ValidationError("QR candidates exceed symbol limit", "maximum_symbols")
            work.append((member, payload, label, search, corners))
        fragments, consumed = [], 0
        for member, payload, label, search, corners in work:
            for quad in corners:
                raw, grid = detector.decodeBytes(search, quad[None])
                consumed += len(raw)
                if consumed > p["maximum_payload_bytes"]:
                    raise ValidationError(
                        "Decoded QR fragments exceed byte limit", "maximum_payload_bytes"
                    )
                record = {
                    **code_record(raw.decode("ascii", errors="backslashreplace"), "QR_CODE", quad),
                    "member_index": member,
                    "member_label": label,
                    "bytes_hex": raw.hex(),
                    "ticket": copy.deepcopy(payload.get("ticket")),
                    "spatial": copy.deepcopy(payload.get("spatial")),
                    "header": None,
                }
                if not raw or grid is None or not grid.size:
                    record["error"] = "Native fragment decoding failed"
                else:
                    try:
                        record["header"] = structured_header(grid)
                    except ValidationError as exc:
                        record["error"] = str(exc)
                fragments.append(record)
        raw, status = assemble(
            fragments,
            int(p["expected_parts"]),
            p["expected_sha256"],
            int(p["maximum_payload_bytes"]),
        )
        text = NO_VALUE
        if raw is not None:
            try:
                text = raw.hex() if p["text_encoding"] == "hex" else raw.decode(p["text_encoding"])
            except UnicodeError:
                status["complete"] = False
                status["reasons"].append("qr_payload_text_encoding_mismatch")
                raw = None
        objects, overlay = [], NO_VALUE
        if not grouped and (self.generate_graphics or p["draw_overlay"]):
            color = (60, 230, 120) if status["complete"] else (0, 190, 255)
            for fragment in fragments:
                header = fragment["header"]
                caption = (
                    f"Part {header['index'] + 1}/{header['count']}: " if header else "Unverified: "
                )
                record = {**fragment, "text": caption + fragment["text"]}
                objects.extend(
                    replace(obj, color=color) for obj in code_graphics(record, (0, 1, 2, 3))
                )
            if p["draw_overlay"]:
                overlay = overlay_image(inputs["image"])
                for obj in objects:
                    draw_object(overlay, obj, np.eye(3))
        status["backend"] = p["backend"]
        status["input_images"] = len(items)
        status["decoded_fragment_bytes"] = consumed
        summary = (
            "Complete structured-append QR" if status["complete"] else "; ".join(status["reasons"])
        )
        return Result(
            {
                "text": text,
                "bytes_hex": raw.hex() if raw is not None else NO_VALUE,
                "complete": status["complete"],
                "fragments": fragments,
                "status": status,
                "overlay": overlay,
            },
            passed=status["complete"],
            message=summary,
            drawings={"QR fragments": GraphicDrawing(tuple(objects))}
            if objects and self.generate_graphics
            else {},
            payloads={"overlay": self.input_payloads.get("image", {})}
            if overlay is not NO_VALUE
            else {},
            fail_reasons=status["reasons"],
            event_data={"qr_append": status},
        )
