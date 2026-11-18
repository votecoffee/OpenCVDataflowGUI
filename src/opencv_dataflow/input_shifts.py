"""Controlled inspection input shifts, paired observations and bounded robustness reports."""

import copy

import cv2
import numpy as np

from .classification_evaluation import output_path, text
from .core import Block, Port, Result
from .images import ImageArray, image_format
from .local_anomaly import hash_image, is_hash, sha
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, moved_payload, require_same_frame
from .tickets import ticket_key
from .unicode_text import TextStyle, draw_unicode

SHIFT = "inspection_input_shift_v1"
OBSERVATION = "inspection_shift_observation_v1"
MAX_PIXELS = 4_194_304


def binary(value, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != shape:
        raise ValidationError("Masks must be uint8 monochrome on the current image grid")
    return np.asarray(value) != 0


def pack(mask):
    return np.packbits(mask.ravel()).tobytes().hex()


def unpack(value, shape):
    count = shape[0] * shape[1]
    if not isinstance(value, str) or len(value) != 2 * ((count + 7) // 8):
        raise ValidationError("Packed observation mask length differs from its grid")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError("Observation masks must use hexadecimal packed bits") from exc
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))[:count].reshape(shape).astype(bool)
    if pack(bits) != value:
        raise ValidationError("Observation mask has noncanonical padding or encoding")
    return bits


def affine(value):
    result = np.asarray(value, np.float64)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or np.max(np.abs(result)) > 1e7
        or not np.array_equal(result[2], [0, 0, 1])
        or abs(np.linalg.det(result)) < 1e-8
    ):
        raise ValidationError("Use a bounded invertible affine reference-to-image transform")
    return result


def warp(value, transform, size, interpolation=cv2.INTER_NEAREST, border=0):
    return cv2.warpAffine(
        value,
        transform[:2],
        size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def iou(first, second, support):
    union = int(np.count_nonzero((first | second) & support))
    return float(np.count_nonzero(first & second & support) / union) if union else None


class _ShiftBlock(Block):
    category = block_type = "ML"
    subtype = "Input-shift evaluation"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or rule.whole and type(p[key]) is not int:
                raise ValidationError(
                    "Use an integer" if rule.whole else "Use a numeric value", key
                )
        for key, default in self.defaults.items():
            if type(default) is bool and type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)


class ImageInputShift(_ShiftBlock):
    type_id, title = "ml.image_input_shift", "Controlled inspection image shift"
    inputs = (Port("image"), Port("truth"))
    outputs = (Port("image"), Port("truth"), Port("support"), Port("shift", "data"))
    defaults = {
        "variant": "baseline",
        "expected_pass": True,
        "gain": 1.0,
        "offset": 0.0,
        "angle_degrees": 0.0,
        "scale": 1.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "interpolation": "linear",
        "border_value": 0.0,
        "blur_kernel": 0,
        "blur_sigma": 0.0,
        "maximum_image_pixels": MAX_PIXELS,
    }
    parameter_choices = {"interpolation": ("nearest", "linear")}
    parameter_ranges = {
        "gain": NumericRange(0, 8),
        "offset": NumericRange(-1, 1),
        "angle_degrees": NumericRange(-180, 180),
        "scale": NumericRange(0.1, 4),
        "translation_x": NumericRange(-16384, 16384),
        "translation_y": NumericRange(-16384, 16384),
        "border_value": NumericRange(0, 1),
        "blur_kernel": NumericRange(0, 31, whole=True),
        "blur_sigma": NumericRange(0, 10),
        "maximum_image_pixels": NumericRange(1, MAX_PIXELS, whole=True),
    }
    parameter_help = {
        "variant": "Unique paired case name. Baseline requires neutral gain/offset/pose/blur.",
        "expected_pass": "Independent image-level label; retained when geometry changes.",
        "offset": "Full-scale offset after affine sampling and gain; clipped [0,1].",
        "blur_kernel": "Zero disables blur; otherwise odd 3..31. Zero sigma selects native sigma.",
        "border_value": "Full-scale affine padding; padded/blur-affected borders are unscored.",
    }
    input_help = {
        "truth": "Independent defect mask. A known-clean sample needs an explicit empty mask."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        text(p["variant"], "variant", maximum=64)
        k = p["blur_kernel"]
        if k and (k < 3 or k % 2 == 0) or not k and p["blur_sigma"]:
            raise ValidationError(
                "Use an odd blur kernel 3..31, or zero kernel/sigma", "blur_kernel", "blur_sigma"
            )
        if p["variant"] == "baseline" and any(
            p[k] != v
            for k, v in {
                "gain": 1,
                "offset": 0,
                "angle_degrees": 0,
                "scale": 1,
                "translation_x": 0,
                "translation_y": 0,
                "blur_kernel": 0,
            }.items()
        ):
            raise ValidationError("Baseline must leave pixels unchanged", "variant")

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        if (
            image.dtype not in (np.uint8, np.uint16, np.float32)
            or image.ndim not in (2, 3)
            or image.ndim == 3
            and image.shape[2] != 3
            or image_format(image) not in ("gray", "bgr", "rgb")
            or max(image.shape[:2]) > 16384
            or image.shape[0] * image.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError(
                "Use bounded gray/BGR/RGB uint8, uint16 or normalized float32 images"
            )
        if (
            not np.isfinite(image).all()
            or image.dtype == np.float32
            and (image.min() < 0 or image.max() > 1)
        ):
            raise ValidationError("Floating input must be finite in [0,1]")
        truth = binary(inputs["truth"], image.shape[:2])
        require_same_frame(
            [image, inputs["truth"]], message="Ground truth uses another source grid"
        )
        context = self.input_payloads["image"]
        a, b = context.get("ticket"), self.input_payloads["truth"].get("ticket")
        if a is not None and b is not None and ticket_key(a) != ticket_key(b):
            raise ValidationError("Ground truth belongs to another source ticket")
        h, w = image.shape[:2]
        transform = np.eye(3)
        transform[:2] = cv2.getRotationMatrix2D(
            ((w - 1) / 2, (h - 1) / 2), p["angle_degrees"], p["scale"]
        )
        transform[:2, 2] += [p["translation_x"], p["translation_y"]]
        maximum = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1
        interpolation = cv2.INTER_LINEAR if p["interpolation"] == "linear" else cv2.INTER_NEAREST
        values = np.asarray(image, np.float64) / maximum
        if not np.array_equal(transform, np.eye(3)):
            values = warp(values, transform, (w, h), interpolation, (p["border_value"],) * 3)
        values = np.clip(values * p["gain"] + p["offset"], 0, 1)
        support = warp(np.ones((h, w), np.float32), transform, (w, h), interpolation) >= 1 - 1e-6
        k = p["blur_kernel"]
        if k:
            values = cv2.GaussianBlur(
                values, (k, k), p["blur_sigma"], borderType=cv2.BORDER_CONSTANT
            )
            support = (
                cv2.erode(
                    support.astype(np.uint8),
                    np.ones((k, k), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                != 0
            )
        output = (
            np.rint(values * maximum).astype(image.dtype)
            if image.dtype.kind == "u"
            else values.astype(np.float32)
        )
        output = ImageArray(output, image_format(image))
        mapped_truth = warp(truth.astype(np.uint8), transform, (w, h)) != 0
        payload = moved_payload(context, np.linalg.inv(transform), (w, h))
        source_hash = hash_image(image)
        shift = {
            "model": SHIFT,
            "variant": p["variant"],
            "expected_pass": p["expected_pass"],
            "origin_id": sha({"image": source_hash, "ticket": list(ticket_key(a)) if a else None}),
            "source_sha256": source_hash,
            "output_sha256": hash_image(output),
            "image_size": [w, h],
            "reference_to_image": transform.tolist(),
            "source_truth_pixels": int(truth.sum()),
            "truth_sha256": hash_image(mapped_truth.astype(np.uint8) * 255),
            "support_sha256": hash_image(support.astype(np.uint8) * 255),
            "settings": {key: p[key] for key in p if key != "maximum_image_pixels"},
        }
        payload["evaluation_shift"] = shift
        outputs = {
            "image": output,
            "truth": mapped_truth.astype(np.uint8) * 255,
            "support": support.astype(np.uint8) * 255,
            "shift": shift,
        }
        return Result(
            outputs,
            payloads={key: copy.deepcopy(payload) for key in outputs},
            message=f"{p['variant']}; {int(support.sum())} measured-support pixels",
        )


class CaptureShiftObservation(_ShiftBlock):
    type_id, title = "ml.capture_shift_observation", "Capture shifted inspection result"
    inputs = (
        Port("image"),
        Port("prediction"),
        Port("truth"),
        Port("support"),
        Port("report", "data"),
        Port("model_support", required=False, required_when_connected=True),
        Port("score", "number", required=False, required_when_connected=True),
        Port("passed", "boolean", required=False, required_when_connected=True),
    )
    outputs = (Port("observation", "data"),)
    defaults = {
        "model_revision_path": "reference_revision",
        "decision_revision_path": "decision_revision",
        "score_kind": "raw",
        "minimum_truth_retention": 0.95,
    }
    parameter_choices = {"score_kind": ("raw", "probability")}
    parameter_ranges = {"minimum_truth_retention": NumericRange(0, 1)}
    parameter_help = {
        "model_revision_path": "Model/reference identity in Report, compared across shifts.",
        "decision_revision_path": "Threshold/decoder identity in Report, compared across shifts.",
        "score_kind": "Raw units, or declared [0,1] probability. This does not calibrate scores.",
        "minimum_truth_retention": "Reject pairs that crop/occlude too much known defect truth.",
    }
    input_help = {
        "passed": "Optional explicit judgment; otherwise use Prediction connection's tool outcome.",
        "image": "Unchanged shifted source, with its evaluation_shift metadata.",
        "model_support": "Optional model-specific scored support, intersected with shift support.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("model_revision_path", "decision_revision_path"):
            text(p[key], key, maximum=256)
            if len(p[key].split(".")) > 16 or any(not s for s in p[key].split(".")):
                raise ValidationError("Use up to 16 dotted revision fields", key)

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        shift = self.input_payloads["image"].get("evaluation_shift")
        if (
            not isinstance(shift, dict)
            or shift.get("model") != SHIFT
            or image.size > MAX_PIXELS * 3
        ):
            raise ValidationError("Connect a bounded controlled-shift image")
        if shift.get("output_sha256") != hash_image(image):
            raise ValidationError("Observation image differs from the controlled shifted pixels")
        for key in ("truth", "support"):
            if shift.get(key + "_sha256") != hash_image(inputs[key]):
                raise ValidationError("Observation " + key + " bytes differ from the shift")
        for key in ("prediction", "truth", "support", "report", "model_support", "score", "passed"):
            if key not in inputs:
                continue
            payload = self.input_payloads[key]
            if key != "passed" or "evaluation_shift" in payload:
                if payload.get("evaluation_shift") != shift:
                    raise ValidationError("Observation " + key + " belongs to another shift")
            a, b = self.input_payloads["image"].get("ticket"), payload.get("ticket")
            if a is not None and b is not None and ticket_key(a) != ticket_key(b):
                raise ValidationError("Observation " + key + " belongs to another ticket")
        masks = [
            inputs[key]
            for key in ("image", "prediction", "truth", "support", "model_support")
            if key in inputs
        ]
        require_same_frame(masks, message="Observation masks use another shifted grid")
        prediction, truth, support = (
            binary(inputs[key], image.shape[:2]) for key in ("prediction", "truth", "support")
        )
        if "model_support" in inputs:
            support &= binary(inputs["model_support"], image.shape[:2])
        if not np.any(support):
            raise ValidationError("Observation has no scored support")
        original_truth = shift["source_truth_pixels"]
        # Pixel-area change accounts for deliberate scale, but rasterization remains discrete.
        expected_truth_area = original_truth * abs(
            np.linalg.det(affine(shift["reference_to_image"])[:2, :2])
        )
        retention = (
            min(1.0, float(np.count_nonzero(truth & support) / expected_truth_area))
            if original_truth
            else None
        )
        if retention is not None and retention < p["minimum_truth_retention"]:
            raise ValidationError(
                "Insufficient retained/scored defect truth", "minimum_truth_retention"
            )
        outcome = self.input_payloads["prediction"].get("outcome", {})
        passed = inputs.get("passed", outcome.get("passed"))
        if type(passed) is not bool:
            raise ValidationError("Connect Passed or a prediction with a current tool outcome")
        model = text(output_path(inputs["report"], p["model_revision_path"]), "model revision")
        decision = text(
            output_path(inputs["report"], p["decision_revision_path"]), "decision revision"
        )
        score = inputs.get("score")
        if score is not None and (
            type(score) not in (int, float)
            or not np.isfinite(score)
            or abs(score) > 1e12
            or p["score_kind"] == "probability"
            and not 0 <= score <= 1
        ):
            raise ValidationError("Use a finite bounded score, or [0,1] probability", "score")
        observation = {
            "model": OBSERVATION,
            "shift": copy.deepcopy(shift),
            "passed": passed,
            "model_revision": model,
            "decision_revision": decision,
            "score": score,
            "score_kind": p["score_kind"],
            "truth_retention": retention,
            "prediction": pack(prediction),
            "truth": pack(truth),
            "support": pack(support),
            "source_ticket": copy.deepcopy(self.input_payloads["image"].get("ticket")),
        }
        return Result(
            {"observation": observation}, message=f"Captured {shift['variant']} inspection"
        )


class EvaluateInputShifts(_ShiftBlock):
    type_id, title = "ml.evaluate_input_shifts", "Evaluate inspection input shifts"
    inputs = (Port("observations", "any"),)
    outputs = (
        Port("report", "data"),
        Port("false_reject_rate", "number", required=False),
        Port("false_accept_rate", "number", required=False),
        Port("plot", required=False),
    )
    defaults = {
        "required_variants": ["baseline", "exposure", "blur", "pose"],
        "score_direction": "lower",
        "maximum_false_reject_rate": 0.0,
        "maximum_false_accept_rate": 0.0,
        "minimum_mean_truth_iou": 0.8,
        "minimum_mean_agreement_iou": 0.8,
        "minimum_pair_support_fraction": 0.5,
        "maximum_score_drop": -1.0,
        "maximum_total_pixels": 16_777_216,
    }
    parameter_choices = {"score_direction": ("lower", "higher")}
    parameter_ranges = {
        **{
            key: NumericRange(0, 1)
            for key in (
                "maximum_false_reject_rate",
                "maximum_false_accept_rate",
                "minimum_mean_truth_iou",
                "minimum_mean_agreement_iou",
                "minimum_pair_support_fraction",
            )
        },
        "maximum_score_drop": NumericRange(-1, 1e12),
        "maximum_total_pixels": NumericRange(1, 64_000_000, whole=True),
    }
    parameter_help = {
        "required_variants": "Exact variant names for each source; include baseline.",
        "maximum_score_drop": "Negative disables the gate. Direction defines worse scores.",
        "minimum_pair_support_fraction": "Minimum shared scored pixels / full source canvas.",
        "minimum_mean_truth_iou": "Empty unions have no IoU and are omitted from the mean.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        variants = p["required_variants"]
        if (
            not isinstance(variants, list)
            or not 1 <= len(variants) <= 16
            or "baseline" not in variants
            or any(not isinstance(v, str) or not 1 <= len(v) <= 64 for v in variants)
            or len(set(variants)) != len(variants)
        ):
            raise ValidationError(
                "Use 1..16 distinct variant names including baseline", "required_variants"
            )

    def process(self, inputs):
        p, data = self.parameters, inputs["observations"]
        rows = [item.value for item in data.items] if isinstance(data, FlowGroup) else data
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValidationError("Supply a bounded observation list or FlowGroup")
        groups, identities, seen, total = {}, set(), set(), 0
        for row in rows:
            if not isinstance(row, dict) or row.get("model") != OBSERVATION:
                raise ValidationError("Use captured input-shift observations")
            shift = row.get("shift", {})
            if shift.get("model") != SHIFT or not is_hash(shift.get("origin_id")):
                raise ValidationError("Invalid observation source identity")
            variant = text(shift.get("variant"), "variant", maximum=64)
            size = shift.get("image_size")
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(type(v) is not int or not 1 <= v <= 16384 for v in size)
                or size[0] * size[1] > MAX_PIXELS
            ):
                raise ValidationError("Observation exceeds source pixel/axis limits")
            total += size[0] * size[1]
            if total > p["maximum_total_pixels"]:
                raise ValidationError(
                    "Observation collection exceeds pixel budget", "maximum_total_pixels"
                )
            key = shift["origin_id"], variant
            if key in seen:
                raise ValidationError("Duplicate source/variant observation")
            seen.add(key)
            if type(row.get("passed")) is not bool or type(shift.get("expected_pass")) is not bool:
                raise ValidationError("Observation judgments must be boolean")
            score = row.get("score")
            if score is not None and (
                type(score) not in (int, float) or not np.isfinite(score) or abs(score) > 1e12
            ):
                raise ValidationError("Invalid observation score")
            if (
                row.get("score_kind") not in ("raw", "probability")
                or row["score_kind"] == "probability"
                and score is not None
                and not 0 <= score <= 1
            ):
                raise ValidationError("Observation probability/score kind differs")
            identities.add(
                (
                    text(row.get("model_revision"), "model revision"),
                    text(row.get("decision_revision"), "decision revision"),
                    row["score_kind"],
                )
            )
            shape = (size[1], size[0])
            decoded = {k: unpack(row.get(k), shape) for k in ("prediction", "truth", "support")}
            if not decoded["support"].any():
                raise ValidationError("Observation has no support")
            groups.setdefault(shift["origin_id"], {})[variant] = (
                row,
                decoded,
                affine(shift.get("reference_to_image")),
            )
        if len(identities) != 1:
            raise ValidationError(
                "Model/decision revision or score kind differs across observations"
            )
        results = []
        for origin, variants in groups.items():
            if set(variants) != set(p["required_variants"]):
                raise ValidationError(
                    "Missing/unexpected shift variants for a source", "required_variants"
                )
            baseline, base_masks, base_transform = variants["baseline"]
            if not np.array_equal(base_transform, np.eye(3)):
                raise ValidationError("Baseline reference transform must be identity")
            for name, (row, masks, transform) in variants.items():
                if (
                    row["shift"]["image_size"] != baseline["shift"]["image_size"]
                    or row["shift"]["expected_pass"] != baseline["shift"]["expected_pass"]
                    or row["shift"]["source_sha256"] != baseline["shift"]["source_sha256"]
                ):
                    raise ValidationError("Paired image size/label/source bytes differ")
                size = tuple(row["shift"]["image_size"])
                base_support = warp(base_masks["support"].astype(np.uint8), transform, size) != 0
                common = base_support & masks["support"]
                fraction = float(common.mean())
                if fraction < p["minimum_pair_support_fraction"] or not common.any():
                    raise ValidationError(
                        "Insufficient shared scored pair support", "minimum_pair_support_fraction"
                    )
                mapped_prediction = (
                    warp(base_masks["prediction"].astype(np.uint8), transform, size) != 0
                )
                truth_iou = iou(masks["prediction"], masks["truth"], masks["support"])
                agreement = iou(mapped_prediction, masks["prediction"], common)
                delta = (
                    row["score"] - baseline["score"]
                    if row["score"] is not None and baseline["score"] is not None
                    else None
                )
                worse = delta if p["score_direction"] == "lower" or delta is None else -delta
                expected, actual = row["shift"]["expected_pass"], row["passed"]
                results.append(
                    {
                        "origin_id": origin,
                        "variant": name,
                        "expected_pass": expected,
                        "passed": actual,
                        "false_reject": expected and not actual,
                        "false_accept": not expected and actual,
                        "truth_iou": truth_iou,
                        "baseline_agreement_iou": agreement,
                        "shared_support_fraction": fraction,
                        "score": row["score"],
                        "score_delta": delta,
                        "score_drop": worse,
                        "judgment_changed": actual != baseline["passed"],
                        "truth_retention": row["truth_retention"],
                    }
                )
        clean = sum(r["expected_pass"] for r in results)
        defective = len(results) - clean
        fr = sum(r["false_reject"] for r in results) / clean if clean else None
        fa = sum(r["false_accept"] for r in results) / defective if defective else None
        means = {
            key: float(np.mean([r[key] for r in results if r[key] is not None]))
            if any(r[key] is not None for r in results)
            else None
            for key in ("truth_iou", "baseline_agreement_iou")
        }
        drops = [r["score_drop"] for r in results if r["score_drop"] is not None]
        if p["maximum_score_drop"] >= 0 and len(drops) != len(results):
            raise ValidationError("Score-drop gating requires all current scores")
        failures = []
        for code, actual, limit, lower in (
            ("false_rejects", fr, p["maximum_false_reject_rate"], False),
            ("false_accepts", fa, p["maximum_false_accept_rate"], False),
            ("truth_iou", means["truth_iou"], p["minimum_mean_truth_iou"], True),
            ("agreement", means["baseline_agreement_iou"], p["minimum_mean_agreement_iou"], True),
            (
                "score_drop",
                max(drops) if drops and p["maximum_score_drop"] >= 0 else None,
                p["maximum_score_drop"],
                False,
            ),
        ):
            if actual is not None and (actual < limit if lower else actual > limit):
                failures.append(
                    {
                        "code": "input_shift." + code,
                        "message": f"{code} {actual:.5g} violates {limit:.5g}",
                    }
                )
        summary = {
            "source_count": len(groups),
            "observation_count": len(results),
            "false_reject_rate": fr,
            "false_accept_rate": fa,
            **means,
            "maximum_score_drop": max(drops) if drops else None,
        }
        by_variant = {
            name: {
                "observations": sum(r["variant"] == name for r in results),
                "false_rejects": sum(r["variant"] == name and r["false_reject"] for r in results),
                "false_accepts": sum(r["variant"] == name and r["false_accept"] for r in results),
            }
            for name in p["required_variants"]
        }
        for name, counts in by_variant.items():
            subset = [row for row in results if row["variant"] == name]
            normal_count = sum(row["expected_pass"] for row in subset)
            defect_count = len(subset) - normal_count
            counts["false_reject_rate"] = (
                counts["false_rejects"] / normal_count if normal_count else None
            )
            counts["false_accept_rate"] = (
                counts["false_accepts"] / defect_count if defect_count else None
            )
            for key in ("score", "score_delta", "truth_iou", "baseline_agreement_iou"):
                available = [row[key] for row in subset if row[key] is not None]
                counts["mean_" + key] = float(np.mean(available)) if available else None
        report = {
            "model": "inspection_input_shift_report_v1",
            "summary": summary,
            "by_variant": by_variant,
            "records": results,
            "model_revision": next(iter(identities))[0],
            "decision_revision": next(iter(identities))[1],
            "score_kind": next(iter(identities))[2],
            "evaluation_revision": sha(p),
            "capture_independence_verified": False,
            "generated_shifts_are_new_captures": False,
            "complete": True,
        }
        plot = NO_VALUE
        if self.generate_graphics:
            plot = np.full((120 + 26 * len(by_variant), 680, 3), 245, np.uint8)
            draw_unicode(
                plot,
                "Input shifts: rejects / accepts / label count",
                (10, 22),
                (30, 30, 30),
                TextStyle(font_size=14),
            )
            for index, (name, counts) in enumerate(by_variant.items()):
                label = (
                    f"{name}: {counts['false_rejects']} / "
                    f"{counts['false_accepts']} / {counts['observations']}"
                )
                draw_unicode(
                    plot, label, (10, 52 + 26 * index), (40, 40, 40), TextStyle(font_size=13)
                )
        context = copy.deepcopy(self.input_payloads.get("observations", {}))
        for key in ("spatial", "pyramid", "evaluation_shift"):
            context.pop(key, None)
        outputs = {
            "report": report,
            "false_reject_rate": NO_VALUE if fr is None else fr,
            "false_accept_rate": NO_VALUE if fa is None else fa,
            "plot": plot,
        }
        return Result(
            outputs,
            passed=not failures,
            fail_reasons=failures,
            payloads={key: copy.deepcopy(context) for key in outputs},
            graphics={"input shifts": plot} if plot is not NO_VALUE else {},
            message=f"{len(groups)} sources, {len(results)} observations; "
            f"{len(failures)} failed limits",
        )


INPUT_SHIFT_BLOCKS = (ImageInputShift, CaptureShiftObservation, EvaluateInputShifts)
