"""Bounded same-image tensor fidelity checks for quantized or alternate model exports."""

import copy
import hashlib

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, require_same_frame
from .processing import image_value
from .tickets import ticket_key


def read_tensor(tensors, name, limit):
    if not isinstance(tensors, dict) or not tensors:
        raise ValidationError("Connect a nonempty named tensor_v1 map")
    name = name or next(iter(tensors))
    record = tensors.get(name)
    if not isinstance(record, dict) or record.get("dtype") != "float32":
        raise ValidationError("Compare explicitly dequantized float32 tensor_v1 outputs")
    values = record.get("values")
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= limit
        or any(type(v) not in (int, float) for v in values)
    ):
        raise ValidationError("Tensor values require a bounded flat numeric list")
    value = selected_tensor(tensors, name, max_elements=limit).astype(np.float64)
    return name, value


class CompareImageTensors(Block):
    type_id, title = "ml.compare_image_tensors", "Compare image model tensors"
    category = block_type = "ML"
    subtype = "Model qualification"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("image"), Port("reference", "data"), Port("candidate", "data"))
    outputs = (
        Port("image"),
        Port("report", "data"),
        Port("maximum_error", "number"),
        Port("rmse", "number"),
        Port("top1_equal", "boolean", required=False),
    )
    defaults = {
        "reference_output_name": "",
        "candidate_output_name": "",
        "task": "classification",
        "comparison_space": "native_values",
        "absolute_tolerance": 0.05,
        "relative_tolerance": 0.01,
        "require_top1": True,
        "top_k": 5,
        "minimum_topk_overlap": 0.0,
        "plot_items": 10,
        "maximum_elements": 2_000_000,
    }
    parameter_choices = {
        "task": ("classification", "tensor"),
        "comparison_space": ("native_values", "softmax"),
    }
    parameter_ranges = {
        "absolute_tolerance": NumericRange(0, 1e12),
        "relative_tolerance": NumericRange(0, 1e6),
        "top_k": NumericRange(1, 1000, whole=True),
        "minimum_topk_overlap": NumericRange(0, 1),
        "plot_items": NumericRange(0, 50, whole=True),
        "maximum_elements": NumericRange(1, 2_000_000, whole=True),
    }
    parameter_help = {
        "reference_output_name": "Exact reference tensor name; blank selects its first output.",
        "candidate_output_name": "Exact candidate tensor name; shapes must match the reference.",
        "task": "Classification accepts [C] or [1,C]; Tensor compares arbitrary matching shapes.",
        "comparison_space": (
            "Native values, or independent softmax of classification logits. Not "
            "calibrated confidence."
        ),
        "absolute_tolerance": (
            "Every element must satisfy abs(candidate-reference) <= atol + rtol*abs(reference)."
        ),
        "relative_tolerance": (
            "Relative tolerance uses reference magnitude, not candidate magnitude."
        ),
        "require_top1": (
            "Classification only: also require the same top class; stable ties use lower index."
        ),
        "top_k": "Classification overlap uses min(top_k,class count); tensor mode ignores it.",
        "minimum_topk_overlap": (
            "Inclusive shared fraction of top-k class indices; zero disables that criterion."
        ),
        "plot_items": (
            "Optional bounded comparison bars; zero disables the plot. Graphics/Lean also apply."
        ),
        "maximum_elements": "Reject larger tensors before conversion; no sampling or truncation.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric comparison settings", key)
        for key in ("reference_output_name", "candidate_output_name"):
            if not isinstance(p[key], str) or len(p[key]) > 256:
                raise ValidationError("Output name must be text up to 256 characters", key)
        if type(p["require_top1"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "require_top1")
        if p["task"] == "tensor" and p["comparison_space"] != "native_values":
            raise ValidationError(
                "Softmax comparison requires Classification task", "task", "comparison_space"
            )

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        contexts = [self.input_payloads.get(k, {}) for k in ("image", "reference", "candidate")]
        for c in contexts:
            check_spatial(c, list(image.shape[1::-1]))
        require_same_frame(
            [ValuePacket(None, c) for c in contexts],
            message="Tensor comparison uses another image frame",
        )
        for key in ("ticket", "recipe_ref", "pyramid"):
            values = [
                ticket_key(c[key]) if key == "ticket" else c[key] for c in contexts if key in c
            ]
            if values and any(v != values[0] for v in values[1:]):
                raise ValidationError(f"Tensor comparison {key} contexts disagree")
        roots = [c["spatial"]["root_size"] for c in contexts if "spatial" in c]
        if roots and any(v != roots[0] for v in roots[1:]):
            raise ValidationError("Tensor comparison root dimensions disagree")
        rn, reference = read_tensor(
            inputs["reference"], p["reference_output_name"], p["maximum_elements"]
        )
        cn, candidate = read_tensor(
            inputs["candidate"], p["candidate_output_name"], p["maximum_elements"]
        )
        if reference.shape != candidate.shape:
            raise ValidationError("Reference and candidate tensor shapes must match")
        raw_reference, raw_candidate = reference, candidate
        classification = p["task"] == "classification"
        if classification:
            if reference.ndim not in (1, 2) or (reference.ndim == 2 and reference.shape[0] != 1):
                raise ValidationError("Classification comparison requires [C] or [1,C]")
            if p["comparison_space"] == "softmax":

                def softmax(x):
                    e = np.exp(x - x.max())
                    return e / e.sum()

                reference, candidate = softmax(reference), softmax(candidate)
        difference = (candidate - reference).reshape(-1)
        absolute = np.abs(difference)
        allowed = p["absolute_tolerance"] + p["relative_tolerance"] * np.abs(reference.reshape(-1))
        within = absolute <= allowed
        maximum = float(absolute.max())
        rmse = float(np.sqrt(np.mean(difference * difference)))
        reasons = []
        if not within.all():
            reasons.append(
                {
                    "code": "tensor_comparison.values",
                    "message": "Tensor values exceed configured tolerance",
                }
            )
        class_report = None
        if classification:
            ro = np.argsort(-reference.reshape(-1), kind="stable")
            co = np.argsort(-candidate.reshape(-1), kind="stable")
            k = min(p["top_k"], reference.size)
            equal = bool(ro[0] == co[0])
            overlap = len(set(ro[:k]) & set(co[:k])) / k
            class_report = {
                "reference_top1": int(ro[0]),
                "candidate_top1": int(co[0]),
                "top1_equal": equal,
                "top_k": k,
                "reference_topk": ro[:k].tolist(),
                "candidate_topk": co[:k].tolist(),
                "topk_overlap": overlap,
            }
            if p["require_top1"] and not equal:
                reasons.append({"code": "tensor_comparison.top1", "message": "Top class changed"})
            if overlap < p["minimum_topk_overlap"]:
                reasons.append(
                    {"code": "tensor_comparison.topk", "message": "Top-k overlap below minimum"}
                )
        report = {
            "model": "image_tensor_comparison_v1",
            "task": p["task"],
            "space": p["comparison_space"],
            "reference_name": rn,
            "candidate_name": cn,
            "shape": list(reference.shape),
            "elements": reference.size,
            "maximum_absolute_error": maximum,
            "rmse": rmse,
            "mean_absolute_error": float(absolute.mean()),
            "bias": float(difference.mean()),
            "within_tolerance_fraction": float(within.mean()),
            "violating_elements": int((~within).sum()),
            "worst_flat_index": int(absolute.argmax()),
            "classification": class_report,
            "absolute_tolerance": p["absolute_tolerance"],
            "relative_tolerance": p["relative_tolerance"],
            "reference_float32_sha256": hashlib.sha256(
                raw_reference.astype("<f4").tobytes()
            ).hexdigest(),
            "candidate_float32_sha256": hashlib.sha256(
                raw_candidate.astype("<f4").tobytes()
            ).hexdigest(),
            "model_identity_verified": False,
            "context_checks": {
                "ticket_checked": all("ticket" in c for c in contexts),
                "frame_checked": all(
                    c.get("spatial", {}).get("root_id") is not None for c in contexts
                ),
            },
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            color = (60, 230, 120) if not reasons else (70, 90, 230)
            drawings = {
                "Model tensor fidelity": GraphicDrawing(
                    (
                        graphic_object(
                            "label",
                            [[8, 24]],
                            color,
                            text=(
                                f"{'PASS' if not reasons else 'FAIL'} tensor difference "
                                f"max {maximum:.5g}"
                            ),
                        ),
                    )
                )
            }
            if p["plot_items"]:
                order = (ro if classification else np.argsort(-absolute, kind="stable"))[
                    : p["plot_items"]
                ]
                a, b = reference.reshape(-1)[order], candidate.reshape(-1)[order]
                chart = np.full((65 + 28 * len(order), 900, 3), 245, np.uint8)
                peak = max(1e-12, float(np.abs(np.r_[a, b]).max()))
                cv2.putText(
                    chart,
                    "Reference / Candidate (signed values)",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 40, 40),
                    1,
                )
                for i, (index, x, y) in enumerate(zip(order, a, b, strict=True)):
                    row = 55 + 28 * i
                    cv2.putText(
                        chart, str(index), (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1
                    )
                    for shift, value, c in ((-10, x, (180, 110, 40)), (0, y, (70, 160, 100))):
                        end = 460 + round(310 * value / peak)
                        cv2.rectangle(
                            chart,
                            (min(460, end), row + shift),
                            (max(460, end), row + shift + 6),
                            c,
                            -1,
                        )
                    cv2.putText(
                        chart,
                        f"{x:.4g} / {y:.4g}",
                        (790, row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (40, 40, 40),
                        1,
                    )
                graphics["Tensor comparison"] = chart
        outputs = {
            "image": inputs["image"],
            "report": report,
            "maximum_error": maximum,
            "rmse": rmse,
        }
        outputs["top1_equal"] = class_report["top1_equal"] if classification else NO_VALUE
        return Result(
            outputs,
            payloads={k: copy.deepcopy(contexts[0]) for k in outputs},
            passed=not reasons,
            score=float(within.mean()),
            fail_reasons=reasons,
            message=f"Max error {maximum:.5g}; RMSE {rmse:.5g}",
            graphics=graphics,
            drawings=drawings,
        )
