"""Paired variational flow refinement with explicit source grids and support diagnostics."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .flow_warp import MAX_PIXELS, flow_field, inside, same_grid, sample_mask
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, require_same_frame
from .processing import gray_image, image_value
from .selection import SELECTION_INPUTS, selection_mask
from .sequence_tools import DenseFlow

SOLVER_CONTROLS = {
    "fixed_point_iterations": ("setFixedPointIterations", int),
    "sor_iterations": ("setSorIterations", int),
    "alpha": ("setAlpha", float),
    "delta": ("setDelta", float),
    "gamma": ("setGamma", float),
    "omega": ("setOmega", float),
    "epsilon": ("setEpsilon", float),
}
SOURCE_KEYS = ("ticket", "spatial", "pyramid", "video_frame", "recipe_ref")


def solver(parameters):
    native = cv2.VariationalRefinement_create()
    for name, (setter, convert) in SOLVER_CONTROLS.items():
        getattr(native, setter)(convert(parameters[name]))
    return native


def mean_on(values, selected):
    return float(values[selected].mean()) if np.any(selected) else None


class RefineDenseFlow(Block):
    type_id, title = "cv.refine_flow", "Refine dense optical flow"
    category = block_type = "Image processing"
    subtype = "Video processing"
    inputs = (
        *SELECTION_INPUTS,
        Port("target"),
        Port("flow", "any"),
        Port("source_support", required=False, required_when_connected=True),
        Port("target_support", required=False, required_when_connected=True),
    )
    outputs = (
        Port("flow", "any"),
        Port("support", visual_type="region"),
        Port("residual"),
        Port("initial_residual"),
        Port("magnitude", required=False),
        Port("angle", required=False),
        Port("visualization", required=False),
        Port("report", "data"),
    )
    flow_primary = ("flow",)
    defaults = {
        "fixed_point_iterations": 5,
        "sor_iterations": 5,
        "alpha": 20.0,
        "delta": 5.0,
        "gamma": 10.0,
        "omega": 1.6,
        "epsilon": 0.001,
        "alignment": "same_frame",
        "border_margin": 4,
        "maximum_motion": 256.0,
        "maximum_work": 64_000_000,
        "minimum_valid_fraction": 0.0,
        "maximum_mean_residual": 0.0,
        "reject_residual_increase": False,
        "visual_max_motion": 10.0,
        "arrow_spacing": 16,
    }
    parameter_choices = {"alignment": ("same_frame", "pixel_grid")}
    parameter_ranges = {
        "fixed_point_iterations": NumericRange(1, 50, whole=True),
        "sor_iterations": NumericRange(1, 100, whole=True),
        **{name: NumericRange(0, 10000) for name in ("alpha", "delta", "gamma")},
        "omega": NumericRange(0.001, 1.999),
        "epsilon": NumericRange(0.000001, 10),
        "border_margin": NumericRange(0, 64, whole=True),
        "maximum_motion": NumericRange(1, 16384),
        "maximum_work": NumericRange(1, 256_000_000, whole=True),
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_mean_residual": NumericRange(0, 255),
        "visual_max_motion": NumericRange(0.001, 16384),
        "arrow_spacing": NumericRange(8, 128, whole=True),
    }
    parameter_help = {
        "fixed_point_iterations": "Native outer iterations; this is a local, single-scale solver.",
        "sor_iterations": "Native inner successive over-relaxation iterations per outer step.",
        "alpha": "Native smoothness weight; zero disables the smoothness term.",
        "delta": "Native brightness-constancy weight; delta or gamma must be positive.",
        "gamma": "Native gradient-constancy weight; delta or gamma must be positive.",
        "omega": "Native SOR relaxation factor, strictly between zero and two.",
        "epsilon": "Native robust-penalty norm shift, not a convergence threshold.",
        "alignment": "Check known coordinate frames, or explicitly pair matching pixel grids.",
        "border_margin": "Exclude source/donor edge bands and support holes from diagnostics.",
        "maximum_motion": "Maximum per-axis seed/output displacement in local pixels.",
        "maximum_work": "Bound pixels × (outer × inner + 3) before invoking the solver.",
        "minimum_valid_fraction": "Valid/selected fraction gate; no valid support always fails.",
        "maximum_mean_residual": "Maximum mean supported gray difference [0,255]; zero disables.",
        "reject_residual_increase": "Reject higher mean residual on common before/after support.",
        "visual_max_motion": "Motion mapped to full brightness in the optional HSV flow view.",
        "arrow_spacing": "Minimum preview spacing; large frames cap the arrow count at 2048.",
    }
    input_help = {
        "image": "Source A: uint8 gray/color, 8+ pixels per axis and at most 2.1M pixels.",
        "target": "Target B on an equal-size grid; sampled at source + flow.",
        "flow": "Finite float32/64 H×W×2 A-to-B seed, indexed by A; copied before refinement.",
        "mask": "Source-grid output selection; intersected with region, not a solver cost mask.",
        "region": "Source-grid output selection; full-frame context still feeds the solver.",
        "source_support": "Trusted uint8 A support; restricts diagnostics, not the native energy.",
        "target_support": "Trusted uint8 B support sampled at the displaced donor position.",
    }
    process_inputs_readonly = parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "target", "flow", "mask", "region"]
        self.exposed_outputs = ["flow", "support", "residual", "report"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not float(p["delta"]) and not float(p["gamma"]):
            raise ValidationError("Delta or gamma must be positive", "delta", "gamma")
        if type(p["reject_residual_increase"]) is not bool:
            raise ValidationError(
                "Reject residual increase must be boolean", "reject_residual_increase"
            )

    def trusted(self, inputs, name, shape, context):
        if name not in inputs:
            value = np.ones(shape, np.uint8)
        else:
            value = inputs[name]
            if value.dtype != np.uint8 or value.shape != shape:
                raise ValidationError(f"{name} must be matching uint8 monochrome support")
            same_grid(self.input_payloads[name], context, name)
            value = (np.asarray(value) != 0).astype(np.uint8)
        margin = int(self.parameters["border_margin"])
        if margin:
            value = cv2.erode(
                value,
                np.ones((2 * margin + 1, 2 * margin + 1), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return value != 0

    def process(self, inputs):
        p = {
            **self.parameters,
            **{
                name: rule.check(self.parameters[name], name, name)
                for name, rule in self.parameter_ranges.items()
            },
        }
        source, target = inputs["image"], inputs["target"]
        image_value(source, (np.uint8,))
        image_value(target, (np.uint8,))
        h, w = source.shape[:2]
        if min(h, w) < 8 or max(h, w) > 16384 or h * w > MAX_PIXELS:
            raise ValidationError("Use 8..16384 per axis and at most 2.1M pixels")
        if target.shape[:2] != (h, w):
            raise ValidationError("Source and target image dimensions must match")
        if h * w * (p["fixed_point_iterations"] * p["sor_iterations"] + 3) > p["maximum_work"]:
            raise ValidationError("Refinement exceeds maximum_work", "maximum_work")
        if p["alignment"] == "same_frame":
            try:
                require_same_frame(
                    [
                        ValuePacket(source, self.input_payloads["image"]),
                        ValuePacket(target, self.input_payloads["target"]),
                    ]
                )
            except ValueError as exc:
                raise ValidationError(
                    "Source/target frames differ; align or choose pixel_grid"
                ) from exc
        source_context = self.input_payloads["image"]
        target_context = self.input_payloads["target"]
        try:
            for context in (source_context, target_context):
                check_spatial(context, [w, h])
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        same_grid(self.input_payloads["flow"], source_context, "flow")
        initial, finite = flow_field(
            inputs["flow"], (h, w), p["maximum_motion"], "flow", self.input_payloads["flow"]
        )
        if not finite.all():
            raise ValidationError("Seed vectors must be finite and within maximum_motion")
        a, b = [np.ascontiguousarray(gray_image(im, (np.uint8,))) for im in (source, target)]
        selected = selection_mask(inputs, source)
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        source_ok = self.trusted(inputs, "source_support", (h, w), source_context)
        target_ok = self.trusted(inputs, "target_support", (h, w), target_context)
        refined = solver(p).calc(a, b, initial.copy()) if selected.any() else initial.copy()
        if refined.shape != initial.shape or not np.isfinite(refined).all():
            raise ValidationError("Native refinement returned invalid vectors")
        if np.any(np.abs(refined) > p["maximum_motion"]):
            raise ValidationError("Native refinement exceeds maximum_motion", "maximum_motion")
        refined[~selected] = initial[~selected]
        yy, xx = np.indices((h, w), dtype=np.float32)

        def diagnostics(field):
            x, y = xx + field[..., 0], yy + field[..., 1]
            valid = selected & source_ok & inside(x, y, (h, w))
            valid &= (np.abs(field) <= p["maximum_motion"]).all(axis=2)
            # Clamp only diagnostic remap coordinates; out-of-range donors never gain support.
            safe_x, safe_y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
            valid &= sample_mask(target_ok, safe_x, safe_y)
            sampled = cv2.remap(b, safe_x, safe_y, cv2.INTER_LINEAR)
            residual = np.abs(a.astype(np.float32) - sampled.astype(np.float32))
            residual[~valid] = np.nan
            return valid, residual

        before_ok, before = diagnostics(initial)
        valid, after = diagnostics(refined)
        common = valid & before_ok
        count = int(valid.sum())
        fraction = count / max(1, int(selected.sum()))
        before_mean, after_mean = mean_on(before, common), mean_on(after, common)
        current_mean = mean_on(after, valid)
        reason = None
        if not count:
            reason = "no_valid_support"
        elif fraction < p["minimum_valid_fraction"]:
            reason = "insufficient_support"
        elif p["maximum_mean_residual"] and current_mean > p["maximum_mean_residual"]:
            reason = "high_residual"
        elif p["reject_residual_increase"]:
            if before_mean is None:
                reason = "no_comparable_support"
            elif after_mean > before_mean:
                reason = "residual_increased"
        pair = {
            "direction": "source_to_target",
            "source": {
                k: copy.deepcopy(source_context[k]) for k in SOURCE_KEYS if k in source_context
            },
            "target": {
                k: copy.deepcopy(target_context[k]) for k in SOURCE_KEYS if k in target_context
            },
        }
        context = copy.deepcopy(source_context)
        context.pop("alpha_representation", None)
        context.pop("raw_sensor", None)
        context["flow_refinement"] = pair
        report = {
            "model": "flow_refinement_report_v1",
            "direction": "source_to_target",
            "units": "local_source_pixels",
            "selected_pixels": int(selected.sum()),
            "valid_pixels": count,
            "valid_fraction": fraction,
            "comparable_pixels": int(common.sum()),
            "initial_mean_residual": before_mean,
            "refined_mean_residual": after_mean,
            "supported_mean_residual": current_mean,
            "residual_units": "uint8_gray_difference",
            "mean_update_pixels": mean_on(np.linalg.norm(refined - initial, axis=2), valid),
            "reason": reason,
            "pair": pair,
            "support_semantics": "source/donor support; not visibility or accuracy confidence",
        }
        magnitude, angle, view = NO_VALUE, NO_VALUE, NO_VALUE
        if self.generate_graphics or {"magnitude", "angle", "visualization"} & set(
            self.exposed_outputs
        ):
            magnitude, angle = cv2.cartToPolar(
                refined[..., 0], refined[..., 1], angleInDegrees=True
            )
            magnitude[~valid], angle[~valid] = np.nan, np.nan
            if self.generate_graphics or "visualization" in self.exposed_outputs:
                hsv = np.zeros((h, w, 3), np.uint8)
                hsv[..., 0][valid] = np.rint(angle[valid] / 2).astype(np.uint8) % 180
                hsv[..., 1][valid] = 255
                hsv[..., 2][valid] = np.clip(
                    magnitude[valid] * 255 / p["visual_max_motion"], 0, 255
                ).astype(np.uint8)
                view = ImageArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "bgr")
        drawings = {}
        if self.generate_graphics:
            color = (60, 200, 100) if reason is None else (40, 70, 240)
            objects = [
                graphic_object(
                    "text",
                    [(5, min(18, h - 1))],
                    color,
                    text=f"Refine flow: {reason or 'pass'}; {fraction:.1%} support",
                )
            ]
            spacing = max(int(p["arrow_spacing"]), math.ceil(math.sqrt(h * w / 2048)))
            while (
                len(range(spacing // 2, h, spacing)) * len(range(spacing // 2, w, spacing)) > 2048
            ):
                spacing += 1
            if reason is None:
                for y in range(spacing // 2, h, spacing):
                    for x in range(spacing // 2, w, spacing):
                        if valid[y, x]:
                            objects.append(
                                graphic_object(
                                    "arrow",
                                    [(x, y), (x + refined[y, x, 0], y + refined[y, x, 1])],
                                    color,
                                    thickness=1,
                                )
                            )
            drawings["refined motion"] = GraphicDrawing(tuple(objects))
        outputs = {
            "flow": DenseFlow(refined, context) if reason is None else NO_VALUE,
            "support": ImageArray(valid.astype(np.uint8) * 255, "gray"),
            "residual": ImageArray(after, "gray"),
            "initial_residual": ImageArray(before, "gray"),
            "magnitude": magnitude,
            "angle": angle,
            "visualization": view,
            "report": report,
        }
        return Result(
            outputs,
            passed=reason is None,
            score=fraction,
            payloads={
                name: copy.deepcopy(context)
                for name, value in outputs.items()
                if value is not NO_VALUE
            },
            drawings=drawings,
            fail_reasons=[]
            if reason is None
            else [{"code": "flow_refinement_" + reason, "message": reason.replace("_", " ")}],
            message=f"{fraction:.1%} support · " + (reason or "refined source-to-target flow"),
        )
