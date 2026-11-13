"""Bounded bidirectional frame interpolation with conservative motion support."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata, require_same_frame
from .processing import gray_image, image_value
from .selection import selection_mask
from .sequence_tools import DenseFlow, FarnebackSequence, _flow_consistency, farneback_calculator

INTERPOLATION = {
    "nearest": (cv2.INTER_NEAREST, 0),
    "linear": (cv2.INTER_LINEAR, 0),
    "cubic": (cv2.INTER_CUBIC, 2),
    "lanczos4": (cv2.INTER_LANCZOS4, 4),
}
FLOW_KEYS = (
    "pyramid_scale",
    "levels",
    "window",
    "iterations",
    "poly_n",
    "poly_sigma",
    "gaussian_window",
)
SOURCE_KEYS = ("ticket", "spatial", "pyramid", "video_frame", "recipe_ref")


def interpolation_time(inputs, a_context, b_context, p):
    explicit = "time_a_ms" in inputs or "time_b_ms" in inputs
    a, b = a_context.get("video_frame", {}), b_context.get("video_frame", {})
    root = a_context.get("spatial", {}).get("root_id")
    same_stream = (
        bool(a.get("filename"))
        and a.get("filename") == b.get("filename")
        and root is not None
        and root == b_context.get("spatial", {}).get("root_id")
    )
    source = "unavailable"
    times = None
    if explicit:
        if "time_a_ms" not in inputs or "time_b_ms" not in inputs:
            raise ValidationError("Supply both frame times in the same clock domain")
        times, source = [inputs["time_a_ms"], inputs["time_b_ms"]], "explicit"
    elif p["timing_source"] == "explicit":
        times, source = [p["time_a_ms"], p["time_b_ms"]], "explicit_parameters"
    elif same_stream and a.get("timestamp_ms") is not None and b.get("timestamp_ms") is not None:
        times, source = [a["timestamp_ms"], b["timestamp_ms"]], "video_frame"
    if times is not None:
        if any(
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, float, np.integer, np.floating))
            for t in times
        ):
            raise ValidationError("Frame times must be finite nonnegative milliseconds")
        try:
            times = [float(t) for t in times]
        except OverflowError:
            raise ValidationError("Frame times exceed the finite range") from None
        if any(not math.isfinite(t) or t < 0 for t in times):
            raise ValidationError("Frame times must be finite nonnegative milliseconds")
        if times[1] <= times[0]:
            raise ValidationError("Frame B time must be later than frame A time")
        t = float(p["fraction"])
        timestamp = (1 - t) * times[0] + t * times[1]
        if not math.isfinite(timestamp):
            raise ValidationError("Interpolated frame time exceeds the finite range")
    else:
        timestamp = None
        if p["timing_policy"] == "required":
            raise ValidationError(
                "Compatible source media times or explicit frame times are required"
            )
    return timestamp, times, source, same_stream


def flow_value(value, image, maximum, name):
    raw = value.vectors if isinstance(value, DenseFlow) else value
    if (
        not isinstance(raw, np.ndarray)
        or raw.shape != (*image.shape[:2], 2)
        or raw.dtype not in (np.float32, np.float64)
        or not np.isfinite(raw).all()
    ):
        raise ValidationError(f"{name} needs finite float32/64 HxWx2 displacement vectors")
    if np.any(np.abs(raw) > maximum):
        raise ValidationError(f"{name} exceeds the configured maximum per-axis displacement")
    require_same_frame([value, image], message=f"{name} does not use its source image's frame")
    return np.asarray(raw, np.float32).copy()


def support_value(value, image, name):
    if value is None:
        return None
    if value.dtype != np.uint8 or value.shape != image.shape[:2]:
        raise ValidationError(f"{name} must be a matching uint8 monochrome support mask")
    require_same_frame([value, image], message=f"{name} does not use its source image's frame")
    return np.asarray(value) != 0


def inverse_map(flow, amount, source_support, grid_x, grid_y, p):
    """Solve source + amount*flow(source) = target, then check the residual/support."""
    x, y = grid_x.copy(), grid_y.copy()
    for _ in range(int(p["inverse_iterations"])):
        sampled = cv2.remap(flow, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y = grid_x - amount * sampled[..., 0], grid_y - amount * sampled[..., 1]
    sampled = cv2.remap(flow, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    residual = cv2.magnitude(
        x + amount * sampled[..., 0] - grid_x, y + amount * sampled[..., 1] - grid_y
    )
    margin = INTERPOLATION[p["interpolation"]][1]
    height, width = flow.shape[:2]
    inside = (x >= margin) & (x <= width - 1 - margin) & (y >= margin) & (y <= height - 1 - margin)
    support = cv2.remap(
        source_support.astype(np.float32) / 255,
        x,
        y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = inside & (support >= 1 - 1e-6) & (residual <= float(p["inverse_tolerance"]))
    return x, y, valid


def blend_supported(a, b, valid_a, valid_b, fraction):
    wa, wb = (1 - fraction) * valid_a, fraction * valid_b
    denominator = wa + wb
    weight = np.divide(wb, denominator, out=np.zeros_like(wb), where=denominator > 0)
    if a.ndim == 3:
        weight = weight[..., None]
        valid_a, valid_b = valid_a[..., None], valid_b[..., None]
    # Work in float64 to avoid uint16 rounding or float32 intermediate overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        output = (
            np.where(valid_a, a, 0).astype(np.float64) * (1 - weight)
            + np.where(valid_b, b, 0).astype(np.float64) * weight
        )
    if not np.isfinite(output[denominator > 0]).all():
        raise ValidationError("Interpolated pixels overflowed; rescale the floating-point images")
    if a.dtype.kind in "ui":
        limits = np.iinfo(a.dtype)
        output = np.clip(np.rint(output), limits.min, limits.max)
    return output.astype(a.dtype), denominator > 0


class InterpolateFrames(Block):
    type_id, title = "cv.interpolate_frames", "Motion-compensated frame interpolation"
    category, block_type, subtype = "Image processing", "Image processing", "Video processing"
    payload_reference = "image_a"
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("forward", "any", required=False, required_when_connected=True),
        Port("backward", "any", required=False, required_when_connected=True),
        Port("support_a", required=False, required_when_connected=True),
        Port("support_b", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("time_a_ms", "number", required=False, required_when_connected=True),
        Port("time_b_ms", "number", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("coverage"),
        Port("info", "data"),
        Port("timestamp_ms", "number", required=False),
        Port("coverage_a"),
        Port("coverage_b"),
        Port("uncertain"),
        Port("forward_flow", "any", required=False),
        Port("backward_flow", "any", required=False),
        Port("coverage_preview", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "fraction": 0.5,
        "flow_mode": "estimate",
        "alignment": "same_frame",
        "interpolation": "linear",
        "roundtrip_tolerance": 1.0,
        "appearance_tolerance": 0.0,
        "inverse_iterations": 6,
        "inverse_tolerance": 0.25,
        "maximum_motion": 256.0,
        "uncovered": "zero",
        "minimum_coverage": 0.0,
        "timing_policy": "if_available",
        "timing_source": "media",
        "time_a_ms": 0.0,
        "time_b_ms": 40.0,
        "emit_flows": False,
        **{k: FarnebackSequence.defaults[k] for k in FLOW_KEYS},
    }
    parameter_choices = {
        "flow_mode": ("estimate", "refine", "provided"),
        "alignment": ("same_frame", "pixel_grid"),
        "interpolation": tuple(INTERPOLATION),
        "uncovered": ("zero", "nearest_endpoint", "crossfade"),
        "timing_policy": ("if_available", "required"),
        "timing_source": ("media", "explicit"),
        "poly_n": ("5", "7"),
    }
    parameter_ranges = {
        "fraction": NumericRange(0, 1),
        "roundtrip_tolerance": NumericRange(0, 100),
        "appearance_tolerance": NumericRange(0, 1),
        "inverse_iterations": NumericRange(1, 20, whole=True),
        "inverse_tolerance": NumericRange(0, 10),
        "maximum_motion": NumericRange(0.01, 4096),
        "minimum_coverage": NumericRange(0, 1),
        "time_a_ms": NumericRange(0),
        "time_b_ms": NumericRange(0),
        "pyramid_scale": NumericRange(0.01, 0.99),
        "levels": NumericRange(1, 8, whole=True),
        "window": NumericRange(3, 255, whole=True, odd=True),
        "iterations": NumericRange(1, 100, whole=True),
        "poly_sigma": NumericRange(0.01, 100),
    }
    parameter_help = {
        "fraction": (
            "Time fraction A to B. Zero/one copy exact selected endpoint pixels without flow."
        ),
        "flow_mode": (
            "Estimate Farneback, refine wired initial fields, or use provided fields directly."
        ),
        "alignment": (
            "Same frame checks coordinate identity. Pixel grid explicitly pairs equal grids."
        ),
        "interpolation": "Image sampling kernel; cubic/Lanczos coverage excludes a 2/4-pixel rim.",
        "roundtrip_tolerance": "Maximum forward/backward closure error in source pixels.",
        "appearance_tolerance": (
            "Optional matched-luminance difference limit on an 8-bit normalized copy. "
            "Zero disables it; illumination changes can also fail this occlusion proxy."
        ),
        "inverse_iterations": "Fixed-point steps to invert each source-to-intermediate motion map.",
        "inverse_tolerance": (
            "Maximum inverse-map closure error in pixels before rejecting support."
        ),
        "maximum_motion": "Reject flow fields exceeding this per-axis source-pixel displacement.",
        "uncovered": "Hole fill has no motion support and never increases the coverage mask.",
        "minimum_coverage": "Fail below this supported fraction of selected output centers.",
        "timing_policy": (
            "Require compatible video timing/explicit times, or report unknown timing."
        ),
        "timing_source": "Use compatible source video metadata or the configured explicit times.",
        "time_a_ms": "Configured A time in milliseconds, used with explicit timing source.",
        "time_b_ms": "Configured B time, later than A. Wired times override configured timing.",
        "emit_flows": (
            "Retain both DenseFlow outputs for inspection/reuse; otherwise no flow events."
        ),
        "pyramid_scale": "Farneback pyramid scale ratio; used in estimate/refine modes.",
        "levels": "Maximum Farneback pyramid layers, including the input image.",
        "window": "Odd Farneback averaging window in pixels.",
        "iterations": "Farneback iterations per pyramid level.",
        "poly_n": "Farneback polynomial neighborhood size.",
        "poly_sigma": "Farneback polynomial smoothing sigma.",
        "gaussian_window": "Use Gaussian weighting in the Farneback averaging window.",
    }
    input_help = {
        "image_a": "First frame and output job ticket/grid. Same shape, encoding and depth as B.",
        "image_b": (
            "Later frame; its ticket and media identity are retained in interpolation lineage."
        ),
        "forward": "A-to-B DenseFlow or float HxWx2 array. Needed for refine/provided modes.",
        "backward": "B-to-A field on B's grid. Needed for refine/provided modes.",
        "support_a": "Optional uint8 known-flow support on A; connect for masked/sparse fields.",
        "support_b": "Optional uint8 known-flow support on B.",
        "mask": "Optional uint8 selection of output centers on A's grid; outside stays A.",
        "region": "Optional circle/rectangle/polygon intersected with the output selection.",
        "time_a_ms": "Optional explicit A timestamp; supply both times in one clock domain.",
        "time_b_ms": "Optional explicit B timestamp, strictly later than A.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "coverage", "info", "timestamp_ms"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("emit_flows", "gaussian_window"):
            if type(p[key]) is not bool:
                raise ValidationError(f"{key} must be boolean", key)

    def fields(self, inputs, a, b):
        p = self.parameters
        initial_a, initial_b = None, None
        if p["flow_mode"] != "estimate":
            if "forward" not in inputs or "backward" not in inputs:
                raise ValidationError(
                    "Refine/provided modes require both forward and backward fields"
                )
            initial_a = flow_value(
                inputs["forward"], inputs["image_a"], p["maximum_motion"], "Forward"
            )
            initial_b = flow_value(
                inputs["backward"], inputs["image_b"], p["maximum_motion"], "Backward"
            )
        elif "forward" in inputs or "backward" in inputs:
            raise ValidationError("Use refine/provided mode when wiring flow fields")
        if p["flow_mode"] == "provided":
            return initial_a, initial_b
        if min(a.shape[:2]) < 8:
            raise ValidationError("Farneback estimation needs at least 8 pixels on each image axis")
        gray_a, gray_b = (
            gray_image(inputs[k], (np.uint8,), auto_depth=True) for k in ("image_a", "image_b")
        )
        calculate = farneback_calculator(p)
        forward, _ = calculate(gray_a, gray_b, initial_a)
        backward, _ = calculate(gray_b, gray_a, initial_b)
        return (
            flow_value(forward, inputs["image_a"], p["maximum_motion"], "Estimated forward"),
            flow_value(backward, inputs["image_b"], p["maximum_motion"], "Estimated backward"),
        )

    def process(self, inputs):
        p, t = self.parameters, float(self.parameters["fraction"])
        a, b = (
            image_value(inputs[k], (np.uint8, np.uint16, np.float32, np.float64))
            for k in ("image_a", "image_b")
        )
        encoding = image_format(inputs["image_a"])
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or encoding != image_format(inputs["image_b"])
            or encoding in ("hsv", "lab")
        ):
            raise ValidationError(
                "Frames need identical grayscale/RGB(A) shape, encoding and depth"
            )
        height, width = a.shape[:2]
        if height * width > 2_100_000 or max(height, width) >= 32767:
            raise ValidationError("Frame interpolation is limited to 2.1 MP and 32,766 per axis")
        if p["alignment"] == "same_frame":
            require_same_frame(
                [inputs["image_a"], inputs["image_b"]],
                message="Frame grids differ; register them or explicitly use pixel grid",
            )
        ca, cb = (metadata(inputs[k]) for k in ("image_a", "image_b"))
        timestamp, times, time_source, same_stream = interpolation_time(inputs, ca, cb, p)
        mask = selection_mask(
            {
                "image": inputs["image_a"],
                **{k: inputs[k] for k in ("mask", "region") if k in inputs},
            },
            a,
        )
        selected = np.ones((height, width), bool) if mask is None else mask != 0
        forward, backward = None, None
        if t in (0, 1):
            output = a.copy() if t == 0 else b.copy()
            va, vb = selected & (t == 0), selected & (t == 1)
            supported = selected.copy()
        else:
            forward, backward = self.fields(inputs, a, b)
            sa = support_value(inputs.get("support_a"), inputs["image_a"], "Support A")
            sb = support_value(inputs.get("support_b"), inputs["image_b"], "Support B")
            ma, _, _ = _flow_consistency(forward, backward, None, p["roundtrip_tolerance"], sa, sb)
            mb, _, _ = _flow_consistency(backward, forward, None, p["roundtrip_tolerance"], sb, sa)
            gx, gy = np.meshgrid(
                np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
            )
            if float(p["appearance_tolerance"]) > 0:
                gray_a, gray_b = (
                    gray_image(inputs[k], (np.uint8,), auto_depth=True).astype(np.float32) / 255
                    for k in ("image_a", "image_b")
                )
                for mask_source, origin, target, field in (
                    (ma, gray_a, gray_b, forward),
                    (mb, gray_b, gray_a, backward),
                ):
                    corresponding = cv2.remap(
                        target,
                        gx + field[..., 0],
                        gy + field[..., 1],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    mask_source[
                        np.abs(origin - corresponding) > float(p["appearance_tolerance"])
                    ] = 0
            xa, ya, va = inverse_map(forward, t, ma, gx, gy, p)
            xb, yb, vb = inverse_map(backward, 1 - t, mb, gx, gy, p)
            va, vb = va & selected, vb & selected
            interpolation = INTERPOLATION[p["interpolation"]][0]
            warped_a = cv2.remap(
                a,
                xa,
                ya,
                interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                hint=cv2.ALGO_HINT_ACCURATE,
            )
            warped_b = cv2.remap(
                b,
                xb,
                yb,
                interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                hint=cv2.ALGO_HINT_ACCURATE,
            )
            output, supported = blend_supported(warped_a, warped_b, va, vb, t)
            holes = selected & ~supported
            if p["uncovered"] == "nearest_endpoint":
                output[holes] = (a if t <= 0.5 else b)[holes]
            elif p["uncovered"] == "crossfade":
                fallback, _ = blend_supported(a, b, selected, selected, t)
                output[holes] = fallback[holes]
            else:
                output[holes] = 0
        output[~selected] = a[~selected]
        uncertain = selected & ~(va & vb) if 0 < t < 1 else np.zeros_like(selected)
        total, count = int(selected.sum()), int(supported.sum())
        fraction = count / total if total else 0.0
        report = {
            "model": "frame_interpolation_v1",
            "fraction": t,
            "selected_pixels": total,
            "supported_pixels": count,
            "coverage_fraction": fraction,
            "both_sources": int((va & vb).sum()),
            "one_source": int((va ^ vb).sum()),
            "uncovered_pixels": total - count,
            "uncertain_pixels": int(uncertain.sum()),
            "motion_evaluated": forward is not None,
            "flow_mode": p["flow_mode"],
            "sampling": p["interpolation"],
            "roundtrip_tolerance_pixels": float(p["roundtrip_tolerance"]),
            "inverse_tolerance_pixels": float(p["inverse_tolerance"]),
            "appearance_tolerance": float(p["appearance_tolerance"]),
            "uncovered_policy": p["uncovered"],
            "outside_selection": "image_a",
            "timestamp_ms": timestamp,
            "source_times_ms": times,
            "timestamp_source": time_source,
            "sources": [{k: copy.deepcopy(c[k]) for k in SOURCE_KEYS if k in c} for c in (ca, cb)],
            "coordinates": "current input pixel grids; output job ticket/grid owned by A",
            "coverage_basis": (
                "round-trip consistency and inverse convergence; not visibility proof"
            ),
        }
        context = copy.deepcopy(ca)
        context.pop("video_frame", None)
        context["frame_interpolation"] = copy.deepcopy(report)
        if same_stream and timestamp is not None:
            video = copy.deepcopy(ca["video_frame"])
            video.pop("frame_index", None)
            video.update(timestamp_ms=timestamp, timestamp_source="interpolated", synthetic=True)
            context["video_frame"] = video
        values = {
            "image": ImageArray(output, encoding),
            "info": report,
            "timestamp_ms": NO_VALUE if timestamp is None else timestamp,
            "coverage": ImageArray(supported.astype(np.uint8) * 255, "gray"),
            "coverage_a": ImageArray(va.astype(np.uint8) * 255, "gray"),
            "coverage_b": ImageArray(vb.astype(np.uint8) * 255, "gray"),
            "uncertain": ImageArray(uncertain.astype(np.uint8) * 255, "gray"),
            "forward_flow": NO_VALUE,
            "backward_flow": NO_VALUE,
            "coverage_preview": NO_VALUE,
            "overlay": NO_VALUE,
        }
        payloads = {key: copy.deepcopy(context) for key in values}
        if p["emit_flows"] and forward is not None:
            for key, flow, source in (
                ("forward_flow", forward, ca),
                ("backward_flow", backward, cb),
            ):
                flow_context = copy.deepcopy(context)
                for field in ("spatial", "pyramid"):
                    flow_context.pop(field, None)
                    if field in source:
                        flow_context[field] = copy.deepcopy(source[field])
                values[key] = DenseFlow(flow, flow_context)
                payloads[key] = flow_context
        passed = fraction >= float(p["minimum_coverage"])
        result = Result(
            values,
            payloads=payloads,
            passed=passed,
            score=fraction,
            message=f"t={t:g}; supported {count}/{total} ({fraction:.1%})",
            fail_reasons=[] if passed else ["insufficient_interpolation_coverage"],
        )
        if self.generate_graphics:
            preview = np.zeros((height, width, 3), np.uint8)
            preview[selected & ~supported] = (40, 40, 230)
            preview[va & ~vb] = (0, 190, 255)
            preview[vb & ~va] = (255, 190, 0)
            preview[va & vb] = (70, 200, 70)
            values["coverage_preview"] = preview
            overlay = overlay_image(values["image"])
            label = graphic_object(
                "label",
                [[8, 18]],
                (70, 220, 140) if passed else (60, 60, 240),
                text=f"t={t:g}; coverage {fraction:.1%}",
            )
            draw_object(overlay, label, np.eye(3))
            result.drawings["Frame interpolation"] = GraphicDrawing((label,), from_output=True)
            values["overlay"] = overlay
            result.graphics.update({"Interpolated frame": overlay, "Motion support": preview})
        return result
