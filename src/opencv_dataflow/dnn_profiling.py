"""Opt-in per-native-forward profiling, separate from image preparation and block timing."""

import copy
from time import perf_counter_ns

import cv2
import numpy as np

from .batch_inference import BatchImageInference
from .core import Port
from .dnn_blocks import DNNInference
from .model_contracts import environment
from .multi_inference import MultiInputInference, PrepareImageTensor
from .parameter_validation import NumericRange, ValidationError

PROFILE_DEFAULTS = {
    "profile_mode": "optional",
    "maximum_profile_layers": 4096,
    "profile_plot_layers": 20,
}
PROFILE_CHOICES = {"profile_mode": ("off", "optional", "required")}
PROFILE_RANGES = {
    "maximum_profile_layers": NumericRange(1, 16384, whole=True),
    "profile_plot_layers": NumericRange(0, 50, whole=True),
}
PROFILE_HELP = {
    "profile_mode": (
        "Off skips added timers, native timing queries and plots. "
        "Optional keeps inference on unavailable timings; Required errors."
    ),
    "maximum_profile_layers": (
        "Total named layer records per activation, across all native calls. "
        "Excess calls are unavailable, never truncated."
    ),
    "profile_plot_layers": (
        "Top timed layers from the last measured native call; 0 disables the optional chart only."
    ),
}


class ProfileUnavailable(ValueError):
    """An observation cannot be represented safely; the forward result can still be valid."""


def native_profile(net, budget):
    names = list(net.getLayerNames())
    if not names or len(names) > budget:
        raise ProfileUnavailable(
            "Native layer names are empty or exceed the remaining layer-record budget"
        )
    if any(not isinstance(n, str) or not n or len(n) > 512 for n in names) or len(
        set(names)
    ) != len(names):
        raise ProfileUnavailable(
            "Native layer names are malformed, duplicated or longer than 512 characters"
        )
    result = net.getPerfProfile()
    if not isinstance(result, tuple) or len(result) != 2:
        raise ProfileUnavailable(
            "The native API did not return the qualified (total ticks, layer ticks) overload"
        )
    total, ticks = result
    ticks = np.asarray(ticks, np.float64)
    if ticks.ndim == 2 and 1 in ticks.shape:
        ticks = ticks.reshape(-1)
    if (
        ticks.ndim != 1
        or len(ticks) != len(names)
        or not np.isfinite(ticks).all()
        or np.any(ticks < 0)
    ):
        raise ProfileUnavailable("Native layer timing vector is incomplete, nonfinite or negative")
    if type(total) not in (int, np.int64, np.int32) or total < 0:
        raise ProfileUnavailable("Native total ticks are not a nonnegative integer")
    frequency = float(cv2.getTickFrequency())
    if not np.isfinite(frequency) or frequency <= 0:
        raise ProfileUnavailable("Native tick frequency is unavailable")
    ids = [net.getLayerId(name) for name in names]
    if any(type(i) not in (int, np.int64, np.int32) for i in ids):
        raise ProfileUnavailable("Native layer IDs are not integers")
    if set(ids) == set(range(len(names))):
        base = 0
    elif set(ids) == set(range(1, len(names) + 1)):
        base = 1
    else:
        raise ProfileUnavailable(
            "Native layer IDs cannot be aligned unambiguously with timing slots"
        )
    if len(set(ids)) != len(names) or total == 0 or not np.any(ticks > 0):
        raise ProfileUnavailable("Native timing sample has no measured ticks or unique layer IDs")
    milliseconds = ticks / frequency * 1000
    native_total_ms = int(total) / frequency * 1000
    if (
        not np.isfinite(milliseconds).all()
        or not np.isfinite(native_total_ms)
        or not np.isfinite(milliseconds.sum())
    ):
        raise ProfileUnavailable("Native timing units overflowed milliseconds")
    rows = [
        {
            "native_layer_id": int(i),
            "timing_index": int(i - base),
            "name": name,
            "ticks": float(ticks[i - base]),
            "milliseconds": float(milliseconds[i - base]),
            "timing_state": "measured" if ticks[i - base] > 0 else "zero_ticks",
        }
        for name, i in zip(names, ids, strict=True)
    ]
    return {
        "status": "available",
        "native_api": "tick_vector",
        "total_ticks": int(total),
        "tick_frequency_hz": frequency,
        "native_total_ms": native_total_ms,
        "layer_total_ms": float(milliseconds.sum()),
        "native_layer_id_base": base,
        "zero_tick_layers": sum(r["timing_state"] == "zero_ticks" for r in rows),
        "layers": rows,
    }


def layer_chart(call, maximum):
    layers = sorted(call["layers"], key=lambda r: (-r["milliseconds"], r["name"]))[:maximum]
    chart = np.full((64 + 24 * len(layers), 900, 3), 245, np.uint8)
    cv2.putText(
        chart,
        f"DNN layers: native call {call['index'] + 1} (ms)",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
    )
    peak = max((r["milliseconds"] for r in layers), default=0)
    for i, row in enumerate(layers):
        y = 51 + 24 * i
        label = row["name"] if len(row["name"]) <= 47 else row["name"][:44] + "..."
        cv2.putText(chart, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 45, 45), 1)
        width = round(410 * row["milliseconds"] / peak) if peak else 0
        color = (190, 125, 55) if row["timing_state"] == "measured" else (150, 150, 150)
        cv2.rectangle(chart, (420, y - 12), (420 + width, y), color, -1)
        cv2.putText(
            chart,
            f"{row['milliseconds']:.4f}",
            (835, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (30, 30, 30),
            1,
        )
    return chart


class _Profiling:
    """Reuse the same inference transaction; never issue a diagnostic second forward."""

    subtype = "DNN profiling"
    preview_safe = True
    parallel_safe = False
    process_inputs_readonly = True

    def reset_runtime(self):
        super().reset_runtime()
        self._profile_calls, self._profile_warnings = [], set()

    close = reset_runtime

    def begin_inference(self):
        super().begin_inference()
        self._profile_calls = []

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in PROFILE_RANGES:
            if type(p[key]) is not int:
                raise ValidationError("Use integer profiling budgets", key)

    def infer_tensors(self, *args, **kwargs):
        return super().infer_tensors(
            *args, **kwargs, observer=None if self.parameters["profile_mode"] == "off" else self
        )

    def forward(self, net, outputs, **context):
        started = perf_counter_ns()
        values = net.forward(outputs)
        elapsed = (perf_counter_ns() - started) / 1e6
        selected = context["selected_execution"]
        used = sum(len(call["layers"]) for call in self._profile_calls)
        try:
            if selected["target"] != "cpu" or selected["backend"] not in ("opencv", "default"):
                raise ProfileUnavailable(
                    "Per-layer timing is qualified only for the OpenCV CPU backend"
                )
            sample = native_profile(net, self.parameters["maximum_profile_layers"] - used)
        except (cv2.error, ProfileUnavailable, AttributeError, TypeError, ValueError) as exc:
            reason = str(exc)[:2048]
            if self.parameters["profile_mode"] == "required":
                raise ValidationError(
                    "Required DNN layer profiling unavailable: " + reason, "profile_mode"
                ) from exc
            sample = {
                "status": "unavailable",
                "reason": reason,
                "layers": [],
                "native_total_ms": None,
                "layer_total_ms": None,
                "total_ticks": None,
                "tick_frequency_hz": None,
                "zero_tick_layers": None,
                "native_layer_id_base": None,
            }
            if reason not in self._profile_warnings:
                self.logger.warning(
                    "DNN layer profiling unavailable; inference retained",
                    parameters={"reason": reason},
                )
                if len(self._profile_warnings) >= 64:
                    self._profile_warnings.clear()
                self._profile_warnings.add(reason)
        self._profile_calls.append(
            {
                "index": len(self._profile_calls),
                **copy.deepcopy(context),
                "forward_outputs": list(outputs),
                "forward_wall_ms": elapsed,
                **sample,
                "native_placement_verified": False,
                "zero_tick_note": (
                    "Zero ticks may mean fused/skipped/not executed/below clock resolution; "
                    "they do not prove zero cost."
                ),
            }
        )
        return values

    def finalize_result(self, result, inputs):
        result = super().finalize_result(result, inputs)
        mode = self.parameters["profile_mode"]
        measured = [c for c in self._profile_calls if c["status"] == "available"]
        available = bool(measured) and len(measured) == len(self._profile_calls)
        status = (
            "disabled"
            if mode == "off"
            else "available"
            if available
            else "partial"
            if measured
            else "unavailable"
        )
        calls = copy.deepcopy(self._profile_calls)
        native_calls = result.outputs.get("info", {}).get("native_calls", [])
        for call, batch in zip(calls, native_calls):
            call["batch"] = copy.deepcopy(batch)
        result.outputs["profile"] = {
            "model": "dnn_layer_profile_v1",
            "status": status,
            "mode": mode,
            "environment": environment(),
            "calls": calls,
            "native_call_count": len(calls) if mode != "off" else None,
            "observed_native_call_count": len(calls),
            "native_total_ms": sum(c["native_total_ms"] for c in measured) if available else None,
            "measured_native_ms_sum": sum(c["native_total_ms"] for c in measured)
            if measured
            else None,
            "forward_wall_ms": sum(c["forward_wall_ms"] for c in calls) if calls else None,
            "layer_records": sum(len(c["layers"]) for c in calls),
            "scope": (
                "Native forward only; excludes import, preparation, output conversion "
                "and profile collection. Existing block timing covers the full execution."
            ),
        }
        context = result.payloads.get("tensors", result.payloads.get("info"))
        if context is not None:
            result.payloads["profile"] = copy.deepcopy(context)
        if self.generate_graphics and measured and self.parameters["profile_plot_layers"]:
            result.graphics["DNN native layer times (last measured call)"] = layer_chart(
                measured[-1], self.parameters["profile_plot_layers"]
            )
        return result


class ProfileDNN(_Profiling, DNNInference):
    type_id, title = "ml.dnn_profile", "Profile ONNX image inference"
    outputs = (*DNNInference.outputs, Port("profile", "data"), Port("image"))
    defaults = {**DNNInference.defaults, **PROFILE_DEFAULTS}
    parameter_choices = {**DNNInference.parameter_choices, **PROFILE_CHOICES}
    parameter_ranges = {
        **{
            k: v
            for k, v in PrepareImageTensor.parameter_ranges.items()
            if k in DNNInference.defaults
        },
        **PROFILE_RANGES,
    }
    parameter_help = {**DNNInference.parameter_help, **PROFILE_HELP}

    def process(self, inputs):
        result = super().process(inputs)
        result.outputs["image"] = inputs["image"]
        return result


class ProfileMultiDNN(_Profiling, MultiInputInference):
    type_id, title = "ml.dnn_multi_profile", "Profile multi-input ONNX inference"
    outputs = (*MultiInputInference.outputs, Port("profile", "data"))
    defaults = {**MultiInputInference.defaults, **PROFILE_DEFAULTS}
    parameter_choices = {**MultiInputInference.parameter_choices, **PROFILE_CHOICES}
    parameter_ranges = {**MultiInputInference.parameter_ranges, **PROFILE_RANGES}
    parameter_help = {**MultiInputInference.parameter_help, **PROFILE_HELP}


class ProfileBatchDNN(_Profiling, BatchImageInference):
    type_id, title = "ml.dnn_batch_profile", "Profile batched ONNX image inference"
    outputs = (*BatchImageInference.outputs, Port("profile", "data"))
    defaults = {**BatchImageInference.defaults, **PROFILE_DEFAULTS}
    parameter_choices = {**BatchImageInference.parameter_choices, **PROFILE_CHOICES}
    parameter_ranges = {**BatchImageInference.parameter_ranges, **PROFILE_RANGES}
    parameter_help = {**BatchImageInference.parameter_help, **PROFILE_HELP}


DNN_PROFILING_BLOCKS = (ProfileDNN, ProfileMultiDNN, ProfileBatchDNN)
