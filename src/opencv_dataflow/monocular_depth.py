"""Explicit monocular-depth quantities with conservative source-grid support."""

import copy

import cv2
import numpy as np

from .core import Port, Result
from .dnn_blocks import BACKENDS, ENGINES, TARGETS, DNNInference
from .images import DepthArray, ImageArray, format_image, image_format, opencv_image
from .ml_common import file_key
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    model_bytes,
    resolve_contract,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value, letterbox_bounds
from .selection import selection_mask

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
PREPARATION_INTERPOLATIONS = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


def linear_support(supported, x, y):
    """Require every nonzero full-float bilinear contributor, including tiny weights."""
    hh, hw = supported.shape
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    x0, x1 = np.clip(ix, 0, hw - 1), np.clip(ix + 1, 0, hw - 1)
    y0, y1 = np.clip(iy, 0, hh - 1), np.clip(iy + 1, 0, hh - 1)
    return (
        supported[y0[:, None], x0]
        & ((fx == 0)[None, :] | supported[y0[:, None], x1])
        & ((fy == 0)[:, None] | supported[y1[:, None], x0])
        & ((fx == 0)[None, :] | (fy == 0)[:, None] | supported[y1[:, None], x1])
    )


def preprocessing(p):
    return {
        "model": "monocular_depth_preparation_v1",
        "source_units": "integer_full_scale_or_float_unit",
        **{
            k: p[k]
            for k in (
                "width",
                "height",
                "color_format",
                "resize_mode",
                "input_interpolation",
                "padding_value",
                "mean",
                "std",
            )
        },
    }


def plane(values, layout):
    if values.ndim != {"HW": 2, "NHW": 3, "CHW": 3, "HWC": 3, "NCHW": 4, "NHWC": 4}[layout]:
        raise ValidationError("Depth head dimensions disagree with Layout", "layout")
    if layout.startswith("N"):
        if values.shape[0] != 1:
            raise ValidationError("Depth head batch must be one", "layout")
        values, layout = values[0], layout[1:]
    if layout == "CHW":
        if values.shape[0] != 1:
            raise ValidationError("Depth head needs one channel", "layout")
        values = values[0]
    elif layout == "HWC":
        if values.shape[-1] != 1:
            raise ValidationError("Depth head needs one channel", "layout")
        values = values[..., 0]
    if not all(1 <= n <= 4096 for n in values.shape):
        raise ValidationError("Depth head axes exceed 1..4096", "layout")
    return values


class MonocularDepth(DNNInference):
    type_id, title = "ml.monocular_depth", "Estimate monocular depth"
    subtype = "Depth estimation"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("preview", required=False),
        Port("valid_mask", visual_type="mask"),
        Port("estimate", "data", required=False),
        Port("depth", "depth", required=False),
        Port("info", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "input_name": "input",
        "output_names": [],
        "width": 256,
        "height": 256,
        "color_format": "rgb",
        "resize_mode": "stretch",
        "input_interpolation": "cubic",
        "padding_value": 0.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "layout": "NHW",
        "convention": "relative_inverse_depth",
        "head_scale": 1.0,
        "head_offset": 0.0,
        "units": "m",
        "minimum_model_value": 0.0,
        "maximum_model_value": 0.0,
        "minimum_depth": 0.001,
        "maximum_depth": 100.0,
        "resampling": "linear",
        "minimum_valid_pixels": 1,
        "minimum_valid_fraction": 0.01,
        "export_estimate": True,
        "preview_mode": "auto_percentile",
        "display_min": 0.0,
        "display_max": 1.0,
        "engine": "classic",
        "backend": "opencv",
        "target": "cpu",
        "fusion": True,
        "maximum_source_pixels": 1_000_000,
        "maximum_input_elements": 3_000_000,
        "max_output_elements": 1_000_000,
    }
    parameter_choices = {
        **CONTRACT_CHOICES,
        "color_format": ("rgb", "bgr", "gray"),
        "resize_mode": ("stretch", "letterbox"),
        "input_interpolation": tuple(PREPARATION_INTERPOLATIONS),
        "layout": ("NHW", "NCHW", "NHWC", "HW", "CHW", "HWC"),
        "convention": (
            "relative_inverse_depth",
            "relative_depth",
            "metric_depth",
            "metric_inverse_depth",
        ),
        "units": ("m", "mm"),
        "resampling": tuple(INTERPOLATIONS),
        "preview_mode": ("auto_percentile", "fixed"),
        "engine": tuple(ENGINES),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_ranges = {
        "width": NumericRange(1, 4096, whole=True),
        "height": NumericRange(1, 4096, whole=True),
        "padding_value": NumericRange(0, 1),
        "head_scale": NumericRange(-1e9, 1e9),
        "head_offset": NumericRange(-1e9, 1e9),
        "minimum_model_value": NumericRange(0, 1e12),
        "maximum_model_value": NumericRange(0, 1e12),
        "minimum_depth": NumericRange(0, 1e12),
        "maximum_depth": NumericRange(1e-12, 1e12),
        "minimum_valid_pixels": NumericRange(1, 4_000_000, whole=True),
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_source_pixels": NumericRange(1, 4_000_000, whole=True),
        "maximum_input_elements": NumericRange(1, 16_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **CONTRACT_HELP,
        "model_path": "Local ONNX or sealed Contract path; no implicit download or calibration.",
        "convention": "Declare model meaning. Relative outputs do not emit physical Depth.",
        "units": "Metric Depth units; metric-inverse model values use their reciprocal.",
        "head_scale": "Fixed nonzero affine scale before resampling; never fitted per image.",
        "head_offset": "Fixed affine offset; display normalization never changes model values.",
        "minimum_model_value": "Affine values at/below this bound are invalid; zero is reserved.",
        "maximum_model_value": "Optional inclusive affine upper bound; zero disables.",
        "minimum_depth": "Metric lower bound after inversion; values must also be positive.",
        "maximum_depth": "Metric upper bound; outside values become invalid, not clipped.",
        "mean": "Per-channel normalized source values, subtracted before Std division.",
        "std": "Positive per-channel divisors; avoid duplicating embedded normalization.",
        "resize_mode": "Rounded letterbox or stretch with explicit half-pixel centers.",
        "resampling": "Interpolate affine model values before inverse-metric conversion.",
        "export_estimate": "Emit JSON quantity map; disable to avoid per-pixel list overhead.",
        "preview_mode": "Display-only stretch. Numerical outputs never normalize per image.",
        "minimum_valid_fraction": "Required selected-pixel coverage; not a confidence score.",
        "maximum_source_pixels": "Source-grid resampling/map bound before model I/O.",
        "maximum_input_elements": "Configured input tensor bound before allocation.",
    }
    input_help = {
        "image": "Original uint8/uint16 or float32 [0,1]; returned unchanged.",
        "mask": "Write selection. Excluded source pixels remain model context.",
        "region": "Intersects Mask; known coordinate-frame conflicts fail.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        channels = 1 if p["color_format"] == "gray" else 3
        for key in ("mean", "std"):
            if (
                not isinstance(p[key], list)
                or len(p[key]) != channels
                or any(
                    type(v) not in (int, float)
                    or not np.isfinite(v)
                    or abs(v) > 1e6
                    or key == "std"
                    and v < 1e-12
                    for v in p[key]
                )
            ):
                raise ValidationError("Use C finite means/positive standard deviations", key)
        for key in ("fusion", "export_estimate"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        for key in (
            "head_scale",
            "head_offset",
            "minimum_model_value",
            "maximum_model_value",
            "minimum_depth",
            "maximum_depth",
            "minimum_valid_fraction",
            "padding_value",
        ):
            if type(p[key]) not in (int, float):
                raise ValidationError("Use a numeric value", key)
        if p["head_scale"] == 0:
            raise ValidationError("Head scale must be nonzero", "head_scale")
        if p["maximum_model_value"] and p["maximum_model_value"] <= p["minimum_model_value"]:
            raise ValidationError(
                "Model bounds must be ordered", "minimum_model_value", "maximum_model_value"
            )
        if p["maximum_depth"] <= p["minimum_depth"]:
            raise ValidationError("Depth bounds must be ordered", "minimum_depth", "maximum_depth")
        for key in ("display_min", "display_max"):
            if type(p[key]) not in (int, float) or not np.isfinite(p[key]) or abs(p[key]) > 1e12:
                raise ValidationError("Use finite bounded display values", key)
        if p["display_max"] <= p["display_min"]:
            raise ValidationError("Display bounds must be ordered", "display_min", "display_max")
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError("Select CUDA backend and target together", "backend", "target")
        if (p["target"] != "cpu" or not p["fusion"]) and p["engine"] != "classic":
            raise ValidationError("Select Classic for non-CPU or disabled fusion", "engine")
        if not isinstance(p["input_name"], str) or len(p["input_name"]) > 256:
            raise ValidationError("Use a bounded input name", "input_name")
        if (
            not isinstance(p["output_names"], list)
            or len(p["output_names"]) > 1
            or any(not isinstance(n, str) or not n or len(n) > 256 for n in p["output_names"])
        ):
            raise ValidationError("Select at most one head", "output_names")

    def process(self, inputs):
        self.begin_inference()
        try:
            return self.estimate(inputs)
        except Exception:
            self.reset_runtime()
            raise

    def estimate(self, inputs):
        source, p = inputs["image"], self.parameters
        image_value(source, (np.uint8, np.uint16, np.float32))
        h, w = source.shape[:2]
        channels = 1 if p["color_format"] == "gray" else 3
        if min(h, w) < 1 or max(h, w) > 16384 or h * w > p["maximum_source_pixels"]:
            raise ValidationError("Source exceeds guard", "maximum_source_pixels")
        nw, nh = int(p["width"]), int(p["height"])
        if nw * nh * channels > p["maximum_input_elements"]:
            raise ValidationError("Input exceeds guard", "maximum_input_elements")
        if source.dtype == np.float32 and (source.min() < 0 or source.max() > 1):
            raise ValidationError("Float source must be in [0,1]")
        encoding = image_format(source)
        expected = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
        if (
            encoding not in expected
            or (1 if source.ndim == 2 else source.shape[2]) != expected[encoding]
        ):
            raise ValidationError("Use matching gray/RGB/BGR channels")
        selection = selection_mask(inputs, source)
        selected = np.ones((h, w), bool) if selection is None else selection != 0
        values = np.zeros((h, w), np.float32)
        valid = np.zeros((h, w), bool)
        artifact = geometry = None
        if selected.any():
            canonical = opencv_image(source).astype(np.float32)
            if source.dtype != np.float32:
                canonical /= np.iinfo(source.dtype).max
            native_encoding = (
                "gray" if source.ndim == 2 else "bgra" if source.shape[2] == 4 else "bgr"
            )
            canonical = np.asarray(
                format_image(ImageArray(canonical, native_encoding), p["color_format"], "unchanged")
            )
            px = py = 0
            rw, rh = nw, nh
            if p["resize_mode"] == "letterbox":
                px, py, rw, rh = letterbox_bounds(w, h, nw, nh)
            resized = cv2.resize(
                canonical,
                (rw, rh),
                interpolation=PREPARATION_INTERPOLATIONS[p["input_interpolation"]],
            )
            canvas = np.full(
                (nh, nw) if channels == 1 else (nh, nw, channels), p["padding_value"], np.float32
            )
            canvas[py : py + rh, px : px + rw] = resized
            hwc = canvas[..., None] if channels == 1 else canvas
            blob = (
                (hwc - np.asarray(p["mean"], np.float32)) / np.asarray(p["std"], np.float32)
            ).transpose(2, 0, 1)[None]
            if not np.isfinite(blob).all():
                raise ValidationError("Input normalization overflowed", "mean", "std")
            path, contract = resolve_contract(self)
            raw, artifact = model_bytes(path, bounded=True)
            if contract and artifact != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ONNX checksum differs", "contract_path")
            self._snapshot = path, contract, file_key(path), raw, artifact
            tensors, _ = self.infer_tensors(
                ((p["input_name"], np.ascontiguousarray(blob, np.float32)),),
                array_outputs=True,
                preprocessing={p["input_name"]: preprocessing(p)},
            )
            if len(tensors) != 1:
                raise ValidationError("Select exactly one depth head", "output_names")
            head = (
                plane(next(iter(tensors.values())), p["layout"]).astype(np.float64)
                * p["head_scale"]
                + p["head_offset"]
            )
            if not np.isfinite(head).all() or np.any(np.abs(head) > np.finfo(np.float32).max):
                raise ValidationError("Affine values exceed float32", "head_scale", "head_offset")
            hh, hw = head.shape
            supported = head > p["minimum_model_value"]
            if p["maximum_model_value"]:
                supported &= head <= p["maximum_model_value"]
            hx = np.arange(hw, dtype=np.float64)
            hy = np.arange(hh, dtype=np.float64)
            supported &= ((hx * nw >= px * hw) & ((hx + 1) * nw <= (px + rw) * hw))[None, :]
            supported &= ((hy * nh >= py * hh) & ((hy + 1) * nh <= (py + rh) * hh))[:, None]
            x = ((np.arange(w, dtype=np.float64) + 0.5) * rw / w + px) * hw / nw - 0.5
            y = ((np.arange(h, dtype=np.float64) + 0.5) * rh / h + py) * hh / nh - 0.5
            mx, my = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
            method = INTERPOLATIONS[p["resampling"]]
            values = cv2.remap(
                np.where(supported, head, 0).astype(np.float32),
                mx,
                my,
                method,
                borderMode=cv2.BORDER_REPLICATE,
            )
            support = (
                linear_support(supported, mx[0], my[:, 0])
                if p["resampling"] == "linear"
                else cv2.remap(
                    supported.astype(np.uint8), mx, my, method, borderMode=cv2.BORDER_REPLICATE
                )
                != 0
            )
            valid = support & selected & np.isfinite(values) & (values > p["minimum_model_value"])
            if p["maximum_model_value"]:
                valid &= values <= p["maximum_model_value"]
            values[~valid] = 0
            geometry = {
                "source_size": [w, h],
                "network_size": [nw, nh],
                "head_size": [hw, hh],
                "content_bounds": [px, py, rw, rh],
                "source_to_network": [
                    [rw / w, 0, px + (rw / w - 1) / 2],
                    [0, rh / h, py + (rh / h - 1) / 2],
                    [0, 0, 1],
                ],
                "coordinate_convention": "half-pixel grid; full unpadded cells/contributors",
            }
        metric = p["convention"].startswith("metric_")
        inverse = p["convention"].endswith("inverse_depth")
        depth = NO_VALUE
        display = values
        if metric:
            z = values.astype(np.float64)
            if inverse:
                np.divide(1, z, out=z, where=valid)
            valid &= (
                np.isfinite(z)
                & (z > 0)
                & (z >= p["minimum_depth"])
                & (z <= p["maximum_depth"])
                & (z <= np.finfo(np.float32).max)
            )
            values[~valid] = 0
            z[~valid] = np.nan
            depth = DepthArray(z, p["units"], p["display_min"], p["display_max"])
            display = z
        count = int(valid.sum())
        denominator = int(selected.sum())
        fraction = count / denominator if denominator else 0.0
        passed = count >= p["minimum_valid_pixels"] and fraction >= p["minimum_valid_fraction"]
        units = ("1/" if inverse else "") + p["units"] if metric else "arbitrary"
        record = NO_VALUE
        if p["export_estimate"]:
            record = {
                "model": "monocular_depth_estimate_v1",
                "image_size": [w, h],
                "convention": p["convention"],
                "units": units,
                "values": values.ravel().tolist(),
                "invalid_value": 0.0,
                "valid_pixels": count,
                "metric_declared": metric,
                "metric_validated": False,
                "artifact": artifact,
                "metric_basis": "caller_declared_model_or_fixed_calibration" if metric else None,
            }
        preview = NO_VALUE
        graphics = {}
        display_range = None
        if self.generate_graphics:
            low, high = float(p["display_min"]), float(p["display_max"])
            if count and p["preview_mode"] == "auto_percentile":
                low, high = map(float, np.percentile(display[valid], [2, 98]))
            normalized = np.zeros((h, w), np.float64)
            if high > low:
                normalized[valid] = np.clip((display[valid] - low) / (high - low), 0, 1)
            preview = cv2.applyColorMap(
                np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            preview[~valid] = 0
            display_range = [low, high]
            title = (
                f"Estimated depth ({p['units']})"
                if metric
                else f"{p['convention']} (arbitrary units)"
            )
            graphics = {title: preview}
        info = {
            "model": "monocular_depth_report_v1",
            "convention": p["convention"],
            "units": units,
            "physical_depth_emitted": metric,
            "depth_units": p["units"] if metric else None,
            "preview_units": p["units"] if metric else "arbitrary",
            "valid_pixels": count,
            "selected_pixels": denominator,
            "valid_fraction": fraction,
            "native_calls": int(artifact is not None),
            "artifact": artifact,
            "geometry": geometry,
            "preview_range": display_range,
            "preview_normalization_is_numeric": False,
            "support_semantics": "nominal source-grid support, not confidence or optical accuracy",
        }
        outputs = {
            "image": source,
            "preview": preview,
            "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
            "estimate": record,
            "depth": depth,
            "info": info,
        }
        payload = copy.deepcopy(self.input_payloads.get("image", {}))
        payload["depth_estimation"] = {
            "model": "monocular",
            "convention": p["convention"],
            "metric_declared": metric,
            "metric_validated": False,
            "artifact": artifact,
            "head_scale": p["head_scale"],
            "head_offset": p["head_offset"],
        }
        return Result(
            outputs,
            payloads={
                name: copy.deepcopy(payload)
                for name in ("estimate", "depth", "preview", "valid_mask")
                if outputs[name] is not NO_VALUE
            },
            graphics=graphics,
            passed=passed,
            score=fraction,
            message=f"{p['convention']}: {count}/{denominator} valid; {units}",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "monocular_depth.coverage",
                    "message": "Insufficient valid selected predictions",
                }
            ],
        )
