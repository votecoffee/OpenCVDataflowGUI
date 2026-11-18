"""Paired ONNX stereo with explicit disparity units on the original left grid."""

import copy

import cv2
import numpy as np

from .core import Port, Result
from .dnn_blocks import BACKENDS, ENGINES, TARGETS, DNNInference
from .images import ImageArray, format_image, image_format, opencv_image
from .ml_common import file_key
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    model_bytes,
    resolve_contract,
)
from .monocular_depth import PREPARATION_INTERPOLATIONS, linear_support, plane
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value, letterbox_bounds
from .selection import selection_mask
from .tickets import ticket_key


def preprocessing(p):
    preparation = {
        "model": "stereo_image_preparation_v1",
        "source_units": "integer_full_scale_or_float_unit",
        **{
            k: copy.deepcopy(p[k])
            for k in (
                "width",
                "height",
                "color_format",
                "resize_mode",
                "input_interpolation",
                "padding_value",
                "mean",
                "std",
                "input_layout",
            )
        },
    }
    if p["input_mode"] == "concatenated":
        return {p["left_name"]: {**preparation, "channel_order": "left_then_right"}}
    return {p[key + "_name"]: {**preparation, "view": key} for key in ("left", "right")}


def prepare_image(image, p, bounds):
    canonical = opencv_image(image).astype(np.float32)
    if image.dtype != np.float32:
        canonical /= np.iinfo(image.dtype).max
    encoding = "gray" if image.ndim == 2 else "bgra" if image.shape[2] == 4 else "bgr"
    canonical = np.asarray(
        format_image(ImageArray(canonical, encoding), p["color_format"], "unchanged")
    )
    px, py, rw, rh = bounds
    channels = 1 if p["color_format"] == "gray" else 3
    canvas = np.full((p["height"], p["width"], channels), p["padding_value"], np.float32)
    resized = cv2.resize(
        canonical, (rw, rh), interpolation=PREPARATION_INTERPOLATIONS[p["input_interpolation"]]
    )
    canvas[py : py + rh, px : px + rw] = resized[..., None] if channels == 1 else resized
    blob = (canvas - np.asarray(p["mean"], np.float32)) / np.asarray(p["std"], np.float32)
    if not np.isfinite(blob).all():
        raise ValidationError("Input normalization overflowed", "mean", "std")
    return blob


class LearnedStereo(DNNInference):
    type_id, title, subtype = "ml.stereo_disparity", "Learned stereo disparity", "Depth estimation"
    payload_reference = "left"
    inputs = (
        Port("left"),
        Port("right"),
        Port("left_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("right_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("region", "data", required=False, required_when_connected=True),
        Port("rectification", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("disparity"),
        Port("valid_mask", visual_type="mask"),
        Port("preview", required=False),
        Port("info", "data"),
        Port("rectification", "data", required=False),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "left_name": "left",
        "right_name": "right",
        "output_names": [],
        "input_mode": "named",
        "input_layout": "NCHW",
        "layout": "NHW",
        "width": 640,
        "height": 480,
        "color_format": "rgb",
        "resize_mode": "stretch",
        "input_interpolation": "linear",
        "padding_value": 0.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "head_units": "network_pixels",
        "head_sign": "left_minus_right",
        "head_scale": 1.0,
        "head_offset": 0.0,
        "invalid_value_enabled": False,
        "invalid_head_value": 0.0,
        "minimum_disparity": 0.0,
        "maximum_disparity": 4096.0,
        "resampling": "linear",
        "check_correspondence": True,
        "minimum_valid_pixels": 1,
        "minimum_valid_fraction": 0.01,
        "ticket_policy": "same_ticket",
        "preview_mode": "auto_percentile",
        "display_min": 0.0,
        "display_max": 64.0,
        "engine": "new",
        "backend": "opencv",
        "target": "cpu",
        "fusion": True,
        "maximum_source_pixels": 1_000_000,
        "maximum_input_elements": 3_000_000,
        "max_output_elements": 1_000_000,
    }
    parameter_choices = {
        **CONTRACT_CHOICES,
        "input_mode": ("named", "concatenated"),
        "input_layout": ("NCHW", "NHWC"),
        "layout": ("NHW", "NCHW", "NHWC", "HW", "CHW", "HWC"),
        "color_format": ("rgb", "bgr", "gray"),
        "resize_mode": ("stretch", "letterbox"),
        "input_interpolation": tuple(PREPARATION_INTERPOLATIONS),
        "head_units": ("network_pixels", "head_pixels", "source_pixels"),
        "head_sign": ("left_minus_right", "right_minus_left"),
        "resampling": ("linear", "nearest"),
        "ticket_policy": ("same_ticket", "same_value", "unchecked"),
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
        "invalid_head_value": NumericRange(-1e12, 1e12),
        "minimum_disparity": NumericRange(-16384, 16384),
        "maximum_disparity": NumericRange(-16384, 16384),
        "minimum_valid_pixels": NumericRange(1, 4_000_000, whole=True),
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_source_pixels": NumericRange(1, 4_000_000, whole=True),
        "maximum_input_elements": NumericRange(1, 16_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **CONTRACT_HELP,
        "left_name": "Left model binding, or the combined binding for concatenated input.",
        "right_name": "Right binding for Named mode; distinct from Left name.",
        "input_mode": "Two named tensors, or one tensor with left channels followed by right.",
        "head_units": "Affine head quantity before conversion to original left-image pixels.",
        "head_sign": "Output always becomes d = x_left - x_right, on the left grid.",
        "head_scale": "Fixed nonzero affine head multiplier, before spatial interpolation.",
        "head_offset": "Fixed affine offset in declared head units.",
        "invalid_head_value": "Exact raw sentinel; invalidate all interpolation contributors.",
        "minimum_disparity": "Inclusive lower bound in source pixels, after unit/sign conversion.",
        "maximum_disparity": "Inclusive upper bound in source pixels; outside samples become NaN.",
        "mean": "Per-view channel values in normalized [0,1] source units, before Std division.",
        "std": "Positive per-view channel divisors; use zero mean/unit std for HITNET.",
        "resize_mode": "Apply identical stretch or rounded letterbox to both rectified images.",
        "check_correspondence": "Reject out-of-source x_right; Right mask always checks it.",
        "ticket_policy": "Match session+kind+value, kind+value, or unchecked; unticketed allowed.",
        "minimum_valid_fraction": "Required selected coverage; not confidence or accuracy.",
        "preview_mode": "Display-only stretch; never changes numerical disparity.",
        "maximum_input_elements": "Combined two-view tensor bound before conversion/allocation.",
    }
    input_help = {
        "left": "Rectified left image; owns source grid/ticket and is returned unchanged.",
        "right": "Equal-size rectified right view. Epipolar alignment is caller-supplied.",
        "left_mask": "Write selection on the left grid; excluded pixels remain model context.",
        "right_mask": "Support at x_left - disparity; fractional samples need both neighbors.",
        "region": "Left-grid region, intersected with Left mask.",
        "rectification": "Optional stereo_rectification_v1 for this size, preserved for 3D.",
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
                raise ValidationError("Use per-view finite means/positive standard deviations", key)
        for key in ("fusion", "invalid_value_enabled", "check_correspondence"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        for key in self.parameter_ranges:
            if (
                type(p[key]) not in (int, float)
                or self.parameter_ranges[key].whole
                and type(p[key]) is not int
            ):
                raise ValidationError("Use typed numeric settings", key)
        if not p["head_scale"]:
            raise ValidationError("Head scale must be nonzero", "head_scale")
        if p["maximum_disparity"] <= p["minimum_disparity"]:
            raise ValidationError(
                "Disparity bounds must be ordered", "minimum_disparity", "maximum_disparity"
            )
        for key in ("display_min", "display_max"):
            if type(p[key]) not in (int, float) or not np.isfinite(p[key]) or abs(p[key]) > 1e12:
                raise ValidationError("Use finite bounded display values", key)
        if p["display_max"] <= p["display_min"]:
            raise ValidationError("Display bounds must be ordered", "display_min", "display_max")
        for key in ("left_name", "right_name"):
            if not isinstance(p[key], str) or not 1 <= len(p[key]) <= 256:
                raise ValidationError("Use a bounded nonempty binding name", key)
        if p["input_mode"] == "named" and p["left_name"] == p["right_name"]:
            raise ValidationError("Named inputs must be distinct", "left_name", "right_name")
        if (
            not isinstance(p["output_names"], list)
            or len(p["output_names"]) > 1
            or any(not isinstance(n, str) or not 1 <= len(n) <= 256 for n in p["output_names"])
        ):
            raise ValidationError(
                "Select at most one left-reference disparity head", "output_names"
            )
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError("Select CUDA backend and target together", "backend", "target")
        if (p["target"] != "cpu" or not p["fusion"]) and p["engine"] != "classic":
            raise ValidationError("Select Classic for non-CPU or disabled fusion", "engine")

    def process(self, inputs):
        self.begin_inference()
        try:
            return self.estimate(inputs)
        except Exception:
            self.reset_runtime()
            raise

    def estimate(self, inputs):
        p = self.parameters
        left, right = inputs["left"], inputs["right"]
        h, w = left.shape[:2]
        channels = 1 if p["color_format"] == "gray" else 3
        if h * w > p["maximum_source_pixels"] or min(h, w) < 1 or max(h, w) > 16384:
            raise ValidationError("Source exceeds guard", "maximum_source_pixels")
        if right.shape[:2] != (h, w):
            raise ValidationError("Rectified left/right image dimensions must match")
        nw, nh = p["width"], p["height"]
        if 2 * nw * nh * channels > p["maximum_input_elements"]:
            raise ValidationError("Paired input exceeds guard", "maximum_input_elements")
        for image in (left, right):
            image_value(image, (np.uint8, np.uint16, np.float32))
            if image.dtype == np.float32 and (image.min() < 0 or image.max() > 1):
                raise ValidationError("Float sources must be in [0,1]")
            expected = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
            if (
                image_format(image) not in expected
                or (1 if image.ndim == 2 else image.shape[2]) != expected[image_format(image)]
            ):
                raise ValidationError("Use matching gray/RGB/BGR channels")
        contexts = [copy.deepcopy(self.input_payloads.get(k, {})) for k in ("left", "right")]
        tickets = [c.get("ticket") for c in contexts]
        verified = False
        if p["ticket_policy"] != "unchecked" and any(t is not None for t in tickets):
            if any(t is None for t in tickets):
                raise ValidationError("Both views need tickets when either view is ticketed")
            a, b = [ticket_key(t) for t in tickets]
            if (a if p["ticket_policy"] == "same_ticket" else a[1:]) != (
                b if p["ticket_policy"] == "same_ticket" else b[1:]
            ):
                raise ValidationError("Stereo inputs belong to different tickets")
            verified = True
        rectification = inputs.get("rectification", NO_VALUE)
        if rectification is not NO_VALUE:
            if (
                not isinstance(rectification, dict)
                or rectification.get("model") != "stereo_rectification_v1"
                or rectification.get("image_size") != [w, h]
            ):
                raise ValidationError("Rectification record must match original paired image size")
            q = np.asarray(rectification.get("Q"), np.float64)
            if q.shape != (4, 4) or not np.isfinite(q).all() or np.linalg.matrix_rank(q) != 4:
                raise ValidationError("Rectification Q must be finite and invertible")
            if (
                rectification.get("units") not in ("m", "mm")
                or not isinstance(rectification.get("frame"), str)
                or not rectification["frame"]
            ):
                raise ValidationError("Rectification needs physical units and a frame")
        selected = selection_mask(
            {
                "image": left,
                **{k: inputs[k] for k in ("region",) if k in inputs},
                **({"mask": inputs["left_mask"]} if "left_mask" in inputs else {}),
            },
            left,
        )
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        right_support = selection_mask(
            {"image": right, **({"mask": inputs["right_mask"]} if "right_mask" in inputs else {})},
            right,
        )
        disparity = np.full((h, w), np.nan, np.float32)
        valid = np.zeros((h, w), bool)
        artifact = geometry = None
        if selected.any():
            bounds = (
                letterbox_bounds(w, h, nw, nh)
                if p["resize_mode"] == "letterbox"
                else (0, 0, nw, nh)
            )
            px, py, rw, rh = bounds
            blobs = [prepare_image(im, p, bounds) for im in (left, right)]

            def pack(blob):
                return np.ascontiguousarray(
                    blob.transpose(2, 0, 1)[None] if p["input_layout"] == "NCHW" else blob[None]
                )

            arrays = (
                [(p["left_name"], pack(np.concatenate(blobs, axis=2)))]
                if p["input_mode"] == "concatenated"
                else [(p[k + "_name"], pack(blob)) for k, blob in zip(("left", "right"), blobs)]
            )
            path, contract = resolve_contract(self)
            raw, artifact = model_bytes(path, bounded=True)
            if contract and artifact != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ONNX checksum differs", "contract_path")
            self._snapshot = path, contract, file_key(path), raw, artifact
            tensors, _ = self.infer_tensors(
                arrays, array_outputs=True, preprocessing=preprocessing(p)
            )
            if len(tensors) != 1:
                raise ValidationError(
                    "Select exactly one left-reference disparity head", "output_names"
                )
            head = plane(next(iter(tensors.values())), p["layout"])
            supported = np.ones(head.shape, bool)
            if p["invalid_value_enabled"]:
                supported &= head != p["invalid_head_value"]
            hh, hw = head.shape
            factor = {
                "network_pixels": w / rw,
                "head_pixels": nw / hw * w / rw,
                "source_pixels": 1,
            }[p["head_units"]]
            factor *= 1 if p["head_sign"] == "left_minus_right" else -1
            values = (head.astype(np.float64) * p["head_scale"] + p["head_offset"]) * factor
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValidationError("Affine disparity overflowed", "head_scale", "head_offset")
            supported &= (values >= p["minimum_disparity"]) & (values <= p["maximum_disparity"])
            hx, hy = np.arange(hw), np.arange(hh)
            supported &= ((hx * nw >= px * hw) & ((hx + 1) * nw <= (px + rw) * hw))[None, :]
            supported &= ((hy * nh >= py * hh) & ((hy + 1) * nh <= (py + rh) * hh))[:, None]
            x = (((np.arange(w, dtype=np.float64) + 0.5) * rw / w + px) * hw / nw - 0.5).astype(
                np.float32
            )
            y = (((np.arange(h, dtype=np.float64) + 0.5) * rh / h + py) * hh / nh - 0.5).astype(
                np.float32
            )
            mx, my = np.meshgrid(x, y)
            method = cv2.INTER_LINEAR if p["resampling"] == "linear" else cv2.INTER_NEAREST
            disparity = cv2.remap(
                np.where(supported, values, 0).astype(np.float32),
                mx,
                my,
                method,
                borderMode=cv2.BORDER_REPLICATE,
            )
            valid = selected & (
                linear_support(supported, x, y)
                if p["resampling"] == "linear"
                else cv2.remap(
                    supported.astype(np.uint8), mx, my, method, borderMode=cv2.BORDER_REPLICATE
                )
                != 0
            )
            valid &= (
                np.isfinite(disparity)
                & (disparity >= p["minimum_disparity"])
                & (disparity <= p["maximum_disparity"])
            )
            xr = np.arange(w)[None, :] - disparity.astype(np.float64)
            if p["check_correspondence"] or right_support is not None:
                valid &= (xr >= 0) & (xr <= w - 1)
            if right_support is not None:
                lo = np.floor(xr).astype(np.int64)
                hi = np.ceil(xr).astype(np.int64)
                rows = np.arange(h)[:, None]
                valid &= (right_support[rows, np.clip(lo, 0, w - 1)] != 0) & (
                    right_support[rows, np.clip(hi, 0, w - 1)] != 0
                )
            geometry = {
                "source_size": [w, h],
                "network_size": [nw, nh],
                "head_size": [hw, hh],
                "content_bounds": list(bounds),
                "head_to_source_disparity_factor": factor,
                "source_to_network": [
                    [rw / w, 0, px + (rw / w - 1) / 2],
                    [0, rh / h, py + (rh / h - 1) / 2],
                    [0, 0, 1],
                ],
                "coordinate_convention": "half-pixel centers; disparity x_left - x_right",
            }
        disparity[~valid] = np.nan
        count, denominator = int(valid.sum()), int(selected.sum())
        fraction = count / denominator if denominator else 0.0
        passed = count >= p["minimum_valid_pixels"] and fraction >= p["minimum_valid_fraction"]
        preview = NO_VALUE
        graphics = {}
        if self.generate_graphics:
            low, high = p["display_min"], p["display_max"]
            if count and p["preview_mode"] == "auto_percentile":
                low, high = map(float, np.percentile(disparity[valid], [2, 98]))
            display = np.zeros((h, w), np.float32)
            if high > low:
                display[valid] = np.clip((disparity[valid] - low) / (high - low), 0, 1)
            preview = cv2.applyColorMap(np.rint(display * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            preview[~valid] = 0
            graphics = {"Disparity (original left-image pixels)": preview}
        pair = {
            "model": "stereo_pair_provenance_v1",
            "ticket_policy": p["ticket_policy"],
            "tickets_verified": verified,
            "rectification_declared": rectification is not NO_VALUE,
            "epipolar_alignment_verified": False,
            "views": {
                name: {
                    k: copy.deepcopy(c.get(k))
                    for k in ("ticket", "spatial", "pyramid", "recipe_ref")
                }
                for name, c in zip(("left", "right"), contexts)
            },
        }
        payload = {
            **contexts[0],
            "stereo_pair": pair,
            "disparity": {
                "units": "pixels",
                "reference": "left",
                "sign": "left_minus_right",
                "artifact": artifact,
            },
        }
        info = {
            "model": "learned_stereo_report_v1",
            "units": "source_pixels",
            "reference": "left",
            "sign": "left_minus_right",
            "geometry": geometry,
            "valid_pixels": count,
            "selected_pixels": denominator,
            "valid_fraction": fraction,
            "native_calls": int(artifact is not None),
            "artifact": artifact,
            "pair": pair,
            "coverage_is_confidence": False,
        }
        outputs = {
            "image": left,
            "disparity": ImageArray(disparity, "gray"),
            "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
            "preview": preview,
            "info": info,
            "rectification": rectification,
        }
        return Result(
            outputs,
            payloads={
                name: copy.deepcopy(payload)
                for name in ("disparity", "valid_mask", "preview", "info")
                if outputs[name] is not NO_VALUE
            },
            graphics=graphics,
            passed=passed,
            score=fraction,
            message=f"Stereo: {count}/{denominator} valid; source pixels",
            fail_reasons=[]
            if passed
            else [{"code": "stereo.coverage", "message": "Insufficient valid selected disparity"}],
        )
