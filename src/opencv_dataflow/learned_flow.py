"""Paired ONNX optical flow with source-grid vectors and content-bound endpoints."""

import copy
import math

import cv2
import numpy as np

from .core import Port, Result
from .dnn_blocks import BACKENDS, ENGINES, TARGETS, DNNInference
from .flow_files import MODEL, StoredFlow, endpoint, field_hash, preview, validate_manifest
from .flow_warp import same_grid
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .learned_stereo import prepare_image
from .ml_common import file_key
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    model_bytes,
    resolve_contract,
)
from .monocular_depth import PREPARATION_INTERPOLATIONS, linear_support
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value, letterbox_bounds
from .selection import selection_mask


def preprocessing(p):
    preparation = {
        "model": "paired_flow_preparation_v1",
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
        return {p["first_name"]: {**preparation, "channel_order": "first_then_second"}}
    return {p[key + "_name"]: {**preparation, "frame": key} for key in ("first", "second")}


def vector_head(values, layout):
    if values.ndim != (4 if layout.startswith("N") else 3):
        raise ValidationError("Flow rank disagrees with Layout", "layout")
    if layout.startswith("N"):
        if values.shape[0] != 1:
            raise ValidationError("Flow head batch must be one", "layout")
        values, layout = values[0], layout[1:]
    if layout == "CHW":
        values = values.transpose(1, 2, 0)
    if values.shape[-1] != 2 or not all(1 <= n <= 4096 for n in values.shape[:2]):
        raise ValidationError("Flow head needs two channels and bounded spatial axes", "layout")
    return values


def endpoint_support(mask, x, y):
    """Require all nonzero bilinear mask contributors, even arbitrarily small ones."""
    h, w = mask.shape
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    x0, x1 = np.clip(ix, 0, w - 1), np.clip(ix + 1, 0, w - 1)
    y0, y1 = np.clip(iy, 0, h - 1), np.clip(iy + 1, 0, h - 1)
    return (
        (mask[y0, x0] != 0)
        & ((fx == 0) | (mask[y0, x1] != 0))
        & ((fy == 0) | (mask[y1, x0] != 0))
        & ((fx == 0) | (fy == 0) | (mask[y1, x1] != 0))
    )


class LearnedFlowField(StoredFlow):
    def describe(self):
        return {
            **super().describe(),
            "coordinates": "first image grid; first-to-second dx,dy in original pixels",
        }


class LearnedFlow(DNNInference):
    type_id, title, subtype = "ml.optical_flow", "Learned optical flow", "Motion estimation"
    payload_reference = "first"
    inputs = (
        Port("first"),
        Port("second"),
        Port("first_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("second_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("flow", "any"),
        Port("valid_mask", visual_type="mask"),
        Port("magnitude", required=False),
        Port("angle", required=False),
        Port("preview", required=False),
        Port("info", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "first_name": "first",
        "second_name": "second",
        "output_names": [],
        "input_mode": "named",
        "input_layout": "NCHW",
        "layout": "NCHW",
        "width": 480,
        "height": 360,
        "color_format": "rgb",
        "resize_mode": "stretch",
        "input_interpolation": "linear",
        "padding_value": 0.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "component_order": "xy",
        "head_scale": [1.0, 1.0],
        "head_offset": [0.0, 0.0],
        "head_units": "network_pixels",
        "invalid_value_enabled": False,
        "invalid_head_value": 0.0,
        "maximum_motion": 4096.0,
        "alignment": "pixel_grid",
        "resampling": "linear",
        "check_correspondence": True,
        "bind_image_content": True,
        "minimum_valid_pixels": 1,
        "minimum_valid_fraction": 0.01,
        "visual_max_motion": 10.0,
        "arrow_spacing": 24,
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
        "input_mode": ("named", "concatenated"),
        "input_layout": ("NCHW", "NHWC"),
        "layout": ("NCHW", "NHWC", "CHW", "HWC"),
        "color_format": ("rgb", "bgr", "gray"),
        "resize_mode": ("stretch", "letterbox"),
        "input_interpolation": tuple(PREPARATION_INTERPOLATIONS),
        "component_order": ("xy", "yx"),
        "head_units": ("network_pixels", "head_pixels", "source_pixels"),
        "alignment": ("pixel_grid", "same_frame"),
        "resampling": ("linear", "nearest"),
        "engine": tuple(ENGINES),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_ranges = {
        "width": NumericRange(1, 4096, whole=True),
        "height": NumericRange(1, 4096, whole=True),
        "padding_value": NumericRange(0, 1),
        "invalid_head_value": NumericRange(-1e12, 1e12),
        "maximum_motion": NumericRange(0, 16384),
        "minimum_valid_pixels": NumericRange(1, 4_000_000, whole=True),
        "minimum_valid_fraction": NumericRange(0, 1),
        "visual_max_motion": NumericRange(0.001, 1e6),
        "arrow_spacing": NumericRange(1, 1024, whole=True),
        "maximum_source_pixels": NumericRange(1, 2_100_000, whole=True),
        "maximum_input_elements": NumericRange(1, 16_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **CONTRACT_HELP,
        "first_name": "First binding, or combined binding with first channels then second.",
        "second_name": "Distinct second binding for Named mode.",
        "head_units": "Vector units before per-axis conversion to original first-image pixels.",
        "head_scale": "Two fixed nonzero multipliers in decoded xy order, before resampling.",
        "head_offset": "Two finite offsets in decoded xy order and declared head units.",
        "component_order": "Native channels encode xy or yx; output is always dx,dy.",
        "mean": "Per-frame means in unit source values; subtract before Std division.",
        "std": "Positive channel divisors; RAFT's raw RGB model uses 1/255 on unit RGB.",
        "alignment": "Same frame rejects known grid conflicts; Pixel grid declares correspondence.",
        "invalid_head_value": "Either component equal to this enabled sentinel invalidates both.",
        "maximum_motion": "Per-axis source-pixel bound; reject vectors rather than clipping them.",
        "check_correspondence": "Require first+flow inside second; Second mask always enforces it.",
        "bind_image_content": "Bind exact images/grids and flow direction for existing consumers.",
        "minimum_valid_fraction": "Selected support coverage, not learned confidence or accuracy.",
        "visual_max_motion": "Magnitude for full display brightness; never clips numerical flow.",
        "arrow_spacing": "Requested drawing stride, increased automatically to limit 2,048 arrows.",
        "maximum_input_elements": "Combined two-frame tensor budget before conversion/allocation.",
    }
    input_help = {
        "first": "First frame; owns returned image/grid/ticket. Vectors map first to second.",
        "second": "Equal-size second frame. Swap inputs for reverse flow; do not negate vectors.",
        "first_mask": "First-grid write selection; excluded pixels remain model context.",
        "second_mask": "Support at first+flow; all bilinear contributors must be valid.",
        "region": "First-grid selection, intersected with First mask.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [
            name for name in self.exposed_outputs if name not in ("magnitude", "angle", "preview")
        ]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        channels = 1 if p["color_format"] == "gray" else 3
        for key, length in (
            ("mean", channels),
            ("std", channels),
            ("head_scale", 2),
            ("head_offset", 2),
        ):
            if (
                not isinstance(p[key], list)
                or len(p[key]) != length
                or any(
                    type(v) not in (int, float)
                    or not np.isfinite(v)
                    or abs(v) > 1e6
                    or key == "std"
                    and v < 1e-12
                    or key == "head_scale"
                    and v == 0
                    for v in p[key]
                )
            ):
                raise ValidationError("Use finite lists, positive Std and nonzero Head scale", key)
        for key in (
            "fusion",
            "invalid_value_enabled",
            "check_correspondence",
            "bind_image_content",
        ):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        for key, constraint in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or constraint.whole and type(p[key]) is not int:
                raise ValidationError("Use typed numeric settings", key)
        for key in ("first_name", "second_name"):
            if not isinstance(p[key], str) or not 1 <= len(p[key]) <= 256:
                raise ValidationError("Use bounded nonempty input names", key)
        if p["input_mode"] == "named" and p["first_name"] == p["second_name"]:
            raise ValidationError("Named bindings must be distinct", "first_name", "second_name")
        if (
            not isinstance(p["output_names"], list)
            or len(p["output_names"]) > 1
            or any(not isinstance(v, str) or not 1 <= len(v) <= 256 for v in p["output_names"])
        ):
            raise ValidationError("Select at most one first-to-second flow head", "output_names")
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError("Select CUDA backend/target together", "backend", "target")
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
        first, second, p = inputs["first"], inputs["second"], self.parameters
        h, w = first.shape[:2]
        if h * w > p["maximum_source_pixels"] or min(h, w) < 1 or max(h, w) > 16384:
            raise ValidationError("Source exceeds guard", "maximum_source_pixels")
        if second.shape[:2] != (h, w):
            raise ValidationError("First/second image dimensions must match")
        nw, nh = p["width"], p["height"]
        channels = 1 if p["color_format"] == "gray" else 3
        if 2 * nw * nh * channels > p["maximum_input_elements"]:
            raise ValidationError("Paired tensors exceed guard", "maximum_input_elements")
        for image in (first, second):
            image_value(image, (np.uint8, np.uint16, np.float32))
            if image.dtype == np.float32 and (image.min() < 0 or image.max() > 1):
                raise ValidationError("Float input must be in [0,1]")
            expected = {"gray": 1, "rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4}
            if (
                image_format(image) not in expected
                or (1 if image.ndim == 2 else image.shape[2]) != expected[image_format(image)]
            ):
                raise ValidationError("Use matching gray/RGB/BGR channels")
        contexts = [copy.deepcopy(self.input_payloads.get(k, {})) for k in ("first", "second")]
        if p["alignment"] == "same_frame":
            same_grid(contexts[0], contexts[1], "First/second")
        selection = {
            "image": first,
            **({"region": inputs["region"]} if "region" in inputs else {}),
            **({"mask": inputs["first_mask"]} if "first_mask" in inputs else {}),
        }
        selected = selection_mask(selection, first)
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        second_support = selection_mask(
            {
                "image": second,
                **({"mask": inputs["second_mask"]} if "second_mask" in inputs else {}),
            },
            second,
        )
        vectors = np.full((h, w, 2), np.nan, np.float32)
        valid = np.zeros((h, w), bool)
        artifact = geometry = None
        if selected.any():
            bounds = (
                letterbox_bounds(w, h, nw, nh)
                if p["resize_mode"] == "letterbox"
                else (0, 0, nw, nh)
            )
            px, py, rw, rh = bounds
            blobs = [prepare_image(image, p, bounds) for image in (first, second)]

            def pack(blob):
                return np.ascontiguousarray(
                    blob.transpose(2, 0, 1)[None] if p["input_layout"] == "NCHW" else blob[None]
                )

            arrays = (
                [(p["first_name"], pack(np.concatenate(blobs, axis=2)))]
                if p["input_mode"] == "concatenated"
                else [(p[k + "_name"], pack(blob)) for k, blob in zip(("first", "second"), blobs)]
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
                    "Select exactly one first-to-second flow head", "output_names"
                )
            head = vector_head(next(iter(tensors.values())), p["layout"])
            supported = np.ones(head.shape[:2], bool)
            if p["invalid_value_enabled"]:
                supported &= (head != p["invalid_head_value"]).all(axis=2)
            if p["component_order"] == "yx":
                head = head[..., ::-1]
            hh, hw = head.shape[:2]
            factor = np.array(
                {
                    "network_pixels": [w / rw, h / rh],
                    "head_pixels": [nw / hw * w / rw, nh / hh * h / rh],
                    "source_pixels": [1, 1],
                }[p["head_units"]]
            )
            values = (head.astype(np.float64) * p["head_scale"] + p["head_offset"]) * factor
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValidationError(
                    "Flow affine conversion overflowed", "head_scale", "head_offset"
                )
            supported &= (np.abs(values) <= p["maximum_motion"]).all(axis=2)
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
            vectors = cv2.remap(
                np.where(supported[..., None], values, 0).astype(np.float32),
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
            valid &= np.isfinite(vectors).all(axis=2) & (
                np.abs(vectors) <= p["maximum_motion"]
            ).all(axis=2)
            # Clamp only indexing coordinates; unsupported endpoints never gain coverage.
            tx = np.arange(w)[None, :] + vectors[..., 0].astype(np.float64)
            ty = np.arange(h)[:, None] + vectors[..., 1].astype(np.float64)
            if p["check_correspondence"] or second_support is not None:
                valid &= (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
            if second_support is not None:
                valid &= endpoint_support(
                    second_support, np.clip(tx, 0, w - 1), np.clip(ty, 0, h - 1)
                )
            geometry = {
                "source_size": [w, h],
                "network_size": [nw, nh],
                "head_size": [hw, hh],
                "content_bounds": list(bounds),
                "head_to_source_vector_factor": factor.tolist(),
                "source_to_network": [
                    [rw / w, 0, px + (rw / w - 1) / 2],
                    [0, rh / h, py + (rh / h - 1) / 2],
                    [0, 0, 1],
                ],
                "coordinate_convention": "half-pixel centers; first+flow indexes second",
            }
        vectors[~valid] = np.nan
        payload = contexts[0]
        payload.pop("flow_association", None)
        pair = {
            "model": "learned_flow_pair_v1",
            "direction": "first_to_second",
            "units": "source_pixels",
            "alignment": p["alignment"],
            "frames": {
                name: {
                    k: copy.deepcopy(c.get(k))
                    for k in ("ticket", "spatial", "pyramid", "recipe_ref")
                }
                for name, c in zip(("first", "second"), contexts)
            },
        }
        payload["flow_pair"] = pair
        association = None
        if p["bind_image_content"]:
            association = {
                "model": MODEL,
                "sha256": field_hash(vectors),
                "shape": list(vectors.shape),
                "direction": "source_to_target",
                "source": endpoint(first, contexts[0], "first"),
                "target": endpoint(second, contexts[1], "second"),
            }
            validate_manifest(association, vectors.shape, association["sha256"])
            payload["flow_association"] = association
        count, denominator = int(valid.sum()), int(selected.sum())
        fraction = count / denominator if denominator else 0.0
        passed = count >= p["minimum_valid_pixels"] and fraction >= p["minimum_valid_fraction"]
        magnitude = angle = visible = NO_VALUE
        graphics = {}
        drawings = {}
        if self.generate_graphics or {"magnitude", "angle"} & set(self.exposed_outputs):
            safe = np.where(valid[..., None], vectors, 0)
            mag, ang = cv2.cartToPolar(safe[..., 0], safe[..., 1], angleInDegrees=True)
            mag[~valid] = np.nan
            ang[~valid] = np.nan
            magnitude, angle = ImageArray(mag, "gray"), ImageArray(ang, "gray")
        if self.generate_graphics:
            visible = preview(vectors, p["visual_max_motion"])
            graphics = {"Flow (first to second; source pixels)": visible}
            spacing = max(p["arrow_spacing"], math.ceil(math.sqrt(h * w / 2048)))
            while (
                len(range(spacing // 2, h, spacing)) * len(range(spacing // 2, w, spacing)) > 2048
            ):
                spacing += 1
            objects = [
                graphic_object(
                    "arrow",
                    [(x, y), (x + vectors[y, x, 0], y + vectors[y, x, 1])],
                    (60, 200, 100),
                    thickness=1,
                )
                for y in range(spacing // 2, h, spacing)
                for x in range(spacing // 2, w, spacing)
                if valid[y, x]
            ]
            drawings = {"learned motion": GraphicDrawing(tuple(objects), source="first")}
        info = {
            "model": "learned_flow_report_v1",
            "direction": "first_to_second",
            "units": "source_pixels",
            "geometry": geometry,
            "pair": pair,
            "image_content_bound": association is not None,
            "artifact": artifact,
            "native_calls": int(artifact is not None),
            "valid_pixels": count,
            "selected_pixels": denominator,
            "valid_fraction": fraction,
            "coverage_is_confidence": False,
        }
        outputs = {
            "image": first,
            "flow": LearnedFlowField(vectors, copy.deepcopy(payload)),
            "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
            "magnitude": magnitude,
            "angle": angle,
            "preview": visible,
            "info": info,
        }
        return Result(
            outputs,
            payloads={
                name: copy.deepcopy(payload)
                for name in ("flow", "valid_mask", "magnitude", "angle", "preview", "info")
                if outputs[name] is not NO_VALUE
            },
            graphics=graphics,
            drawings=drawings,
            passed=passed,
            score=fraction,
            message=f"Flow first to second: {count}/{denominator} valid source pixels",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "learned_flow.coverage",
                    "message": "Insufficient valid selected motion support",
                }
            ],
        )
