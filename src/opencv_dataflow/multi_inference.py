"""Explicit image/tensor preparation and complete named multi-input inference."""

import copy
import math

import numpy as np

from .core import Block, Port, Result
from .dnn_blocks import DNNInference, preprocess
from .ml_common import tensor_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import SELECTION_INPUTS
from .tickets import ticket_key

SLOTS = tuple(f"input_{i}" for i in range(1, 5))
PREPROCESS_FIELDS = (
    "width",
    "height",
    "color_format",
    "resize_mode",
    "padding_value",
    "scale",
    "mean",
    "std",
)
NETWORK_FIELDS = (
    "model_path",
    "output_names",
    "engine",
    "backend",
    "target",
    "fusion",
    "max_output_elements",
)


def shape(value, field, *, wildcard=False):
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 8
        or any(
            type(n) is not int or not (n == -1 and wildcard or 1 <= n <= 2_000_000) for n in value
        )
    ):
        raise ValidationError(
            "Shape needs 1..8 positive integer axes; declared signatures also allow -1", field
        )
    return value


def float_tensor(value, limit, field="data"):
    if not isinstance(value, (list, int, float)):
        raise ValidationError("Supply flat finite numeric tensor values", field)
    if isinstance(value, list) and (
        len(value) > limit or any(type(n) not in (int, float, bool) for n in value)
    ):
        raise ValidationError("Tensor values must be a bounded flat numeric list", field)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(value, np.float32).reshape(-1)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Tensor conversion to float32 failed", field) from exc
    if not 1 <= values.size <= limit or not np.isfinite(values).all():
        raise ValidationError("Tensor must contain bounded finite float32 values", field)
    return values


def read_tensor(value, selector, limit, port):
    if not isinstance(value, dict):
        raise ValidationError(f"{port} needs a tensor_v1 record or named tensor map")
    if value.get("model") != "tensor_v1":
        if selector:
            value = value.get(selector)
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            raise ValidationError(f"{port}: select an output name for a multi-output tensor map")
    elif selector:
        raise ValidationError(f"{port}: leave Tensor name empty for a single tensor record")
    if (
        not isinstance(value, dict)
        or value.get("model") != "tensor_v1"
        or value.get("dtype") != "float32"
    ):
        raise ValidationError(f"{port} requires a selected float32 tensor_v1 record")
    dimensions = shape(value.get("shape"), port)
    if math.prod(dimensions) > limit:
        raise ValidationError(f"{port} exceeds the input element budget", "maximum_input_elements")
    values = float_tensor(value.get("values"), limit, port)
    if values.size != math.prod(dimensions):
        raise ValidationError(f"{port} values do not match its declared tensor shape")
    return np.ascontiguousarray(values.reshape(dimensions))


class MakeTensor(Block):
    type_id, title = "ml.make_tensor", "Create float32 tensor"
    category = block_type = "ML"
    subtype = "Tensor preparation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("data", "data", required=False),)
    outputs = (Port("tensor", "data"),)
    defaults = {"shape": [1, 3], "values": [1.0, 1.0, 1.0], "maximum_elements": 2_000_000}
    parameter_ranges = {"maximum_elements": NumericRange(1, 2_000_000, whole=True)}
    parameter_help = {
        "shape": "Explicit positive axes. Their product must equal the flat value count.",
        "values": (
            "Fallback scalar/flat numeric list when Data is absent; converted to finite float32."
        ),
        "maximum_elements": "Bound values before array allocation; graph snapshots are separate.",
    }
    input_help = {"data": "Optional scalar or flat numeric list, replacing the Values parameter."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        shape(p["shape"], "shape")
        if type(p["maximum_elements"]) is not int or math.prod(p["shape"]) > p["maximum_elements"]:
            raise ValidationError(
                "Tensor shape exceeds the declared integer element budget",
                "shape",
                "maximum_elements",
            )
        float_tensor(p["values"], p["maximum_elements"], "values")

    def process(self, inputs):
        p = self.parameters
        values = float_tensor(inputs.get("data", p["values"]), p["maximum_elements"])
        if values.size != math.prod(p["shape"]):
            raise ValidationError("Value count does not match Shape", "shape", "values")
        return Result({"tensor": tensor_record(values.reshape(p["shape"]), p["maximum_elements"])})


class PrepareImageTensor(Block):
    type_id, title = "ml.image_tensor", "Image to DNN tensor"
    category = block_type = "ML"
    subtype = "Tensor preparation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = SELECTION_INPUTS
    outputs = (Port("image"), Port("tensor", "data"), Port("preprocessing", "data"))
    defaults = {
        **{k: copy.deepcopy(DNNInference.defaults[k]) for k in PREPROCESS_FIELDS},
        "layout": "nchw",
        "maximum_image_pixels": 16_777_216,
        "maximum_elements": 2_000_000,
    }
    parameter_choices = {
        **{k: v for k, v in DNNInference.parameter_choices.items() if k in PREPROCESS_FIELDS},
        "layout": ("nchw", "nhwc"),
    }
    parameter_ranges = {
        "width": NumericRange(1, 4096, whole=True),
        "height": NumericRange(1, 4096, whole=True),
        "scale": NumericRange(-float(np.finfo(np.float32).max), float(np.finfo(np.float32).max)),
        "padding_value": NumericRange(
            -float(np.finfo(np.float32).max), float(np.finfo(np.float32).max)
        ),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_elements": NumericRange(1, 2_000_000, whole=True),
    }
    parameter_help = {
        "width": "Network image width after resize, pixels.",
        "height": "Network image height, pixels.",
        "color_format": (
            "Explicit network channel order; this does not rescale source intensity units."
        ),
        "resize_mode": (
            "Stretch axes or retain aspect ratio with centered rounded letterbox padding."
        ),
        "padding_value": "Letterbox intensity before normalization.",
        "scale": "Normalization is (pixel - mean) * scale / std, in chosen channel order.",
        "mean": "One value per network channel, in original intensity units.",
        "std": "One strictly positive divisor per network channel.",
        "layout": "NCHW [1,C,H,W] or NHWC [1,H,W,C]; never guessed from the model.",
        "maximum_image_pixels": "Source pixel guard before conversion; graph copies are separate.",
        "maximum_elements": "Prepared tensor budget checked before resizing/allocation.",
    }
    input_help = {
        "image": "Original ticketed image; returned unchanged with source coordinates.",
        "mask": "Excluded source samples become zero before resize/normalization.",
        "region": "Intersect with the mask on the source grid; keep original dimensions.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for k, r in self.parameter_ranges.items():
            if type(p[k]) not in (int, float) or r.whole and type(p[k]) is not int:
                raise ValidationError("Use typed numeric tensor settings", k)
        channels = 1 if p["color_format"] == "gray" else 3
        if p["width"] * p["height"] * channels > p["maximum_elements"]:
            raise ValidationError(
                "Prepared image tensor exceeds the element budget",
                "width",
                "height",
                "maximum_elements",
            )
        for name in ("mean", "std"):
            values = p[name]
            if (
                not isinstance(values, list)
                or len(values) != channels
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in values)
                or name == "std"
                and any(n <= 0 for n in values)
            ):
                raise ValidationError(
                    "Mean/std need finite per-channel values; std must be positive", name
                )
            with np.errstate(over="ignore", under="ignore"):
                native = np.asarray(values, np.float32)
            if not np.isfinite(native).all() or (name == "std" and np.any(native <= 0)):
                raise ValidationError("Mean/std must be representable in float32", name)

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        if (
            max(image.shape[:2]) > 16384
            or image.shape[0] * image.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError(
                "Source image exceeds the configured tensor-preparation budget",
                "maximum_image_pixels",
            )
        tensor, geometry = preprocess(inputs, p)
        if p["layout"] == "nhwc":
            tensor = np.ascontiguousarray(tensor.transpose(0, 2, 3, 1))
        geometry.update(layout=p["layout"], tensor_shape=list(tensor.shape), dtype="float32")
        context = copy.deepcopy(self.input_payloads.get("image", {}))
        tensor_context = {**context, "tensor_preprocessing": copy.deepcopy(geometry)}
        return Result(
            {
                "image": image,
                "tensor": tensor_record(tensor, p["maximum_elements"]),
                "preprocessing": geometry,
            },
            payloads={"image": context, "tensor": tensor_context, "preprocessing": context},
            message=f"{p['layout'].upper()} {list(tensor.shape)}",
        )


class MultiInputInference(DNNInference):
    type_id, title = "ml.dnn_multi", "Multi-input ONNX inference"
    subtype = "DNN inference"
    preview_safe = parallel_safe = False
    process_inputs_readonly = True
    inputs = tuple(
        Port(name, "data", required=i == 0, required_when_connected=True)
        for i, name in enumerate(SLOTS)
    )
    outputs = (
        Port("tensors", "data"),
        Port("preprocessing", "data", required=False),
        Port("input_info", "data"),
    )
    defaults = {
        **{k: copy.deepcopy(DNNInference.defaults[k]) for k in NETWORK_FIELDS},
        "active_inputs": 2,
        "primary_input": "input_1",
        "ticket_policy": "same_ticket",
        "maximum_input_elements": 2_000_000,
        "maximum_total_elements": 4_000_000,
        **{
            f"{slot}_{key}": value
            for slot, name in zip(SLOTS, ("left", "right", "aux1", "aux2"), strict=True)
            for key, value in (
                ("name", name),
                ("shape", [1, 3, -1, -1]),
                ("tensor", ""),
                ("role", "current" if slot in SLOTS[:2] else "reference"),
            )
        },
    }
    parameter_choices = {
        **{k: v for k, v in DNNInference.parameter_choices.items() if k in NETWORK_FIELDS},
        "primary_input": SLOTS,
        "ticket_policy": ("same_ticket", "same_value", "unchecked"),
        **{slot + "_role": ("current", "reference") for slot in SLOTS},
    }
    parameter_ranges = {
        "active_inputs": NumericRange(1, 4, whole=True),
        "maximum_input_elements": NumericRange(1, 2_000_000, whole=True),
        "maximum_total_elements": NumericRange(1, 8_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **{
            k: DNNInference.parameter_help.get(k, "Shared native inference setting.")
            for k in NETWORK_FIELDS
        },
        "active_inputs": (
            "First N slots are required every activation; no last-value cache or partial inference."
        ),
        "primary_input": (
            "Owns output ticket, spatial/pyramid context and optional preprocessing. Must "
            "be Current."
        ),
        "ticket_policy": (
            "Current inputs: full ticket identity, explicit kind/value pairing, or "
            "unchecked. Reference inputs are exempt."
        ),
        "maximum_input_elements": (
            "Bound each selected float32 tensor before conversion/allocation."
        ),
        "maximum_total_elements": (
            "Combined input element budget; native network workspace is separate."
        ),
        **{
            slot + "_name": "Exact model input name; distinct across all active slots."
            for slot in SLOTS
        },
        **{
            slot + "_shape": ("Expected axes; -1 permits variation if the model supports it.")
            for slot in SLOTS
        },
        **{
            slot + "_tensor": ("Empty for a tensor/one-entry map; otherwise select an output name.")
            for slot in SLOTS
        },
        **{
            slot + "_role": ("Current checks live tickets; Reference allows cached/constant input.")
            for slot in SLOTS
        },
    }
    input_help = {
        slot: ("Connect prepared images or float32 tensor records/maps.") for slot in SLOTS
    }

    @property
    def payload_reference(self):
        return self.parameters["primary_input"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for k in self.parameter_ranges:
            if type(p[k]) is not int:
                raise ValidationError("Use integer input counts/budgets", k)
        if type(p["fusion"]) is not bool:
            raise ValidationError("Fusion must be a checkbox boolean", "fusion")
        active = SLOTS[: p["active_inputs"]]
        if p["primary_input"] not in active or p[p["primary_input"] + "_role"] != "current":
            raise ValidationError("Primary input must be an active Current slot", "primary_input")
        names = []
        for slot in active:
            name, selector = p[slot + "_name"], p[slot + "_tensor"]
            if (
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 256
                or name != name.strip()
                or not isinstance(selector, str)
                or len(selector) > 256
            ):
                raise ValidationError(
                    "Use nonempty exact model names and optional output selectors",
                    slot + "_name",
                    slot + "_tensor",
                )
            shape(p[slot + "_shape"], slot + "_shape", wildcard=True)
            names.append(name)
        if len(names) != len(set(names)):
            raise ValidationError(
                "Each active model input name must be unique", *(s + "_name" for s in active)
            )

    def process(self, inputs):
        # Bad or missing inputs also discard a previously prepared native network.
        try:
            return self.infer_complete(inputs)
        except Exception:
            self.reset_runtime()
            raise

    def infer_complete(self, inputs):
        p = self.parameters
        active = SLOTS[: p["active_inputs"]]
        missing, extra = set(active) - inputs.keys(), inputs.keys() - set(active)
        if missing or extra:
            raise ValidationError(
                f"Incomplete input binding; missing={sorted(missing)}, inactive={sorted(extra)}"
            )
        arrays, records, total = [], [], 0
        for slot in active:
            tensor = read_tensor(
                inputs[slot], p[slot + "_tensor"], p["maximum_input_elements"], slot
            )
            declared = p[slot + "_shape"]
            if len(declared) != tensor.ndim or any(
                a != -1 and a != b for a, b in zip(declared, tensor.shape, strict=True)
            ):
                raise ValidationError(
                    f"{slot} shape {list(tensor.shape)} does not match {declared}", slot + "_shape"
                )
            total += tensor.size
            if total > p["maximum_total_elements"]:
                raise ValidationError(
                    "Combined input tensors exceed the element budget", "maximum_total_elements"
                )
            context = self.input_payloads.get(slot, {})
            prepared = context.get("tensor_preprocessing")
            if prepared is not None and (
                not isinstance(prepared, dict)
                or prepared.get("tensor_shape") != list(tensor.shape)
                or prepared.get("dtype") != "float32"
                or prepared.get("layout") not in ("nchw", "nhwc")
                or context.get("spatial")
                and prepared.get("input_size") != context["spatial"]["image_size"]
            ):
                raise ValidationError(f"{slot} tensor/preprocessing provenance disagrees")
            arrays.append((p[slot + "_name"], tensor))
            records.append(
                {
                    "slot": slot,
                    "name": p[slot + "_name"],
                    "shape": list(tensor.shape),
                    "role": p[slot + "_role"],
                    "dtype": "float32",
                    "ticket": copy.deepcopy(context.get("ticket")),
                    "recipe_ref": copy.deepcopy(context.get("recipe_ref")),
                    "spatial": copy.deepcopy(context.get("spatial")),
                    "pyramid": copy.deepcopy(context.get("pyramid")),
                    "preprocessing": copy.deepcopy(context.get("tensor_preprocessing")),
                }
            )
        current = [r for r in records if r["role"] == "current"]
        tickets = [r["ticket"] for r in current if r["ticket"] is not None]
        if p["ticket_policy"] != "unchecked" and tickets:
            if len(tickets) != len(current):
                raise ValidationError(
                    "Every Current input needs a ticket when any Current input is ticketed"
                )
            identities = [
                ticket_key(t) if p["ticket_policy"] == "same_ticket" else ticket_key(t)[1:]
                for t in tickets
            ]
            if any(t != identities[0] for t in identities[1:]):
                raise ValidationError("Current model inputs belong to different tickets")
        token = tuple((name, tuple(a.shape)) for name, a in arrays)
        tensors, filename = self.infer_tensors(arrays, token)
        primary = next(r for r in records if r["slot"] == p["primary_input"])
        context = copy.deepcopy(self.input_payloads.get(p["primary_input"], {}))
        context.pop("tensor_preprocessing", None)
        info = {
            "model": "multi_input_inference_v1",
            "primary_input": p["primary_input"],
            "ticket_policy": p["ticket_policy"],
            "recipe_consistency": "shared_runtime",
            "tickets_verified": p["ticket_policy"] != "unchecked" and len(tickets) == len(current),
            "input_elements": total,
            "inputs": records,
        }
        return Result(
            {
                "tensors": tensors,
                "preprocessing": primary["preprocessing"] or NO_VALUE,
                "input_info": info,
            },
            payloads={
                name: copy.deepcopy(context) for name in ("tensors", "preprocessing", "input_info")
            },
            message=f"{filename}: {len(arrays)} named inputs, {len(tensors)} outputs",
        )


MULTI_INFERENCE_BLOCKS = (MakeTensor, PrepareImageTensor, MultiInputInference)
