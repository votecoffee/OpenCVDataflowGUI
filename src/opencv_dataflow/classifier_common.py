"""Shared finite descriptor rows, normalization and portable classifier contracts."""

import copy
import hashlib
import json

import numpy as np

from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, metadata, require_same_frame
from .sample_tables import finite_features
from .tickets import ticket_key


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def model_revision(model):
    return hashlib.sha256(
        canonical_json({k: v for k, v in model.items() if k != "revision"}).encode()
    ).hexdigest()


def label_strings(value, name, *, maximum=2048):
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(s, str) or not 1 <= len(s) <= 128 for s in value)
    ):
        raise ValidationError(
            f"{name} requires up to {maximum} nonempty strings (1..128 characters)"
        )
    return value


def descriptor_matrix(
    value, *, max_rows=2048, max_dimensions=65_536, max_elements=2_000_000, max_abs=None
):
    contracts = []
    if isinstance(value, FlowGroup):
        contracts = [metadata(item).get("feature_contract") for item in value.items]
        value = [item.value for item in value.items]
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Features must be a rectangular numeric table") from exc
    if array.dtype.kind not in "iuf":
        raise ValidationError("Features must be finite numeric values, not strings or booleans")
    if array.ndim == 1:
        array = array[None, :]
    if (
        array.ndim != 2
        or not 1 <= array.shape[0] <= max_rows
        or not 1 <= array.shape[1] <= max_dimensions
        or array.size > max_elements
        or not np.isfinite(array).all()
        or np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max)
    ):
        raise ValidationError("Features exceed finite float32 row/dimension/element limits")
    if max_abs is not None and np.any(np.abs(array.astype(np.float64)) > max_abs):
        raise ValidationError(f"Features must be within ±{max_abs:.0e}")
    if contracts and any(contract != contracts[0] for contract in contracts):
        raise ValidationError("Training features have different preprocessing contracts")
    converted = np.ascontiguousarray(array, np.float32)
    if np.any((array != 0) & (converted == 0)):
        raise ValidationError("Feature values underflow float32; rescale before classification")
    return converted, copy.deepcopy(contracts[0]) if contracts else None


def normalize_features(values, mode):
    if mode == "none":
        return values
    if mode != "l2":
        raise ValidationError("Unknown classifier feature normalization")
    wide = values.astype(np.float64)
    peak = np.max(np.abs(wide), axis=1, keepdims=True)
    if np.any(peak == 0):
        raise ValidationError("Cannot L2 normalize a zero feature vector")
    scaled = wide / peak
    return (scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))).astype(np.float32)


def feature_contract(value):
    if value is not None and (
        not isinstance(value, dict) or len(canonical_json(value).encode()) > 16384
    ):
        raise ValidationError("Feature contract must be a JSON dictionary of at most 16 KiB")
    return copy.deepcopy(value)


def descriptor_rows64(block, value, *, max_columns=65536):
    """Finite float64 descriptor rows with common and per-row wire provenance."""
    context = copy.deepcopy(block.input_payloads.get("samples", {}))
    if isinstance(value, FlowGroup):
        contexts = [metadata(item) for item in value.items]
        contracts = [feature_contract(item.get("feature_contract")) for item in contexts]
        if any(contract != contracts[0] for contract in contracts):
            raise ValidationError("Features have different preprocessing contracts")
        context["feature_contract"] = contracts[0]
        context["feature_rows"] = [
            {
                key: copy.deepcopy(item[key])
                for key in ("ticket", "spatial", "pyramid", "recipe_ref")
                if key in item
            }
            for item in contexts
        ]
        value = [item.value for item in value.items]
    contract = feature_contract(context.get("feature_contract"))
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Features must be a rectangular numeric table") from exc
    if array.ndim == 1 and array.size:
        array = array[None, :]
    return finite_features(array, limit=1e100, max_columns=max_columns), contract, context


def normalize_rows64(values, mode):
    if mode == "none":
        return values
    if mode != "l2":
        raise ValidationError("Unknown descriptor normalization")
    peak = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(peak == 0):
        raise ValidationError("Cannot L2 normalize a zero descriptor")
    scaled = values / peak
    return np.ascontiguousarray(scaled / np.linalg.norm(scaled, axis=1, keepdims=True))


def validate_prediction_image(block, inputs, count):
    """Optional unchanged review image must belong to the current descriptor invocation."""
    image = inputs.get("image", NO_VALUE)
    if image is NO_VALUE:
        return image
    if count != 1:
        raise ValidationError("Connect an image only for a single descriptor prediction")
    features = block.input_payloads.get("samples", {})
    require_same_frame(
        (image, ValuePacket(None, features)), message="Prediction image frame differs"
    )
    image_context = block.input_payloads["image"]
    first, second = image_context.get("ticket"), features.get("ticket")
    if first is not None and second is not None and ticket_key(first) != ticket_key(second):
        raise ValidationError("Prediction image and features belong to different tickets")
    for key in ("recipe_ref", "pyramid"):
        first, second = image_context.get(key), features.get(key)
        if first is not None and second is not None and first != second:
            raise ValidationError("Prediction image and features have conflicting " + key)
    return image


def prediction_drawing(block, image, text, passed, title):
    drawings = {}
    if image is not NO_VALUE and block.generate_graphics:
        color = (100, 220, 90) if passed else (70, 80, 240)
        drawings[title] = GraphicDrawing((graphic_object("label", [(8, 8)], color, text=text),))
    return drawings
