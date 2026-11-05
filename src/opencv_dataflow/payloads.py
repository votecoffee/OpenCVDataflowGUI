"""Optional wire payloads and explicit image coordinate provenance, independent of Qt."""

import copy
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FlowSignal(Enum):
    EMPTY = "no emission"


NO_VALUE = FlowSignal.EMPTY


@dataclass
class ValuePacket:
    value: object
    payload: dict


@dataclass
class FlowGroup:
    items: tuple[ValuePacket, ...]
    labels: tuple[str, ...]

    def __post_init__(self):
        if not 1 <= len(self.items) <= 64 or len(self.labels) != len(self.items):
            raise ValueError("Flow groups require 1..64 labeled entries")
        for packet in self.items:
            if not isinstance(packet, ValuePacket) or packet.value is NO_VALUE:
                raise ValueError("Flow groups require actual value packets")
            validate_payload(packet.payload)


@dataclass
class ImagePyramid:
    """Largest-first image set. Each image retains its own spatial payload."""

    levels: tuple

    def __post_init__(self):
        if not 1 <= len(self.levels) <= 9:
            raise ValueError("Image pyramids require 1..9 levels")
        previous = None
        for index, image in enumerate(self.levels):
            if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
                raise ValueError("Every pyramid level must be a nonempty image")
            if previous is not None and (
                image.size >= previous.size
                or image.dtype != previous.dtype
                or image.shape[2:] != previous.shape[2:]
            ):
                raise ValueError("Pyramid levels must decrease in size and share depth/channels")
            payload = metadata(image)
            validate_payload(payload)
            if (
                "spatial" not in payload
                or payload.get("pyramid", {}).get("level") != index
                or payload["pyramid"].get("largest_size")
                != [self.levels[0].shape[1], self.levels[0].shape[0]]
            ):
                raise ValueError("Pyramid images require matching spatial and per-level payloads")
            previous = image

    @property
    def largest(self):
        return self.levels[0]

    @property
    def shape(self):
        return self.largest.shape


def validate_payload(payload):
    from .recipes import validate_reference

    if isinstance(payload, dict) and "processing_time" in payload:
        from .timing import validate_timing

        validate_timing(payload["processing_time"])

    if isinstance(payload, dict) and "recipe_ref" in payload:
        validate_reference(payload["recipe_ref"])
    from .outcomes import validate_outcome

    if isinstance(payload, dict) and "outcome" in payload:
        validate_outcome(payload["outcome"])
    if not isinstance(payload, dict):
        raise ValueError("Port payload must be a JSON object")
    json.dumps(payload, allow_nan=False)
    if "ticket" in payload:
        from .tickets import validate_ticket

        validate_ticket(payload["ticket"])
    if "spatial" in payload:
        spatial = payload["spatial"]
        if not isinstance(spatial, dict) or not isinstance(spatial.get("image_size"), list):
            raise ValueError("Spatial payload must declare image_size")
        check_spatial(payload, spatial["image_size"])
    if "pyramid" in payload:
        pyramid = payload["pyramid"]
        if not isinstance(pyramid, dict) or not isinstance(pyramid.get("level"), int):
            raise ValueError("Pyramid payload must declare an integer level")
        matrix = np.asarray(pyramid.get("to_largest"), float)
        size = pyramid.get("largest_size")
        if (
            not 0 <= pyramid["level"] <= 8
            or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) != 3
            or not isinstance(size, list)
            or len(size) != 2
            or any(not isinstance(n, int) or n < 1 for n in size)
        ):
            raise ValueError("Invalid pyramid level, largest_size or to_largest transform")


def metadata(value):
    if isinstance(value, ValuePacket):
        return {**metadata(value.value), **copy.deepcopy(value.payload)}
    if isinstance(value, ImagePyramid):
        return metadata(value.largest)
    return copy.deepcopy(getattr(value, "metadata", {}))


def frame(size, root_id=None):
    return {
        "root_id": root_id,
        "root_size": list(size),
        "image_size": list(size),
        "to_root": np.eye(3).tolist(),
    }


def image_with_payload(image, payload):
    from .images import DepthArray, ImageArray, image_format

    if isinstance(image, DepthArray):
        output = image.view(DepthArray)
        output.metadata = copy.deepcopy(payload)
        return output
    return ImageArray(image, image_format(image), metadata=payload)


def check_spatial(payload, size):
    spatial = payload.get("spatial")
    if spatial is None:
        return
    if (
        not isinstance(spatial, dict)
        or spatial.get("image_size") != list(size)
        or len(size) != 2
        or any(not isinstance(n, int) or n < 1 for n in size)
    ):
        raise ValueError("Spatial payload dimensions do not match the image")
    if spatial.get("root_id") is not None and not isinstance(spatial["root_id"], str):
        raise ValueError("Spatial root_id must be a string or null")
    matrix = np.asarray(spatial.get("to_root"), float)
    root_size = spatial.get("root_size")
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
        or not isinstance(root_size, list)
        or len(root_size) != 2
        or any(not isinstance(n, int) or n < 1 for n in root_size)
    ):
        raise ValueError("Spatial payload needs valid root_size and invertible to_root matrix")


def prepare_inputs(block, values):
    prepared, payloads = {}, {}
    for port in block.inputs:
        if port.name not in values:
            continue
        wrapped = values[port.name]
        payload = metadata(wrapped)
        value = wrapped.value if isinstance(wrapped, ValuePacket) else wrapped
        if (
            isinstance(value, ImagePyramid)
            and port.kind == "image"
            and not getattr(block, "preserve_image_sets", False)
        ):
            choice = block.input_levels.get(port.name, "largest")
            index = 0 if choice == "largest" else int(choice)
            if not 0 <= index < len(value.levels):
                raise ValueError(f"{port.name} has no pyramid level {index}")
            value = value.levels[index]
            payload = {
                **metadata(value),
                **{key: item for key, item in payload.items() if key not in ("spatial", "pyramid")},
            }
        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            payload.setdefault("spatial", frame([value.shape[1], value.shape[0]]))
            check_spatial(payload, [value.shape[1], value.shape[0]])
            value = image_with_payload(value, payload)
        validate_payload(payload)
        prepared[port.name], payloads[port.name] = value, payload
    return prepared, payloads


def moved_payload(payload, matrix, size):
    output = copy.deepcopy(payload)
    if "spatial" in output:
        spatial = output["spatial"]
        spatial["to_root"] = (np.asarray(spatial["to_root"]) @ matrix).tolist()
        spatial["image_size"] = list(size)
    if "pyramid" in output:
        output["pyramid"]["to_largest"] = (
            np.asarray(output["pyramid"]["to_largest"]) @ matrix
        ).tolist()
    return output


def finish_outputs(block, result):
    if not isinstance(result.payloads, dict) or not result.payloads.keys() <= result.outputs.keys():
        raise ValueError("Result payloads must name declared output ports")
    reference = next(
        (
            key
            for key, value in block.last_inputs.items()
            if isinstance(value, np.ndarray) and key not in ("mask", "window", "template")
        ),
        None,
    )
    base = copy.deepcopy(block.input_payloads.get(reference, {}))
    if not base:
        base = copy.deepcopy(next((p for p in block.input_payloads.values() if "ticket" in p), {}))
    if reference is None and len(block.input_payloads) == 1:
        base = copy.deepcopy(next(iter(block.input_payloads.values())))
    if block.last_recipe_ref is not None:
        base["recipe_ref"] = copy.deepcopy(block.last_recipe_ref)
    if getattr(block, "_source_ticket", None) is not None:
        base["ticket"] = block._source_ticket
    geometry = result.outputs.get("geometry")
    if reference is not None and isinstance(geometry, dict) and "output_to_input" in geometry:
        base = moved_payload(base, np.asarray(geometry["output_to_input"]), geometry["output_size"])
    first_image = next(
        (value for value in result.outputs.values() if isinstance(value, np.ndarray)), None
    )
    if reference is None and first_image is not None and "spatial" not in base:
        base["spatial"] = frame([first_image.shape[1], first_image.shape[0]], f"{block.id}:image")
    for port in block.execution_outputs:
        value = result.outputs[port.name]
        if value is NO_VALUE:
            continue
        if isinstance(value, ValuePacket):
            result.payloads.setdefault(port.name, metadata(value))
            value = value.value
            result.outputs[port.name] = value
        explicit = result.payloads.get(port.name)
        inherited = metadata(value)
        if reference is not None and isinstance(geometry, dict) and "output_to_input" in geometry:
            inherited = {
                **base,
                **{
                    key: item
                    for key, item in inherited.items()
                    if key not in ("spatial", "pyramid")
                },
            }
        payload = copy.deepcopy(explicit if explicit is not None else (inherited or base))
        if "ticket" in base and "ticket" not in payload:
            payload["ticket"] = copy.deepcopy(base["ticket"])
        if "recipe_ref" in base and "recipe_ref" not in payload:
            payload["recipe_ref"] = copy.deepcopy(base["recipe_ref"])
        if isinstance(value, np.ndarray):
            size = [value.shape[1], value.shape[0]]
            if payload.get("spatial", {}).get("image_size") != size:
                if explicit is not None or metadata(value):
                    check_spatial(payload, size)
                # Plots and response maps are different grids; do not label them as source pixels.
                payload.pop("spatial", None)
                payload.pop("pyramid", None)
            result.outputs[port.name] = image_with_payload(value, payload)
        elif (
            isinstance(value, dict)
            and value.get("model") == "image_region_v1"
            and "spatial" in payload
        ):
            value["coordinate_context"] = copy.deepcopy(payload["spatial"])
        validate_payload(payload)
        if payload:
            result.payloads[port.name] = payload
    pyramid = base.get("pyramid")
    if pyramid:
        level = pyramid["level"]
        size = base.get("spatial", {}).get("image_size", [])
        result.message += f" · pyramid source level {level}, image {size}; coordinates are local"
    return result


def packet_for(result, name):
    payload = result.payloads.get(name, {})
    return ValuePacket(result.outputs[name], payload) if payload else result.outputs[name]


def describe_value(value):
    if callable(getattr(value, "describe", None)):
        return describe_value(value.describe())
    """Bounded JSON descriptions for receipts and inspectors, never image pixels."""
    if value is NO_VALUE:
        return {"emitted": False}
    if isinstance(value, np.ndarray):
        return {"image_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, ImagePyramid):
        return {"pyramid": [describe_value(level) for level in value.levels]}
    if isinstance(value, FlowGroup):
        return {
            "model": "flow_group_v1",
            "entries": [
                {"label": label, "value": describe_value(packet.value), "payload": packet.payload}
                for label, packet in zip(value.labels, value.items)
            ],
        }
    if isinstance(value, dict):
        return {key: describe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compatible(out_kind, in_kind):
    return (
        out_kind == in_kind
        or "any" in (out_kind, in_kind)
        or (out_kind == "pyramid" and in_kind == "image")
    )


def require_same_frame(values):
    contexts = [metadata(value).get("spatial") for value in values]
    known = [c for c in contexts if c and c.get("root_id") is not None]
    if known and any(
        c["root_id"] != known[0]["root_id"]
        or c["image_size"] != known[0]["image_size"]
        or not np.allclose(c["to_root"], known[0]["to_root"], atol=1e-9, rtol=0)
        for c in known[1:]
    ):
        raise ValueError("Channel images use different coordinate frames")
