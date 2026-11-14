"""Explicit reuse of static CameraInfo without inventing acquisition timestamps."""

import copy
import hashlib
import json
import time

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import camera
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket
from .ros2_context import acquisition_header, check_current, scope_key, validate_context
from .ros2_support.wire import MAX_DATA_BYTES, header_value, json_bytes
from .tickets import ticket_key, validate_ticket
from .visual_style import INDUSTRIAL

CORE_FIELDS = ("model", "camera_matrix", "distortion", "image_size", "units", "frame", "lens")


def calibration_revision(model):
    from .ros2_calibration import camera_from_info

    camera(model)
    ros = model.get("ros2", {})
    original = ros.get("camera_info")
    expected, _ = camera_from_info(original, space=ros.get("space"))
    core = {k: model.get(k) for k in CORE_FIELDS}
    if core != {k: expected.get(k) for k in CORE_FIELDS}:
        raise ValidationError("Camera model no longer matches its decoded CameraInfo grid")
    if ros.get("header") != expected["ros2"]["header"]:
        raise ValidationError("Camera header no longer matches its original CameraInfo")
    original = copy.deepcopy(original)
    original["header"] = {"frame_id": original["header"]["frame_id"]}
    # Receipt/acquisition stamps are not calibration revisions. ROI/binning/R/P are.
    value = {"camera": core, "camera_info": original, "space": ros["space"]}
    json_bytes(value, MAX_DATA_BYTES)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def static_binding_header(model, payload):
    """Validate an explicit binding before RGB-D uses its image acquisition header."""
    binding = model.get("ros2_static_binding")
    if not isinstance(binding, dict) or binding.get("model") != "ros_static_camera_binding_v1":
        raise ValidationError("Static calibration requires a Cache ROS 2 CameraInfo binding")
    if binding.get("calibration_revision") != calibration_revision(model):
        raise ValidationError("Static calibration changed after binding to the image")
    if binding.get("calibration_header") != model["ros2"]["header"]:
        raise ValidationError("Original calibration header changed after binding")
    if binding.get("image_size") != model["image_size"]:
        raise ValidationError("Static calibration binding has a different image grid")
    try:
        key = ticket_key(binding.get("image_ticket"))
        if key != ticket_key(payload.get("ticket")):
            raise ValueError("Ticket mismatch")
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            "Static calibration binding has a different acquisition ticket"
        ) from exc
    context = binding.get("context")
    if binding.get("context_scope") != (scope_key(context) if context is not None else None):
        raise ValidationError("Static calibration binding has an inconsistent run scope")
    age, maximum = binding.get("receipt_age_seconds"), binding.get("max_age_seconds")
    if (
        type(age) not in (int, float)
        or not np.isfinite(age)
        or age < 0
        or type(maximum) not in (int, float)
        or not 0 <= maximum <= 86400
        or maximum
        and age > maximum
        or binding.get("timing") != "static_calibration_applied_to_image_not_new_measurement"
    ):
        raise ValidationError("Static calibration binding has invalid receipt-age metadata")
    if context is not None:
        check_current(model, payload, context)
    from .recipes import unique_ref

    unique_ref(
        [
            ValuePacket(model, {"recipe_ref": binding["calibration_recipe_ref"]})
            if binding.get("calibration_recipe_ref") is not None
            else ValuePacket(model, {}),
            ValuePacket(model, payload),
        ]
    )
    header = header_value(binding.get("image_header"))
    if header["frame_id"] != model["frame"]:
        raise ValidationError("Static calibration binding has a different optical frame")
    return header


class CacheROSCamera(Block):
    type_id, title = "industrial.ros2_camera_cache", "Cache ROS 2 CameraInfo"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (
        Port("context", "data", required=False),
        Port("reset", "boolean", required=False),
        Port("camera", "data", required=False),
        Port("image", required=False),
    )
    outputs = (Port("image"), Port("camera", "data"), Port("info", "data"))
    defaults = {"max_age_seconds": 0.0, "require_context": False, "allow_zero_stamps": False}
    parameter_ranges = {"max_age_seconds": NumericRange(0, 86400)}
    parameter_help = {
        "max_age_seconds": (
            "Maximum monotonic receipt age of the cached calibration. Zero explicitly "
            "allows unlimited static reuse; this is not camera exposure age."
        ),
        "require_context": (
            "Require a current Run identity and tagged image. Changing context clears "
            "the calibration; prime it again before the next image."
        ),
        "allow_zero_stamps": (
            "Explicit offline/unknown-time mode for image headers. Does not establish "
            "acquisition timing. The original CameraInfo stamp is never rewritten."
        ),
    }
    input_help = {
        "camera": "Decoded CameraInfo model. Updates only the cache; no output event.",
        "image": "Triggers one binding using an already cached matching frame/grid.",
        "context": "Share Run identity; a changed context invalidates the static cache.",
        "reset": "Rising true clears calibration once. Holding true does not repeatedly clear it.",
    }
    accept_events = True
    tolerates_unavailable_inputs = True
    requires_serial_execution = True
    preview_safe = False
    payload_reference = "image"
    flow_primary = ("image",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self.runtime = {"context": None, "cached": None, "reset": False}

    def copy_runtime_from(self, other):
        self.runtime = copy.deepcopy(other.runtime)

    def close(self):
        self.reset_runtime()

    def now(self):
        return time.monotonic()

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["max_age_seconds"]) not in (int, float):
            raise ValidationError("Use numeric receipt age", "max_age_seconds")
        for key in ("require_context", "allow_zero_stamps"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a boolean setting", key)

    def waiting(self, message):
        return Result({p.name: NO_VALUE for p in self.outputs}, message=message)

    def process(self, inputs):
        state, p, now = self.runtime, self.parameters, self.now()
        if "context" in inputs:
            context = validate_context(inputs["context"])
            if context != state["context"]:
                state["context"], state["cached"] = context, None
        if "reset" in inputs:
            if inputs["reset"] and not state["reset"]:
                state["cached"] = None
            state["reset"] = inputs["reset"]
        context = state["context"]
        if "camera" in inputs:
            model = inputs["camera"]
            payload = self.input_payloads["camera"]
            if failed_input(ValuePacket(model, payload)):
                raise ValidationError("Failed calibration cannot enter the static cache")
            if not isinstance(model, dict) or "ros2_static_binding" in model:
                raise ValidationError("Cache original decoded CameraInfo, not an old image binding")
            revision = calibration_revision(model)
            if "acquisition" in payload:
                if context is None:
                    raise ValidationError("Tagged calibration needs its current Run identity")
                check_current(model, payload, context)
            if p["require_context"] and context is None:
                return self.waiting("Prime current Run identity before calibration")
            state["cached"] = {
                "camera": copy.deepcopy(model),
                "payload": copy.deepcopy(payload),
                "revision": revision,
                "received": now,
            }
        if "image" not in inputs:
            return self.waiting(
                "Static calibration cached" if state["cached"] else "Waiting for calibration"
            )
        image, payload = inputs["image"], copy.deepcopy(self.input_payloads["image"])
        if p["require_context"] and context is None:
            return self.waiting("Waiting for current Run identity; image not queued")
        if context is not None:
            check_current(image, payload, context)
        cached = state["cached"]
        if cached is None:
            return self.waiting("Waiting for static calibration; image not queued")
        age = now - cached["received"]
        if age < 0 or (p["max_age_seconds"] and age > p["max_age_seconds"]):
            state["cached"] = None
            return self.waiting("Static calibration expired; image not queued")
        model = copy.deepcopy(cached["camera"])
        from .recipes import unique_ref

        reference = unique_ref([ValuePacket(model, cached["payload"]), ValuePacket(image, payload)])
        if reference is not None:
            payload["recipe_ref"] = copy.deepcopy(reference)
        _, _, size = camera(model)
        ros = payload.get("ros2", {})
        header = acquisition_header(image, payload)
        if header["frame_id"] != model["frame"]:
            raise ValidationError("Image and static CameraInfo optical frames must match")
        if [image.shape[1], image.shape[0]] != list(size) or [
            ros.get("width"),
            ros.get("height"),
        ] != list(size):
            raise ValidationError("Image must use the original ROS grid matching CameraInfo")
        if not np.allclose(payload["spatial"]["to_root"], np.eye(3), atol=1e-9, rtol=0):
            raise ValidationError("Transformed pixels need corresponding updated ROS calibration")
        stamp = header["stamp"]
        stamp_ns = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        if stamp_ns < 0 or (stamp_ns == 0 and not p["allow_zero_stamps"]):
            raise ValidationError("Image acquisition stamp must be positive and nonzero")
        try:
            ticket = validate_ticket(payload.get("ticket"))
        except ValueError as exc:
            raise ValidationError("Static calibration binding requires a ticketed image") from exc
        binding = {
            "model": "ros_static_camera_binding_v1",
            "calibration_revision": cached["revision"],
            "calibration_header": copy.deepcopy(model["ros2"]["header"]),
            "calibration_ticket": copy.deepcopy(cached["payload"].get("ticket")),
            "calibration_recipe_ref": copy.deepcopy(cached["payload"].get("recipe_ref")),
            "image_header": header,
            "image_ticket": copy.deepcopy(ticket),
            "image_size": list(size),
            "receipt_age_seconds": age,
            "max_age_seconds": p["max_age_seconds"],
            "context": copy.deepcopy(context),
            "context_scope": scope_key(context) if context else None,
            "timing": "static_calibration_applied_to_image_not_new_measurement",
        }
        model["ros2_static_binding"] = binding
        payload["ros2_static_binding"] = copy.deepcopy(binding)
        return Result(
            {"image": image, "camera": model, "info": copy.deepcopy(binding)},
            payloads={key: copy.deepcopy(payload) for key in ("image", "camera", "info")},
            message="Static calibration applied to image; original calibration header retained",
        )
