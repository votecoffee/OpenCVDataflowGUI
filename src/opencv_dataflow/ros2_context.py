"""Optional run/reset identity on local wire payloads; never a motion authorization."""

import copy
import hashlib
import json
import uuid

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .payloads import FlowGroup, metadata
from .ros2_support.wire import header_value
from .tickets import DIRECT_CLOCK, validate_ticket
from .visual_style import INDUSTRIAL


def validate_context(value):
    if not isinstance(value, dict) or set(value) != {
        "model",
        "dataset_id",
        "run_id",
        "session_id",
        "generation",
    }:
        raise ValidationError("Expected a complete ros_run_context_v1 identity")
    if (
        value["model"] != "ros_run_context_v1"
        or any(
            not isinstance(value[k], str) or not 1 <= len(value[k]) <= 256 or "\0" in value[k]
            for k in ("dataset_id", "run_id", "session_id")
        )
        or type(value["generation"]) is not int
        or not 0 <= value["generation"] < 2**53
    ):
        raise ValidationError("Invalid ROS run identity fields")
    return copy.deepcopy(value)


def scope_key(context):
    encoded = json.dumps(
        validate_context(context), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return "ros-acquisition:" + hashlib.sha256(encoded).hexdigest()


def acquisition_header(value, payload):
    if isinstance(value, dict) and "header" in value:
        return header_value(value["header"])
    if isinstance(value, dict) and "header" in value.get("ros2", {}):
        return header_value(value["ros2"]["header"])
    header = payload.get("ros2", {}).get("header")
    return header_value(header)


def check_current(value, payload, context):
    context = validate_context(context)
    expected_session = scope_key(context)
    count = 0

    def visit(item, attributes, depth=0):
        nonlocal count
        if depth > 16 or count >= 4096:
            raise ValidationError("Run guard exceeds bounded group nesting")
        if isinstance(item, FlowGroup):
            for packet in item.items:
                visit(packet.value, metadata(packet), depth + 1)
            return
        count += 1
        acquisition = attributes.get("acquisition")
        if not isinstance(acquisition, dict) or acquisition.get("model") != "ros_acquisition_v1":
            raise ValidationError(
                "Input has no ROS acquisition identity; attach it before queuing work"
            )
        if validate_context(acquisition.get("context")) != context:
            raise ValidationError(
                "Input belongs to a stale or different dataset/run/reset generation"
            )

        capture = acquisition.get("capture")
        if not isinstance(capture, dict) or set(capture) != {"kind", "value"}:
            raise ValidationError("Acquisition identity is missing its capture key")
        try:
            validate_ticket(
                {
                    "model": "flow_ticket_v1",
                    "source": "run_guard",
                    "sequence": 1,
                    "session": expected_session,
                    **capture,
                }
            )
        except ValueError as exc:
            raise ValidationError("Invalid acquisition capture key") from exc
        ticket = attributes.get("ticket", {})
        if ticket.get("session", "").startswith("ros-acquisition:") and (
            ticket["session"] != expected_session
            or ticket.get("kind") != capture["kind"]
            or ticket.get("value") != capture["value"]
        ):
            raise ValidationError("Scoped ticket does not match its acquisition identity")

    visit(value, payload)
    return count


class RunIdentity(Block):
    type_id, title = "industrial.ros2_run", "ROS 2 run identity"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("reset", "boolean", required=False),)
    outputs = (Port("context", "data"), Port("run_token", "text"))
    defaults = {"dataset_id": "inspection", "run_id": ""}
    parameter_help = {
        "dataset_id": "Dataset/job identity shared by related acquisitions.",
        "run_id": (
            "Optional run label; empty generates a UUID. Runtime session nonce is "
            "always fresh after flow reset/reopen."
        ),
    }
    input_help = {
        "reset": "A rising true edge advances generation once; holding true does not repeat reset."
    }
    preview_safe = False
    requires_serial_execution = True
    payload_reference = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._runtime_id = uuid.uuid4().hex
        self.reset_runtime()

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("dataset_id", "run_id"):
            if (
                not isinstance(p[key], str)
                or len(p[key]) > 256
                or "\0" in p[key]
                or (key == "dataset_id" and not p[key])
            ):
                raise ValidationError(
                    "Use a dataset ID and optional run label up to 256 characters", key
                )

    def reset_runtime(self):
        self._context, self._last_reset, self._context_owner = None, False, None
        self._parameters_snapshot = None

    def copy_runtime_from(self, other):
        self._context, self._last_reset = copy.deepcopy(other._context), other._last_reset
        self._context_owner = other._context_owner
        self._parameters_snapshot = other._parameters_snapshot

    def process(self, inputs):
        settings = (self.parameters["dataset_id"], self.parameters["run_id"])
        if (
            self._context is None
            or self._context_owner != self._runtime_id
            or self._parameters_snapshot != settings
        ):
            self._context_owner, self._last_reset = self._runtime_id, False
            self._parameters_snapshot = settings
            self._context = {
                "model": "ros_run_context_v1",
                "dataset_id": self.parameters["dataset_id"],
                "run_id": self.parameters["run_id"] or uuid.uuid4().hex,
                "session_id": uuid.uuid4().hex,
                "generation": 0,
            }
        resetting = inputs.get("reset", False)
        if resetting and not self._last_reset:
            self._context["generation"] += 1
            validate_context(self._context)
        self._last_reset = resetting
        return Result(
            {"context": copy.deepcopy(self._context), "run_token": scope_key(self._context)}
        )


class AttachAcquisition(Block):
    type_id, title = "industrial.ros2_acquisition", "Attach ROS 2 acquisition identity"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("value", "any"), Port("context", "data"))
    outputs = (Port("value", "any"), Port("ticket", "data"), Port("acquisition", "data"))
    flow_primary = ("value",)
    payload_reference = "value"
    defaults = {"capture_mode": "ticket", "capture_field": ""}
    parameter_choices = {"capture_mode": ("ticket", "header_stamp", "field")}
    parameter_help = {
        "capture_mode": (
            "Use the incoming ticket, nonzero header stamp without frame_id, or a "
            "numeric/string data field."
        ),
        "capture_field": (
            "Dotted object field path when Field is selected. Do not replace physical "
            "frame names to correlate tickets."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["capture_field"], str) or (
            p["capture_mode"] == "field"
            and (not p["capture_field"] or any(not k for k in p["capture_field"].split(".")))
        ):
            raise ValidationError("Choose a dotted capture field path", "capture_field")

    def process(self, inputs):
        value, context = inputs["value"], validate_context(inputs["context"])
        if isinstance(value, FlowGroup):
            raise ValidationError("Attach identity to individual streams before grouping")
        payload = copy.deepcopy(self.input_payloads["value"])
        existing = payload.get("acquisition")
        if existing is not None and (
            not isinstance(existing, dict)
            or existing.get("model") != "ros_acquisition_v1"
            or validate_context(existing.get("context")) != context
        ):
            raise ValidationError(
                "Refusing to relabel an acquisition from a different run/reset generation"
            )
        previous = payload.get("ticket")
        mode = self.parameters["capture_mode"]
        clock = getattr(self, "_clock", DIRECT_CLOCK)
        if mode == "ticket":
            if previous is None:
                raise ValidationError("Ticket mode requires an input acquisition ticket")
            ticket = clock.issue(self.id, "external", previous)
        elif mode == "header_stamp":
            stamp = acquisition_header(value, payload)["stamp"]
            if stamp["sec"] * 1_000_000_000 + stamp["nanosec"] <= 0:
                raise ValidationError("Capture header stamp must be positive and nonzero")
            ticket = clock.issue(self.id, "external", f"{stamp['sec']}:{stamp['nanosec']:09d}")
        else:
            selected = value
            try:
                for key in self.parameters["capture_field"].split("."):
                    selected = selected[key]
            except (KeyError, TypeError) as exc:
                raise ValidationError("Capture field is missing") from exc
            if type(selected) not in (str, int, float):
                raise ValidationError("Capture field must be a finite number or nonempty string")
            ticket = clock.issue(self.id, "external", selected)
        ticket["session"] = scope_key(context)
        acquisition = {
            "model": "ros_acquisition_v1",
            "context": context,
            "capture": {"kind": ticket["kind"], "value": ticket["value"]},
        }
        if previous is not None:
            acquisition["original_ticket"] = copy.deepcopy(previous)
        payload.update(ticket=ticket, acquisition=acquisition)
        if hasattr(value, "metadata"):
            value.metadata = copy.deepcopy(payload)
        return Result(
            {"value": value, "ticket": ticket, "acquisition": acquisition},
            payloads={name: copy.deepcopy(payload) for name in ("value", "ticket", "acquisition")},
        )


class GuardRunIdentity(Block):
    type_id, title = "industrial.ros2_run_guard", "Validate ROS 2 run identity"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("value", "any"), Port("context", "data"))
    outputs = (Port("value", "any"), Port("info", "data"))
    payload_reference = "value"
    flow_primary = ("value",)
    defaults = {"identity_source": "payload", "run_field": "run_id"}
    parameter_choices = {"identity_source": ("payload", "envelope")}
    parameter_help = {
        "identity_source": (
            "Check full local acquisition payloads or an explicit external JSON run-token field."
        ),
        "run_field": (
            "Envelope mode: dotted field containing the current Run identity run_token; "
            "no acquisition tag is invented."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["run_field"], str) or (
            p["identity_source"] == "envelope"
            and (not p["run_field"] or any(not k for k in p["run_field"].split(".")))
        ):
            raise ValidationError("Choose a dotted envelope run-token field", "run_field")

    def process(self, inputs):
        context = validate_context(inputs["context"])
        envelope = self.parameters["identity_source"] == "envelope"
        if envelope:
            selected = inputs["value"]
            try:
                for key in self.parameters["run_field"].split("."):
                    selected = selected[key]
            except (KeyError, TypeError) as exc:
                raise ValidationError("Response envelope lacks its run token") from exc
            if type(selected) is not str or selected != scope_key(context):
                raise ValidationError("Response envelope belongs to a stale/different run")
            count = 1
        else:
            count = check_current(inputs["value"], self.input_payloads["value"], context)
        return Result(
            {
                "value": inputs["value"],
                "info": {
                    "context": validate_context(inputs["context"]),
                    "checked_members": count,
                    "current": True,
                    "identity_source": self.parameters["identity_source"],
                    "capture_checked": not envelope,
                },
            },
            payloads={"value": copy.deepcopy(self.input_payloads["value"])},
        )


ROS2_CONTEXT_BLOCKS = (RunIdentity, AttachAcquisition, GuardRunIdentity)
