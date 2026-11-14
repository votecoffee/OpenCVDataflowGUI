"""Read-only, persistent exact-time TF lookup through the optional ROS worker."""

import copy
import re

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .ros2_client import ROSClient, ROSRemoteError
from .ros2_context import acquisition_header, check_current, scope_key
from .ros2_poses import frame_name, pose_from_ros
from .visual_style import INDUSTRIAL


class LookupROSTF(Block):
    type_id, title = "industrial.ros2_tf", "Lookup ROS 2 transform at acquisition"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    preview_safe = False
    requires_serial_execution = True
    validation_skip_output = True
    payload_reference = "value"
    flow_primary = ("value",)
    inputs = (
        Port("value", "any"),
        Port("context", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("value", "any"),
        Port("pose", "data", visual_type="pose"),
        Port("message", "data"),
        Port("info", "data"),
    )
    defaults = {
        "enabled": False,
        "domain_id": 71,
        "namespace": "/opencv_dataflow",
        "discovery_range": "localhost",
        "tf_topic": "/tf",
        "tf_static_topic": "/tf_static",
        "clock_topic": "/clock",
        "source_frame": "",
        "target_frame": "base_link",
        "timeout_seconds": 0.25,
        "startup_timeout_seconds": 30.0,
        "cache_seconds": 10.0,
        "use_sim_time": False,
        "max_age_seconds": 1.0,
        "future_tolerance_seconds": 0.05,
        "units": "m",
        "unavailable": "error",
    }
    parameter_choices = {
        "discovery_range": ("localhost", "subnet"),
        "units": ("m", "mm"),
        "unavailable": ("error", "wait"),
    }
    parameter_ranges = {
        "domain_id": NumericRange(0, 232, whole=True),
        "timeout_seconds": NumericRange(0.01, 30),
        "startup_timeout_seconds": NumericRange(1, 60),
        "cache_seconds": NumericRange(0.1, 120),
        "max_age_seconds": NumericRange(0, 3600),
        "future_tolerance_seconds": NumericRange(0, 30),
    }
    parameter_help = {
        "enabled": "Start a persistent read-only TF listener. Disabled blocks never start ROS.",
        "domain_id": "DDS domain shared with the transform/clock publishers.",
        "namespace": "Namespace for the private TF worker node.",
        "discovery_range": "Localhost by default; subnet opts in to network discovery.",
        "tf_topic": "Dynamic TransformListener topic; standard /tf or a private test topic.",
        "tf_static_topic": "Static TransformListener topic using standard transient-local TF QoS.",
        "clock_topic": "Node clock topic when Use sim time is enabled.",
        "source_frame": (
            "Empty uses the acquisition header frame. Pose maps this source into Target frame."
        ),
        "target_frame": "Explicit destination/parent frame, e.g. base_link.",
        "timeout_seconds": (
            "Maximum wall-clock wait for the exact requested transform; no latest fallback."
        ),
        "startup_timeout_seconds": "Bounded startup of the separate configured ROS Python worker.",
        "cache_seconds": (
            "TF2 time cache. Start the listener before collecting dynamic acquisitions."
        ),
        "use_sim_time": (
            "Use the selected ROS clock; observed backward jumps latch until reset/new context."
        ),
        "max_age_seconds": (
            "Acquisition age against the node clock. Zero explicitly disables age "
            "checking for historical queries."
        ),
        "future_tolerance_seconds": "Allowed lead over node time when age checking is enabled.",
        "units": "TF translations are meters; optionally convert the app pose to millimeters.",
        "unavailable": (
            "Report unavailable exact-time TF as an error or wait without an output event."
        ),
    }
    input_help = {
        "value": "Ticketed image/data/pose carrying a positive ROS acquisition header stamp.",
        "context": (
            "Optional current run identity: reject stale acquisition payloads before lookup."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = self._settings = self._scope = None
        self._reset_required = False

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("enabled", "use_sim_time"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a boolean switch", key)
        for key in self.parameter_ranges:
            if type(p[key]) not in (int, float):
                raise ValidationError("Use numeric timing/domain settings", key)
        for key in ("tf_topic", "tf_static_topic", "clock_topic", "namespace"):
            if (
                not isinstance(p[key], str)
                or len(p[key]) > 240
                or not (key == "namespace" and p[key] == "/")
                and not re.fullmatch(r"/?[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*", p[key])
            ):
                raise ValidationError("Use a slash-separated ROS name", key)
        frame_name(p["target_frame"])
        if p["source_frame"] != "":
            frame_name(p["source_frame"])

    def connection(self):
        if self._client is None or self._settings != self.parameters:
            self.close()
            config = {
                **self.parameters,
                "kind": "tf",
                "role": "lookup",
                "publication_mode": "asynchronous",
            }
            self._client = ROSClient(
                config, self.parameters["startup_timeout_seconds"], self._cancelled
            )
            self._settings = copy.deepcopy(self.parameters)
        return self._client

    def close(self):
        client, self._client = self._client, None
        self._settings = None
        if client is not None:
            client.close()

    def reset_runtime(self):
        self.close()
        self._reset_required, self._scope = False, None

    def copy_runtime_from(self, other):
        self._reset_required, self._scope = other._reset_required, other._scope

    def waiting(self, message):
        return Result({p.name: NO_VALUE for p in self.outputs}, message=message)

    def process(self, inputs):
        p = self.parameters
        if not p["enabled"]:
            self.close()
            return self.waiting("TF lookup disabled")
        attributes = self.input_payloads["value"]
        if "context" in inputs:
            check_current(inputs["value"], attributes, inputs["context"])
            current = scope_key(inputs["context"])
            if self._scope is not None and current != self._scope:
                self.close()
                self._reset_required = False
            self._scope = current
        if self._reset_required:
            raise ValidationError(
                "TF clock changed; reset flow or supply a new current run context"
            )
        header = acquisition_header(inputs["value"], attributes)
        stamp = header["stamp"]
        if stamp["sec"] * 1_000_000_000 + stamp["nanosec"] <= 0:
            raise ValidationError(
                "TF lookup requires positive acquisition time; zero/latest is forbidden"
            )
        source, target = frame_name(p["source_frame"] or header["frame_id"]), p["target_frame"]
        try:
            client = self.connection()
            response = client.request(
                "lookup",
                p["timeout_seconds"] + 2,
                self._cancelled,
                target=target,
                source=source,
                stamp=stamp,
                timeout=p["timeout_seconds"],
                expected_epoch=client.info["tf_epoch"],
                max_age=p["max_age_seconds"],
                future_tolerance=p["future_tolerance_seconds"],
            )
        except InterruptedError:
            self._close_safely()
            return self.waiting("TF lookup stopped")
        except ROSRemoteError as exc:
            if exc.code == "TFClockChanged":
                self._reset_required = True
                self._close_safely()
            if exc.code == "TFUnavailable" and p["unavailable"] == "wait":
                return self.waiting(str(exc))
            raise ValidationError(str(exc)) from exc
        except Exception:
            self._close_safely()
            raise
        pose, _ = pose_from_ros(
            response["message"], expected_parent=target, expected_child=source, units=p["units"]
        )
        info = {**response["info"], "context_checked": "context" in inputs}
        payload = copy.deepcopy(attributes)
        payload["tf_lookup"] = copy.deepcopy(info)
        return Result(
            {"value": inputs["value"], "pose": pose, "message": response["message"], "info": info},
            payloads={
                name: copy.deepcopy(payload) for name in ("value", "pose", "message", "info")
            },
        )
