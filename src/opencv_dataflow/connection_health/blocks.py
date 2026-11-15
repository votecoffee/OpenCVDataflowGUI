"""Scheduled connection status with a shared provider and diagnostic contract."""

import copy
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from ..core import Block, Port, Result
from ..parameter_validation import NumericRange, ValidationError
from ..visual_style import INDUSTRIAL
from .providers import (
    EtherNetIPProbe,
    FanucProbe,
    LogixProbe,
    MQTTProbe,
    ROSProbe,
    endpoint_identity,
)

NAME = r"/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*"
MSG = r"[a-z][a-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*"
SRV = r"[a-z][a-z0-9_]*/srv/[A-Za-z][A-Za-z0-9_]*"


def text_list(value, field, *, pattern=None):
    if (
        not isinstance(value, list)
        or len(value) > 16
        or any(
            not isinstance(v, str)
            or not v.strip()
            or len(v) > 256
            or (pattern and not re.fullmatch(pattern, v))
            for v in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValidationError("Use up to sixteen unique valid resource names", field)


class ConnectionHealth(Block):
    """Independent root block. Each instance owns its probe; logs share endpoint identity."""

    category = block_type = INDUSTRIAL
    preview_safe = False
    parallel_safe = True
    validation_skip_output = True
    probe_factory = None
    outputs = (
        Port("ok", "boolean"),
        Port("warning", "boolean"),
        Port("alarm", "boolean"),
        Port("connected", "boolean"),
        Port("state", "text"),
        Port("updated", "boolean"),
        Port("report", "data"),
        Port("logs", "data"),
    )
    defaults = {
        "enabled": False,
        "poll_interval_seconds": 2.0,
        "timeout_seconds": 2.0,
        "fail_on_alarm": False,
        "include_logs": False,
        "stale_after_seconds": 10.0,
    }
    parameter_ranges = {
        "poll_interval_seconds": NumericRange(0.1, 3600),
        "timeout_seconds": NumericRange(0.1, 30),
        "stale_after_seconds": NumericRange(0.1, 7200),
    }
    parameter_help = {
        "enabled": "Permit read-only network probes during Run/Continuous. No I/O when disabled.",
        "poll_interval_seconds": "Minimum interval between probes; each graph cycle emits status.",
        "timeout_seconds": "Per-request timeout. Some checks require multiple bounded requests.",
        "fail_on_alarm": "Also fail the graph inspection on a red status; off by default.",
        "include_logs": "Include up to twenty recent diagnostics from blocks on the same endpoint.",
        "stale_after_seconds": "Thumbnail becomes amber when its last check is older than this. "
        "Must be at least the polling interval; outputs update during runs.",
    }
    plugin_dialog_label = "Connection diagnostics…"
    plugin_dialog_modeless = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = []
        self._probe = self._settings = None
        self._report = None
        self._checked = self._next_check = 0.0
        self._connection_sources = ()
        self._logged_state = None

    def bind_connection_sources(self, blocks):
        self._connection_sources = blocks

    def related_log_sources(self):
        own = endpoint_identity(self)
        paths = {self.logger.name}

        def walk(blocks):
            for block in tuple(blocks):
                yield block
                child = getattr(block, "graph", None)
                if child is not None:
                    yield from walk(child.blocks.values())

        for block in walk(self._connection_sources):
            try:
                if own is not None and endpoint_identity(block) == own:
                    paths.add(block.logger.name)
            except ValueError, TypeError, KeyError:
                continue  # An unfinished peer's configuration cannot break the observer.
        return paths

    def recent_logs(self):
        paths = self.related_log_sources()
        log = self._activity or self.logger.standalone
        return [
            e
            for e in log.snapshot()["events"]
            if e["kind"] in ("diagnostic", "exception")
            and e["source"].get("logical_path", e["source"]["path"]) in paths
        ][-20:]

    def create_plugin_dialog(self, parent, parameters, context):
        from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

        from ..diagnostic_viewer import DiagnosticViewer

        dialog = QDialog(parent)
        dialog.setWindowTitle(f"Connection diagnostics — {self.name}")
        dialog.resize(1100, 650)
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Aggregated application logs for this endpoint. "
                "This window performs no network I/O."
            )
        )
        viewer = DiagnosticViewer(lambda: self._activity or self.logger.standalone, dialog)
        viewer.set_sources(self.related_log_sources, "Connection endpoint")
        layout.addWidget(viewer)
        return dialog

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("enabled", "fail_on_alarm", "include_logs"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a boolean setting", key)
        if p["stale_after_seconds"] < p["poll_interval_seconds"]:
            raise ValidationError(
                "Stale limit must be at least the poll interval",
                "stale_after_seconds",
                "poll_interval_seconds",
            )

    def close(self):
        probe, self._probe = self._probe, None
        self._next_check = 0.0
        if probe is not None:
            probe.close()

    def reset_runtime(self):
        self.close()
        self._settings = self._report = self._logged_state = None

    def copy_runtime_from(self, other):
        self._report = copy.deepcopy(other._report)
        self._checked, self._next_check = other._checked, other._next_check
        self._logged_state = other._logged_state

    def status_indicator(self):
        report = self.last_result.outputs["report"] if self.last_result else None
        state = (
            report["state"]
            if report
            else "alarm"
            if self.error
            else ("unchecked" if self.parameters["enabled"] else "disabled")
        )
        detail = (
            report["message"] if report else "Run to check" if not self.error else "Check error"
        )
        title = "NOT CHECKED" if state == "unchecked" else state.upper()
        if report and report["checked_utc"]:
            checked = datetime.fromisoformat(report["checked_utc"])
            age = (datetime.now(timezone.utc) - checked).total_seconds()
            detail = f"Checked {checked:%H:%M:%S} UTC\n{detail}"
            if age > self.parameters["stale_after_seconds"]:
                state, title = "warning", "STALE"
        return {"state": state, "title": title, "detail": detail}

    def process(self, inputs):
        p, now = self.parameters, time.monotonic()
        updated = False
        if not p["enabled"]:
            self.close()
            report = {
                "state": "disabled",
                "connected": False,
                "resources": [],
                "message": "Monitoring disabled",
                "checked_utc": None,
                "latency_ms": None,
            }
        else:
            if p != self._settings:
                self.close()
                self._settings, self._report = copy.deepcopy(p), None
            if self._report is None or now >= self._next_check:
                started = time.monotonic()
                try:
                    if self._probe is None:
                        self._probe = self.probe_factory()
                    report = self._probe.check(p, self.project_dir, self._cancelled)
                    if (
                        type(report.get("connected")) is not bool
                        or not isinstance(report.get("message"), str)
                        or not isinstance(report.get("resources"), list)
                        or len(report["resources"]) > 64
                    ):
                        raise ValueError("Invalid health-provider response")
                    if len(json.dumps(report, allow_nan=False).encode()) > 262144:
                        raise ValueError("Health-provider report exceeds 256 KiB")
                    states = [r["state"] for r in report["resources"]]
                    if any(s not in ("ok", "warning", "alarm") for s in states):
                        raise ValueError("Invalid resource health state")
                    report["state"] = (
                        "alarm"
                        if not report["connected"] or "alarm" in states
                        else "warning"
                        if "warning" in states
                        else "ok"
                    )
                    faults = [r for r in report["resources"] if r["state"] != "ok"]
                    if faults:
                        report["message"] = (
                            f"{len(faults)} resource fault(s): {faults[0]['message']}"
                        )
                except InterruptedError:
                    self.close()
                    raise
                except Exception as exc:
                    self._close_safely()
                    report = {
                        "connected": False,
                        "state": "alarm",
                        "resources": [],
                        "message": "Endpoint did not return a valid protocol response",
                        "error_type": type(exc).__name__,
                    }
                    self.logger.exception("Connection health probe failed")
                self._checked = time.monotonic()
                report.update(
                    checked_utc=datetime.now(timezone.utc).isoformat(),
                    latency_ms=(self._checked - started) * 1000,
                )
                self._report = copy.deepcopy(report)
                self._next_check = self._checked + p["poll_interval_seconds"]
                updated = True
            report = copy.deepcopy(self._report)
        state = report["state"]
        report.update(
            model="connection_health_v1",
            protocol=self.subtype,
            age_seconds=max(0, time.monotonic() - self._checked) if report["checked_utc"] else None,
            updated=updated,
            log_sources=sorted(self.related_log_sources()),
        )
        if state != self._logged_state:
            self._logged_state = state
            level = "error" if state == "alarm" else "warning" if state == "warning" else "info"
            getattr(self.logger, level)(f"Connection health {state}: {report['message']}")
        failed = p["fail_on_alarm"] and state == "alarm"
        return Result(
            {
                "ok": state == "ok",
                "warning": state == "warning",
                "alarm": state == "alarm",
                "connected": report["connected"],
                "state": state,
                "updated": updated,
                "report": report,
                "logs": self.recent_logs() if p["include_logs"] else [],
            },
            passed=not failed,
            message=f"{self.subtype} health: {state}",
            fail_reasons=[{"code": "connection_alarm", "message": report["message"]}]
            if failed
            else [],
            event_data={"connection_health": report},
        )


class ROSHealth(ConnectionHealth):
    type_id, title = "industrial.ros2_health", "Monitor ROS 2 health"
    subtype = "ROS 2"
    probe_factory = ROSProbe
    defaults = {
        **ConnectionHealth.defaults,
        "domain_id": 71,
        "namespace": "/opencv_dataflow",
        "discovery_range": "localhost",
        "startup_timeout_seconds": 30.0,
        "required_nodes": [],
        "required_services": [],
        "topics": [],
        "reliability": "best_effort",
        "durability": "volatile",
    }
    parameter_choices = {
        "discovery_range": ("localhost", "subnet"),
        "reliability": ("reliable", "best_effort"),
        "durability": ("volatile", "transient_local"),
    }
    parameter_ranges = {
        **ConnectionHealth.parameter_ranges,
        "domain_id": NumericRange(0, 232, whole=True),
        "startup_timeout_seconds": NumericRange(1, 60),
    }
    parameter_help = {
        **ConnectionHealth.parameter_help,
        "required_nodes": 'Fully qualified names, e.g. ["/camera/driver"]. Missing nodes alarm.',
        "required_services": 'List of {"name":"/driver/health","type":"std_srvs/srv/Trigger"}. '
        "Discovery by default; missing services alarm. check_response=true is supported only "
        "for read-only rcl_interfaces/srv/GetParameters with an empty names request.",
        "topics": 'List of {"name":"/camera/image","type":"sensor_msgs/msg/Image",'
        '"minimum_publishers":1,"max_age_seconds":2}. Resource faults warn. '
        "Age zero disables receipt freshness checks; payloads are discarded.",
        "timeout_seconds": "Discovery/response observation window; worker RPC is bounded.",
        "namespace": "Namespace of the observer node. Resource names must be fully qualified.",
        "domain_id": "DDS domain to observe. There is no central ROS server.",
        "discovery_range": "Localhost for local testing; subnet for configured remote ROS peers.",
        "startup_timeout_seconds": "Bounded time to start the configured trusted Lyrical worker.",
        "reliability": "Reliability requested by observer subscriptions; affects matching.",
        "durability": "Durability requested by observer subscriptions.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["namespace"] != "/" and not re.fullmatch(NAME, str(p["namespace"])):
            raise ValidationError("Use an absolute ROS namespace", "namespace")
        text_list(p["required_nodes"], "required_nodes", pattern=NAME)
        for key in ("required_services", "topics"):
            if not isinstance(p[key], list) or len(p[key]) > 16:
                raise ValidationError("Use up to sixteen resource objects", key)
            names = []
            for row in p[key]:
                allowed = {"name", "type"} | (
                    {"minimum_publishers", "max_age_seconds"}
                    if key == "topics"
                    else {"check_response"}
                )
                if (
                    not isinstance(row, dict)
                    or row.keys() - allowed
                    or not re.fullmatch(NAME, str(row.get("name", "")))
                    or not re.fullmatch(
                        SRV if key == "required_services" else MSG, str(row.get("type", ""))
                    )
                ):
                    raise ValidationError(
                        "Use absolute names and installed ROS interface types", key
                    )
                names.append(row["name"])
                if key == "required_services":
                    if type(row.get("check_response", False)) is not bool:
                        raise ValidationError("Check response must be boolean", key)
                    if (
                        row.get("check_response")
                        and row["type"] != "rcl_interfaces/srv/GetParameters"
                    ):
                        raise ValidationError(
                            "Response probes only support read-only GetParameters with empty names",
                            key,
                        )
                if key == "topics":
                    NumericRange(1, 64, whole=True).check(
                        row.get("minimum_publishers", 1), key, key
                    )
                    NumericRange(0, 3600).check(row.get("max_age_seconds", 0), key, key)
            text_list(names, key)


class MQTTHealth(ConnectionHealth):
    type_id, title = "industrial.mqtt_health", "Monitor MQTT health"
    subtype = "MQTT"
    probe_factory = MQTTProbe
    project_path_parameters = ("ca_file",)
    defaults = {
        **ConnectionHealth.defaults,
        "host": "127.0.0.1",
        "port": 1883,
        "tls": False,
        "ca_file": "",
        "username": "",
        "password_env": "",
        "qos": 0,
        "subscriptions": [],
        "require_message": False,
        "include_retained": True,
    }
    parameter_choices = {"qos": (0, 1, 2)}
    parameter_ranges = {
        **ConnectionHealth.parameter_ranges,
        "port": NumericRange(1, 65535, whole=True),
    }
    parameter_help = {
        **ConnectionHealth.parameter_help,
        "host": "Broker hostname/IP; every probe requires a fresh CONNACK, not only an open port.",
        "port": "MQTT TCP/TLS port.",
        "tls": "Verify TLS certificate and hostname.",
        "ca_file": "Optional PEM CA bundle, relative to the project or absolute.",
        "username": "MQTT username.",
        "password_env": "Environment-variable name, not a password.",
        "subscriptions": "Up to sixteen MQTT topic filters; rejected/missing SUBACK warns.",
        "require_message": "Also warn if a filter receives no message during the probe window.",
        "include_retained": "Count retained messages as receipt evidence when enabled.",
        "qos": "Requested subscription QoS; this monitor never publishes application data.",
    }

    def validate_parameters(self, parameters=None):
        from ..mqtt import validate_topic

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for k in ("host", "ca_file", "username", "password_env"):
            if not isinstance(p[k], str):
                raise ValidationError("Use text", k)
        for k in ("tls", "require_message", "include_retained"):
            if type(p[k]) is not bool:
                raise ValidationError("Use a boolean", k)
        if not p["host"].strip() or "://" in p["host"]:
            raise ValidationError("Use a broker hostname/IP", "host")
        if p["password_env"] and not p["username"]:
            raise ValidationError("Password environment requires a username", "username")
        text_list(p["subscriptions"], "subscriptions")
        for topic in p["subscriptions"]:
            validate_topic(topic, subscribe=True)


class EtherNetIPHealth(ConnectionHealth):
    type_id, title = "industrial.ethernetip_health", "Monitor EtherNet/IP health"
    subtype = "EtherNet/IP"
    probe_factory = EtherNetIPProbe
    defaults = {**ConnectionHealth.defaults, "route": ""}
    parameter_help = {
        **ConnectionHealth.parameter_help,
        "route": "IP/CIP route. Read-only CIP Identity request; no outputs or tag writes.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["route"], str) or (p["enabled"] and not p["route"].strip()):
            raise ValidationError("Configure an EtherNet/IP route", "route")


class LogixHealth(EtherNetIPHealth):
    type_id, title = "industrial.logix_health", "Monitor ControlLogix health"
    subtype = "ControlLogix"
    probe_factory = LogixProbe
    defaults = {**EtherNetIPHealth.defaults, "tags": []}
    parameter_help = {
        **EtherNetIPHealth.parameter_help,
        "tags": "Optional named tag reads. Individual faults warn if CIP Identity still answers.",
    }

    def validate_parameters(self, parameters=None):
        from ..logix import tag_name

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        text_list(p["tags"], "tags")
        for tag in p["tags"]:
            tag_name(tag)


class FanucHealth(ConnectionHealth):
    type_id, title = "industrial.fanuc_health", "Monitor FANUC health"
    subtype = "FANUC"
    probe_factory = FanucProbe
    project_path_parameters = ("profile",)
    defaults = {
        **ConnectionHealth.defaults,
        "profile": "",
        "robot_id": "",
        "group": 1,
        "registers": [],
    }
    parameter_ranges = {
        **ConnectionHealth.parameter_ranges,
        "group": NumericRange(1, 8, whole=True),
    }
    parameter_help = {
        **ConnectionHealth.parameter_help,
        "profile": "Configured FANUC Workbench policy file; uses its normal read permissions.",
        "robot_id": "Configured Workbench robot ID. No motion or register writes are performed.",
        "group": "Motion group for the controller-status heartbeat.",
        "registers": "Optional R[index]/PR[group,index] reads. Isolated access faults warn.",
    }

    def validate_parameters(self, parameters=None):
        from ..fanuc import _TARGET

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if any(not isinstance(p[k], str) for k in ("profile", "robot_id")):
            raise ValidationError("Use profile and robot ID text")
        if p["enabled"] and (not p["robot_id"].strip() or not p["profile"].strip()):
            raise ValidationError("Configure the FANUC profile and robot", "profile", "robot_id")
        if p["enabled"] and not (Path(self.project_dir or Path.cwd()) / p["profile"]).is_file():
            raise ValidationError("FANUC Workbench profile does not exist", "profile")
        text_list(p["registers"], "registers", pattern=_TARGET)


CONNECTION_HEALTH_BLOCKS = (ROSHealth, MQTTHealth, EtherNetIPHealth, LogixHealth, FanucHealth)
