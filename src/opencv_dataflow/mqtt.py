"""Ticketed MQTT subscriptions and explicitly enabled publication using Paho."""

import json
import os
from collections import deque
from pathlib import Path
from time import monotonic

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .tickets import DIRECT_CLOCK
from .visual_style import INDUSTRIAL

MAX_PAYLOAD_BYTES = 1_048_576
MAX_PENDING_MESSAGES = 32


def validate_topic(topic, *, subscribe):
    if not topic or "\0" in topic or len(topic.encode("utf-8")) > 65535:
        raise ValidationError("Enter a valid MQTT topic or topic filter", "topic")
    if not subscribe and ("#" in topic or "+" in topic):
        raise ValidationError("Publish topics cannot contain MQTT wildcards", "topic")
    levels = topic.split("/")
    if any(
        ("+" in level and level != "+")
        or ("#" in level and (level != "#" or index != len(levels) - 1))
        for index, level in enumerate(levels)
    ):
        raise ValidationError("Use + for a whole level and # only as the last level", "topic")


class MQTTConnection:
    def __init__(self, settings, project_dir, *, subscribe):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("Install the mqtt extra to use MQTT blocks") from exc
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
            reconnect_on_failure=False,
        )
        self.timeout = float(settings["timeout_seconds"])
        self.client.connect_timeout = self.timeout
        self.connected = False
        self.subscribed = False
        self.pending = deque()
        self.topic = settings["topic"]
        self.topic_matches = mqtt.topic_matches_sub
        self.include_retained = settings.get("include_retained", True)
        self.client.on_connect = self.on_connect
        if subscribe:
            self.client.on_subscribe = self.on_subscribe
            self.client.on_message = self.on_message
        username, password_env = settings["username"], settings["password_env"]
        if username:
            password = os.environ.get(password_env) if password_env else None
            if password_env and password is None:
                raise ValidationError(
                    "The MQTT password environment variable is unset", "password_env"
                )
            self.client.username_pw_set(username, password)
        if settings["tls"]:
            ca_file = Path(settings["ca_file"]) if settings["ca_file"] else None
            if ca_file and not ca_file.is_absolute() and project_dir is not None:
                ca_file = project_dir / ca_file
            self.client.tls_set(ca_certs=str(ca_file) if ca_file else None)

    def open(self, settings, cancelled, *, subscribe):
        if cancelled():
            raise InterruptedError("MQTT operation stopped")
        rc = self.client.connect(settings["host"].strip(), int(settings["port"]), keepalive=60)
        if rc:
            raise ConnectionError(f"MQTT connect failed ({rc})")
        if not self.wait(lambda: self.connected, cancelled):
            raise TimeoutError("MQTT broker did not acknowledge the connection")
        if subscribe:
            rc, _ = self.client.subscribe(settings["topic"], int(settings["qos"]))
            if rc:
                raise ConnectionError(f"MQTT subscribe failed ({rc})")
            if not self.wait(lambda: self.subscribed, cancelled):
                raise TimeoutError("MQTT broker did not acknowledge the subscription")

    def on_connect(self, _client, _userdata, _flags, reason, _properties):
        if reason.is_failure:
            raise ConnectionError(f"MQTT broker refused the connection ({reason})")
        self.connected = True

    def on_subscribe(self, _client, _userdata, _mid, reasons, _properties):
        if len(reasons) != 1 or reasons[0].is_failure:
            raise ConnectionError("MQTT broker rejected the subscription")
        self.subscribed = True

    def on_message(self, _client, _userdata, message):
        if not self.topic_matches(self.topic, message.topic) or (
            message.retain and not self.include_retained
        ):
            return
        if len(message.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("MQTT payload exceeds 1 MiB")
        if len(self.pending) >= MAX_PENDING_MESSAGES:
            raise OverflowError("MQTT subscription buffer is full")
        self.pending.append(message)

    def wait(self, ready, cancelled):
        deadline = monotonic() + self.timeout
        while not ready():
            if cancelled():
                raise InterruptedError("MQTT operation stopped")
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            rc = self.client.loop(timeout=min(remaining, 0.1))
            if rc:
                raise ConnectionError(f"MQTT connection lost ({rc})")
        return True

    def close(self, *, abort=False):
        sock = self.client.socket()
        try:
            if not abort:
                self.client.disconnect()
        finally:
            if sock is not None:
                sock.close()


class MQTTBlock(Block):
    category = block_type = INDUSTRIAL
    subtype = "MQTT"
    preview_safe = False
    requires_serial_execution = True
    validation_skip_output = True
    defaults = {
        "host": "127.0.0.1",
        "port": 1883,
        "topic": "",
        "qos": 1,
        "timeout_seconds": 2.0,
        "payload_format": "json",
        "tls": False,
        "ca_file": "",
        "username": "",
        "password_env": "",
    }
    parameter_choices = {"qos": (0, 1, 2), "payload_format": ("json", "text", "bytes")}
    parameter_ranges = {
        "port": NumericRange(1, 65535, whole=True),
        "timeout_seconds": NumericRange(0.1, 30),
    }
    parameter_help = {
        "host": "Broker hostname or IP address. Use a host without a URI scheme.",
        "port": "Broker TCP port, often 1883 for TCP or 8883 for TLS.",
        "qos": "0 sends without an acknowledgment; 1 and 2 use MQTT acknowledgments.",
        "timeout_seconds": "Time limit for each connect, subscribe, receive or publish wait.",
        "payload_format": "JSON value, UTF-8 text or raw bytes.",
        "tls": "Verify the broker certificate and hostname over TLS.",
        "ca_file": "Optional PEM CA bundle. Relative paths use the saved project's directory.",
        "password_env": "Name of the environment variable that holds the password.",
    }
    subscribes = False
    enable_parameter = "write_enabled"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connection = None
        self._connection_settings = None

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("host", "topic", "ca_file", "username", "password_env"):
            if not isinstance(p[key], str):
                raise ValidationError(f"MQTT {key} must be text", key)
        for key in ("tls", self.enable_parameter, "retain", "include_retained"):
            if key in p and type(p[key]) is not bool:
                raise ValidationError(f"MQTT {key} must be boolean", key)
        if p[self.enable_parameter] and (not p["host"].strip() or "://" in p["host"]):
            raise ValidationError("Enter a broker hostname or IP address", "host")
        if p["topic"] or p[self.enable_parameter]:
            validate_topic(p["topic"], subscribe=self.subscribes)
        if p["password_env"] and not p["username"]:
            raise ValidationError(
                "A password requires an MQTT username", "username", "password_env"
            )

    def connection(self):
        settings = self.parameters.copy()
        if self._connection is None or settings != self._connection_settings:
            self.close()
            connection = MQTTConnection(settings, self.project_dir, subscribe=self.subscribes)
            self._connection = connection
            try:
                connection.open(settings, self._cancelled, subscribe=self.subscribes)
            except Exception:
                self._close_safely()
                raise
            self._connection_settings = settings
        return self._connection

    def reset_runtime(self):
        self._close_safely()

    def close(self):
        self.disconnect()

    def disconnect(self, *, abort=False):
        connection, self._connection = self._connection, None
        self._connection_settings = None
        if connection is not None:
            connection.close(abort=abort)


class ReadMQTT(MQTTBlock):
    type_id, title = "input.mqtt", "Read MQTT message"
    outputs = (Port("payload", "any"), Port("message", "data"), Port("trigger", "boolean"))
    flow_primary = ("payload",)
    subscribes = True
    enable_parameter = "enabled"
    defaults = {
        **MQTTBlock.defaults,
        "enabled": False,
        "include_retained": True,
        "ticket_field": "",
    }
    parameter_help = {
        **MQTTBlock.parameter_help,
        "topic": "Subscription topic or filter, such as cell/+/trigger.",
        "include_retained": "Accept the retained message replayed when a subscription starts.",
        "ticket_field": "Optional top-level JSON field containing an external ticket value.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["ticket_field"], str) or (
            p["ticket_field"] and p["payload_format"] != "json"
        ):
            raise ValidationError("Ticket field requires a JSON payload", "ticket_field")

    def issue_source_ticket(self):
        # Waiting for a message is not an acquisition.
        return None

    def waiting(self, message):
        return Result({port.name: NO_VALUE for port in self.outputs}, message=message)

    def process(self, inputs):
        p = self.parameters
        if p["enabled"] is not True:
            return self.waiting("MQTT read disabled")
        try:
            connection = self.connection()
            if not connection.wait(lambda: bool(connection.pending), self._cancelled):
                return self.waiting("Waiting for an MQTT message")
            message = connection.pending.popleft()
        except InterruptedError:
            self._close_safely()
            return self.waiting("MQTT read stopped")
        except Exception:
            self._close_safely()
            raise
        payload = message.payload
        try:
            if p["payload_format"] != "bytes":
                payload = payload.decode("utf-8")
            if p["payload_format"] == "json":
                payload = json.loads(payload)
                json.dumps(payload, allow_nan=False)
        except (ValueError, RecursionError) as exc:
            raise ValidationError("MQTT payload does not match the configured format") from exc

        field = p["ticket_field"]
        if field:
            if not isinstance(payload, dict) or field not in payload:
                raise ValidationError("MQTT JSON payload is missing its configured ticket field")
            self._source_ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(
                self.id,
                "external",
                payload[field],
            )
        else:
            self._source_ticket = super().issue_source_ticket()
        self._report_activity_start(inputs)
        self.report_event("ticket_assigned", values={"ticket": self._source_ticket})
        return Result(
            {
                "payload": payload,
                "message": {
                    "topic": message.topic,
                    "qos": message.qos,
                    "retained": bool(message.retain),
                },
                "trigger": True,
            }
        )


class PublishMQTT(MQTTBlock):
    type_id, title = "industrial.mqtt_publish", "Publish MQTT message"
    inputs = (Port("payload", "any"), Port("enable", "boolean"))
    outputs = (Port("published", "boolean"), Port("delivery", "data"))
    flow_primary = ("published",)
    defaults = {**MQTTBlock.defaults, "write_enabled": False, "retain": False}
    parameter_help = {
        **MQTTBlock.parameter_help,
        "topic": "Publish to one exact topic.",
        "write_enabled": "Also wire a true Enable input for each intended publication.",
        "retain": "Ask the broker to retain the last value on the topic.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uncertain = False

    def reset_runtime(self):
        super().reset_runtime()
        self._uncertain = False

    def copy_runtime_from(self, other):
        self._uncertain = other._uncertain

    def process(self, inputs):
        p = self.parameters
        if p["write_enabled"] is not True or not inputs["enable"] or self._cancelled():
            return Result(
                {port.name: NO_VALUE for port in self.outputs}, message="MQTT write disabled"
            )
        if self._uncertain:
            raise ValidationError(
                "Inspect the unconfirmed MQTT publish before resetting flow state"
            )

        payload, payload_format = inputs["payload"], p["payload_format"]
        try:
            if payload_format == "json":
                payload = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
            elif payload_format == "text" and isinstance(payload, str):
                payload = payload.encode("utf-8")
            elif payload_format != "bytes" or not isinstance(payload, bytes):
                raise ValueError("Payload type does not match its format")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError("MQTT payload does not match the configured format") from exc
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValidationError("MQTT payload exceeds 1 MiB")

        connection = self.connection()
        if self._cancelled():
            return Result(
                {port.name: NO_VALUE for port in self.outputs}, message="MQTT write stopped"
            )
        topic, qos, retain = p["topic"], int(p["qos"]), p["retain"]
        self._uncertain = True
        try:
            info = connection.client.publish(topic, payload, qos, retain)
            if info.rc or not connection.wait(info.is_published, self._cancelled):
                raise TimeoutError("MQTT publish did not complete")
        except Exception as exc:
            try:
                self.disconnect(abort=True)
            except OSError:
                self.logger.exception("Closing the unconfirmed MQTT connection failed")
            raise RuntimeError(
                "MQTT publish outcome is unknown; inspect the broker or consumer"
            ) from exc
        self._uncertain = False
        return Result(
            {
                "published": True,
                "delivery": {
                    "topic": topic,
                    "qos": qos,
                    "retained": retain,
                    "acknowledged": qos > 0,
                },
            },
            message="MQTT publish acknowledged" if qos else "MQTT publish sent",
        )


MQTT_BLOCKS = (ReadMQTT, PublishMQTT)
