"""Explicit ControlLogix tag I/O through the optional pycomm3 client."""

import json
import math
import struct

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .tickets import DIRECT_CLOCK
from .visual_style import INDUSTRIAL


def open_logix(route, timeout, program_tags):
    try:
        from pycomm3 import LogixDriver
    except ImportError as exc:
        raise RuntimeError("Install the logix extra to use ControlLogix tags") from exc
    client = LogixDriver(route, init_tags=True, init_program_tags=program_tags)
    client.socket_timeout = timeout
    try:
        if not client.open():
            raise ConnectionError("ControlLogix connection did not open")
    except Exception:
        client.close()
        raise
    return client


def check_reply(reply, address):
    if reply is None or reply.tag not in (address, address.split("{", 1)[0]):
        raise ConnectionError(f"ControlLogix returned an unexpected reply for {address}")
    if reply.error:
        raise RuntimeError(f"ControlLogix tag {address}: {reply.error}")
    return reply.value


def tag_name(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(
        char.isspace() or ord(char) < 32 for char in value
    ):
        raise ValidationError("Enter a nonempty Logix tag address without spaces")
    return value


class LogixBlock(Block):
    category = block_type = INDUSTRIAL
    subtype = "ControlLogix"
    preview_safe = False
    requires_serial_execution = True
    validation_skip_output = True
    defaults = {"route": "", "timeout_seconds": 2.0}
    parameter_ranges = {"timeout_seconds": NumericRange(0.1, 30)}
    parameter_help = {"route": "Controller IP or explicit CIP route, such as 192.0.2.1/0."}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
        self._connection = None

    def client(self, tags):
        route = self.parameters["route"].strip()
        if not route:
            raise ValidationError("Configure the ControlLogix route", "route")
        options = (
            route, float(self.parameters["timeout_seconds"]),
            any(tag.casefold().startswith("program:") for tag in tags),
        )
        if self._client is None or options != self._connection:
            self.close()
            self._client = open_logix(*options)
            self._connection = options
        return self._client

    def close(self):
        client, self._client = self._client, None
        self._connection = None
        if client is not None:
            client.close()


class ReadLogixTags(LogixBlock):
    type_id, title = "input.logix_tags", "Read Logix tags and trigger"
    outputs = (Port("values", "data"), Port("value", "any"), Port("trigger", "boolean"))
    flow_primary = ("values",)
    defaults = {
        **LogixBlock.defaults,
        "enabled": False,
        "tags": [],
        "trigger_tag": "",
        "trigger_mode": "rising",
        "ticket_tag": "",
        "value_tag": "",
    }
    parameter_choices = {"trigger_mode": ("rising", "change", "high")}
    parameter_help = {
        **LogixBlock.parameter_help,
        "tags": "Ordered JSON list of up to 32 named tags. Read with the trigger and ticket tags.",
        "trigger_tag": "Optional BOOL or integer tag. With no trigger tag, emit each poll.",
        "trigger_mode": "Rising and change establish a baseline on the first successful poll.",
        "ticket_tag": "Optional numeric or text sequence tag to use as the acquisition ticket.",
        "value_tag": "Optional tag to expose as an individual value port.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._previous = None

    def reset_runtime(self):
        self._previous = None

    def copy_runtime_from(self, other):
        self._previous = other._previous

    def issue_source_ticket(self):
        # A poll alone is not an acquisition. Assign a ticket only after the trigger admits it.
        return None

    def process(self, inputs):
        p = self.parameters
        waiting = Result({port.name: NO_VALUE for port in self.outputs}, message="No PLC trigger")
        if p["enabled"] is not True:
            waiting.message = "Logix read disabled"
            return waiting
        tags = p["tags"]
        if not isinstance(tags, list) or len(tags) > 32:
            raise ValidationError("Configure up to 32 tag addresses", "tags")
        trigger, ticket_tag, value_tag = p["trigger_tag"], p["ticket_tag"], p["value_tag"]
        addresses = tuple(dict.fromkeys(tag_name(tag) for tag in (
            *tags, *(tag for tag in (trigger, ticket_tag, value_tag) if tag != "")
        )))
        if not addresses:
            raise ValidationError("Configure a tag or trigger to read", "tags", "trigger_tag")
        try:
            reply = self.client(addresses).read(*addresses)
            replies = reply if isinstance(reply, list) else [reply]
            if len(replies) != len(addresses):
                raise ConnectionError("ControlLogix returned an incomplete tag batch")
            values = {tag: check_reply(item, tag)
                      for tag, item in zip(addresses, replies, strict=True)}
            json.dumps(values, allow_nan=False)
        except Exception:
            self._close_safely()
            raise

        if trigger:
            value = values[trigger]
            if type(value) not in (bool, int):
                raise ValidationError("Use a BOOL or integer trigger tag", "trigger_tag")
            previous, self._previous = self._previous, value
            mode = p["trigger_mode"]
            if mode == "high":
                admitted = bool(value)
            elif mode == "change":
                admitted = previous is not None and previous != value
            else:
                admitted = previous is not None and not previous and bool(value)
            if not admitted:
                return waiting

        if ticket_tag:
            self._source_ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(
                self.id, "external", values[ticket_tag]
            )
        else:
            self._source_ticket = super().issue_source_ticket()
        self._report_activity_start(inputs)
        self.report_event("ticket_assigned", values={"ticket": self._source_ticket})
        return Result(
            {"values": values, "value": values.get(value_tag, NO_VALUE), "trigger": True},
            message=f"Read {len(values)} Logix tags",
        )


def write_matches(expected, reply):
    actual = reply.value
    if type(expected) is float and reply.type == "REAL":
        expected = struct.unpack("<f", struct.pack("<f", expected))[0]
    return type(actual) is type(expected) and actual == expected


class WriteLogixTag(LogixBlock):
    type_id, title = "industrial.logix_write", "Write Logix tag"
    inputs = (Port("value", "any"), Port("enable", "boolean"))
    outputs = (Port("written", "boolean"), Port("readback", "any"))
    flow_primary = ("written",)
    defaults = {**LogixBlock.defaults, "tag": "", "write_enabled": False}
    parameter_help = {
        **LogixBlock.parameter_help,
        "tag": "One explicit scalar destination, including a specific UDT member or array element.",
        "write_enabled": "Also wire a true enable input for each intended write.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uncertain = False

    def reset_runtime(self):
        self._uncertain = False

    def copy_runtime_from(self, other):
        self._uncertain = other._uncertain

    def process(self, inputs):
        if self.parameters["write_enabled"] is not True or not inputs["enable"]:
            return Result(
                {"written": NO_VALUE, "readback": NO_VALUE}, message="Logix write disabled"
            )
        if self._uncertain:
            raise ValidationError("Inspect the unverified Logix write before resetting flow state")
        address = tag_name(self.parameters["tag"])
        if "{" in address:
            raise ValidationError("Write one scalar tag or element at a time", "tag")
        value = inputs["value"]
        if type(value) not in (bool, int, float, str) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ValidationError("Logix writes need a finite scalar value")
        try:
            client = self.client((address,))
            before = client.read(address)
            current = check_reply(before, address)
            if type(current) is not type(value):
                raise ValidationError(f"Value type must match the configured Logix tag {address}")
            if before.type == "REAL" and type(value) is float:
                try:
                    struct.pack("<f", value)
                except OverflowError as exc:
                    raise ValidationError(
                        f"Value exceeds the Logix REAL range for {address}"
                    ) from exc
            self._uncertain = True
            check_reply(client.write((address, value)), address)
            reply = client.read(address)
            actual = check_reply(reply, address)
            if not write_matches(value, reply):
                raise RuntimeError(f"Logix write readback differs for {address}")
            self._uncertain = False
        except Exception:
            self._close_safely()
            raise
        return Result({"written": True, "readback": actual}, message=f"Verified {address}")


LOGIX_BLOCKS = (ReadLogixTags, WriteLogixTag)
