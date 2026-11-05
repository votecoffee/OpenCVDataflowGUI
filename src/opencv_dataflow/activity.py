"""Bounded, thread-safe activity events shared by an entire running job."""

import copy
import json
import math
import threading
import uuid
from collections import deque
from datetime import datetime

from .outcomes import packets_in, source_info
from .tickets import ticket_key, validate_ticket
from .timing import mark


def validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("Activity event must be an object")
    kind = event.get("kind")
    if not isinstance(kind, str) or not 1 <= len(kind) <= 80:
        raise ValueError("Invalid activity kind")
    source = event.get("source", {})
    if not isinstance(source, dict) or any(
        not isinstance(source.get(k), str) for k in ("id", "name", "type", "path")
    ):
        raise ValueError("Activity requires source id/name/type/path")
    if "logical_path" in source and not isinstance(source["logical_path"], str):
        raise ValueError("Activity logical path must be text")
    invocation = event.get("invocation")
    if invocation is not None and (not isinstance(invocation, str) or len(invocation) > 128):
        raise ValueError("Invalid activity invocation")
    for key in ("tickets", "parent_tickets"):
        tickets = event.get(key)
        if not isinstance(tickets, list) or len(tickets) > 64:
            raise ValueError("Activity requires at most 64 tickets/parent tickets")
        for ticket in tickets:
            validate_ticket(ticket)
    data = event.get("data")
    if not isinstance(data, dict) or len(json.dumps(data, allow_nan=False).encode()) > 8192:
        raise ValueError("Activity data must be a JSON object of at most 8 KiB")
    for point in (
        [event.get("time"), event.get("started")] if kind == "finish" else [event.get("time")]
    ):
        if not isinstance(point, dict) or not isinstance(point.get("utc"), str):
            raise ValueError("Activity requires UTC and monotonic timestamps")
        if datetime.fromisoformat(point["utc"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("Activity timestamps require a timezone")
        if (
            type(point.get("ns")) is not int
            or point["ns"] < 0
            or not isinstance(point.get("clock"), str)
        ):
            raise ValueError("Activity requires a monotonic reading and clock identity")
    if kind == "finish":
        duration = event.get("duration_ms")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Activity duration must be finite and nonnegative")
        if (
            event["started"]["clock"] != event["time"]["clock"]
            or event["started"]["ns"] > event["time"]["ns"]
        ):
            raise ValueError("Activity span must use ordered readings from one clock")
        if not isinstance(event.get("judgment"), str):
            raise ValueError("Activity finish requires a judgment")


def identities(values):
    tickets, parents = {}, {}
    for value in values.values():
        for packet in packets_in(value):
            payload = packet.payload if isinstance(packet.payload, dict) else {}
            ticket = payload.get("ticket")
            if ticket is None and isinstance(packet.value, dict):
                if packet.value.get("model") == "flow_ticket_v1":
                    ticket = packet.value
            inspection = payload.get("inspection")
            parent = inspection.get("parent_ticket") if isinstance(inspection, dict) else None
            for target, candidate in ((tickets, ticket), (parents, parent)):
                if candidate is not None:
                    try:
                        validate_ticket(candidate)
                        target.setdefault(ticket_key(candidate), copy.deepcopy(candidate))
                    except (ValueError, TypeError, KeyError):
                        pass
    return list(tickets.values()), list(parents.values())


class ActivityLog:
    def __init__(self, max_events=20000, max_bytes=16 * 1024 * 1024):
        if any(type(n) is not int or n < 1 for n in (max_events, max_bytes)):
            raise ValueError("Activity retention limits must be positive")
        self.max_events, self.max_bytes = max_events, max_bytes
        self.lock = threading.RLock()
        self.events = deque()
        self.bytes = self.sequence = self.evicted = 0

    def emit(self, event):
        from .diagnostics import enrich

        validate_event(event)
        event = copy.deepcopy(event)
        enrich(event)
        event["model"] = "block_event_v1"
        event["id"] = uuid.uuid4().hex
        encoded = json.dumps(event, allow_nan=False).encode("utf-8")
        if len(encoded) > 256 * 1024:
            from .diagnostics import bounded_text

            event["parameters"] = {"omitted": "event byte limit"}
            event["message"] = bounded_text(event["message"], 2048)
            if event["exception"]:
                event["exception"]["traceback"] = bounded_text(
                    event["exception"]["traceback"], 4096
                )
                event["exception"]["message"] = bounded_text(event["exception"]["message"], 2048)
                event["exception"]["truncated"] = True
            event["diagnostic_truncated"] = True
            encoded = json.dumps(event, allow_nan=False).encode("utf-8")
        if len(encoded) > 256 * 1024:
            raise ValueError("Activity event exceeds 256 KiB")
        with self.lock:
            self.sequence += 1
            event["sequence"] = self.sequence
            size = len(encoded) + 32
            self.events.append((event, size))
            self.bytes += size
            while len(self.events) > self.max_events or self.bytes > self.max_bytes:
                _, removed = self.events.popleft()
                self.bytes -= removed
                self.evicted += 1

    def snapshot(self):
        with self.lock:
            snapshot = {
                "model": "block_activity_v1",
                "sequence": self.sequence,
                "evicted_events": self.evicted,
                "retained_bytes": self.bytes,
                "events": [e for e, _ in self.events],
            }
        # Stored events are immutable; copy outside the writer lock so viewers do not
        # block processing threads while copying retained history.
        return copy.deepcopy(snapshot)

    def revision(self):
        with self.lock:
            return self.sequence


def report(block, kind, data=None, values=None, invocation=None, at=None, diagnostic=None):
    from .diagnostics import parameters_snapshot

    log = getattr(block, "_activity", None)
    if log is None:
        log = getattr(getattr(block, "logger", None), "standalone", None)
    if log is None:
        return
    if not isinstance(kind, str) or not 1 <= len(kind) <= 80:
        raise ValueError("Activity kind must contain 1..80 characters")
    data = {} if data is None else data
    if not isinstance(data, dict) or len(json.dumps(data, allow_nan=False).encode()) > 8192:
        raise ValueError("Activity data must be a JSON object of at most 8 KiB")
    if values is None and getattr(block, "_execution_owner", None) == threading.get_ident():
        from .payloads import ValuePacket

        values = {
            name: ValuePacket(value, block.input_payloads.get(name, {}))
            for name, value in block.last_inputs.items()
        }
    if (
        getattr(block, "_execution_owner", None) == threading.get_ident()
        and getattr(block, "_activity_started", False)
        and getattr(block, "_source_ticket", None) is not None
    ):
        values = {**(values or {}), "source_ticket": block._source_ticket}
    tickets, parents = identities(values or {})
    source = source_info(block)
    source["path"] = getattr(block, "_activity_path", source["path"])
    source["logical_path"] = getattr(block, "_activity_logical_path", source["path"])
    if invocation is None and getattr(block, "_execution_owner", None) == threading.get_ident():
        invocation = getattr(block, "_activity_invocation", None)
    event = {
        "kind": kind,
        "invocation": invocation,
        "source": source,
        "tickets": tickets[:64],
        "parent_tickets": parents[:64],
        "omitted_tickets": max(0, len(tickets) - 64) + max(0, len(parents) - 64),
        "time": at or mark(),
        "thread": threading.get_ident(),
        "data": data,
        "parameters": parameters_snapshot(
            {
                **getattr(block, "parameters", {}),
                **(getattr(block, "last_recipe_parameters", {}) if kind == "finish" else {}),
            }
        ),
    }
    if diagnostic is not None:
        event.update(diagnostic)
    if kind == "finish":
        if block.state == "error":
            event["exception"] = getattr(block, "_diagnostic_exception", None)
        event["started"] = block._work_started
        event["duration_ms"] = block.timing.last_ms
        event["judgment"] = block.state
        result = block.last_result
        event["result"] = {
            "passed": result.passed if result else None,
            "score": result.score if result else None,
            "final_judgment": result.final_judgment if result else None,
            "message": str(result.message if result else block.error)[:2048],
        }
    log.emit(event)


def spans(snapshot):
    """Reconstruct bars, including finishes whose starts were evicted."""
    calls = {}
    for event in snapshot["events"]:
        key = event.get("invocation")
        if key is None or event["kind"] not in ("start", "finish", "ticket_assigned"):
            continue
        previous = calls.get(key)
        if event["kind"] == "ticket_assigned":
            if previous is not None:
                previous["tickets"] = event["tickets"]
            continue
        if event["kind"] == "start":
            if previous is not None and previous["finish"] is not None:
                continue
            calls[key] = {
                "invocation": key,
                "source": event["source"],
                "start": event["time"],
                "finish": None,
                "judgment": "running",
                "tickets": event["tickets"],
                "parent_tickets": event["parent_tickets"],
                "event": event,
            }
        else:
            calls[key] = {
                "invocation": key,
                "source": event["source"],
                "start": event["started"],
                "finish": event["time"],
                "judgment": event["judgment"],
                "tickets": event["tickets"],
                "parent_tickets": event["parent_tickets"],
                "event": event,
            }
    return list(calls.values())
