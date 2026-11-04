"""Typed acquisition identities and an execution-session ticket clock."""

import copy
import math
import threading
import uuid
from datetime import datetime, timedelta, timezone

TICKET_MODES = ("auto_number", "auto_timestamp", "external", "fixed")
DEFAULT_TICKET_SETTINGS = {"mode": "auto_number", "value": 1}


def ticket_key(ticket):
    validate_ticket(ticket)
    value = ticket["value"]
    if ticket["kind"] == "timestamp":
        value = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
        )
    return ticket["session"], ticket["kind"], value


def validate_ticket(ticket):
    if not isinstance(ticket, dict) or ticket.get("model") != "flow_ticket_v1":
        raise ValueError("Expected a flow_ticket_v1 ticket")
    kind, value = ticket.get("kind"), ticket.get("value")
    if kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Ticket number must be finite and cannot be a boolean")
    elif kind in ("string", "timestamp"):
        if not isinstance(value, str) or not 1 <= len(value) <= 1024:
            raise ValueError("Ticket text must contain 1..1024 characters")
        if kind == "timestamp":
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                raise ValueError("Timestamp tickets require a timezone")
    else:
        raise ValueError("Ticket kind must be number, string or timestamp")
    if (
        not isinstance(ticket.get("session"), str)
        or not ticket["session"]
        or not isinstance(ticket.get("source"), str)
        or type(ticket.get("sequence")) is not int
        or ticket["sequence"] < 1
    ):
        raise ValueError("Ticket needs a session, source and positive sequence")
    return ticket


class TicketClock:
    def __init__(self, session=None, sequence=0):
        self.session, self.sequence = session or uuid.uuid4().hex, sequence
        self.last_timestamp = None
        self.lock = threading.Lock()

    def clone(self):
        with self.lock:
            clock = TicketClock(self.session, self.sequence)
            clock.last_timestamp = self.last_timestamp
            return clock

    def issue(self, source, mode="auto_number", value=None):
        if mode not in TICKET_MODES:
            raise ValueError("Unknown ticket mode")
        if (
            mode == "external"
            and isinstance(value, dict)
            and value.get("model") == "flow_ticket_v1"
        ):
            ticket = copy.deepcopy(validate_ticket(value))
            ticket["source"] = source
            return ticket
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            if mode == "auto_timestamp":
                instant = datetime.now(timezone.utc)
                if self.last_timestamp is not None and instant <= self.last_timestamp:
                    instant = self.last_timestamp + timedelta(microseconds=1)
                self.last_timestamp = instant
                value = instant.isoformat(timespec="microseconds")
        if mode == "auto_number":
            value = sequence
        if isinstance(value, dict) and set(value) == {"timestamp"}:
            value = datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError("External timestamps require a timezone")
            value = value.astimezone(timezone.utc).isoformat(timespec="microseconds")
            kind = "timestamp"
        else:
            kind = (
                "timestamp"
                if mode == "auto_timestamp"
                else ("number" if type(value) in (int, float) else "string")
            )
        return validate_ticket(
            {
                "model": "flow_ticket_v1",
                "kind": kind,
                "value": value,
                "session": "external" if mode in ("external", "fixed") else self.session,
                "sequence": sequence,
                "source": source,
            }
        )


DIRECT_CLOCK = TicketClock()
