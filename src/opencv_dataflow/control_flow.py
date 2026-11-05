"""Bounded ticket-aware flow operators. Buffers are runtime state, never project data."""

import copy
import json

import numpy as np

from .core import Block, Port, Result
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .pointclouds import PointCloud
from .processing import finite, integer
from .tickets import DIRECT_CLOCK, TICKET_MODES, ticket_key

BRANCHES = tuple(f"branch_{i}" for i in range(1, 5))
LIMITS = {"max_pending": 64, "max_bytes_mb": 128.0}


def packet(block, name, inputs):
    return ValuePacket(inputs[name], copy.deepcopy(block.input_payloads.get(name, {})))


def identity(value):
    if "ticket" not in value.payload:
        raise ValueError("Input has no ticket; connect an input source or Assign ticket first")
    return ticket_key(value.payload["ticket"])


def byte_size(value):
    if isinstance(value, PointCloud):
        return value.nbytes + len(json.dumps(value.metadata).encode())
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, ValuePacket):
        return byte_size(value.value) + len(json.dumps(value.payload).encode())
    if isinstance(value, FlowGroup):
        return sum(byte_size(p) for p in value.items)
    if isinstance(value, dict):
        return sum(byte_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(byte_size(v) for v in value)
    return len(str(value).encode())


class FlowBlock(Block):
    outcome_passthrough = True
    category, block_type, subtype = "Control flow", "Control flow", "Routing"
    preserve_image_sets = True


class BufferedFlow(FlowBlock):
    accept_events = True
    subtype = "Synchronization"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self.runtime = {"pending": {}, "order": [], "seen": []}

    def copy_runtime_from(self, other):
        self.runtime = copy.deepcopy(other.runtime)

    def close(self):
        self.reset_runtime()

    def bounded(self, count):
        maximum = integer(self.parameters["max_pending"], "Maximum pending", 1, 4096)
        megabytes = finite(self.parameters["max_bytes_mb"], "Maximum buffer MB", 0)
        if megabytes <= 0 or megabytes > 4096:
            raise ValueError("Buffer size must be in (0,4096] MB")
        if count > maximum or byte_size(self.runtime["pending"]) > megabytes * 1024 * 1024:
            raise ValueError("Flow buffer limit exceeded; reset state or resolve missing tickets")

    def remember(self, key):
        self.runtime["seen"].append(key)
        self.runtime["seen"] = self.runtime["seen"][-4096:]


class AssignTicket(FlowBlock):
    type_id, title = "flow.ticket", "Assign ticket"
    requires_serial_execution = True
    inputs = (Port("value", "any"), Port("ticket", "data", required=False))
    outputs = (Port("value", "any"), Port("ticket", "data"))
    defaults = {"mode": "auto_number", "value": 1}
    parameter_choices = {"mode": TICKET_MODES}
    parameter_help = {"value": "Fixed number/string, or {timestamp: ISO8601 with timezone}."}

    def process(self, inputs):
        mode = self.parameters["mode"]
        if mode == "external" and "ticket" not in inputs:
            raise ValueError("External mode requires the ticket input")
        ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(
            self.input_payloads["value"].get("ticket", {}).get("source", self.id),
            mode,
            inputs.get("ticket", self.parameters["value"]),
        )
        context = {**self.input_payloads["value"], "ticket": ticket}
        return Result(
            {"value": inputs["value"], "ticket": ticket},
            payloads={"value": context, "ticket": {"ticket": ticket}},
        )


class Split(FlowBlock):
    type_id, title = "flow.split", "Split parallel branches"
    inputs = (Port("value", "any"),)
    outputs = tuple(Port(name, "any") for name in BRANCHES)
    defaults = {"branches": 2, "workers": 2}
    parameter_help = {
        "workers": "1..8 workers for independent nodes when the graph's Branch workers is Auto."
    }

    def process(self, inputs):
        count = integer(self.parameters["branches"], "Branches", 2, 4)
        integer(self.parameters["workers"], "Workers", 1, 8)
        value = packet(self, "value", inputs)
        identity(value)
        return Result(
            {
                name: value.value if index < count else NO_VALUE
                for index, name in enumerate(BRANCHES)
            },
            payloads={name: value.payload for name in BRANCHES[:count]},
        )


class Join(BufferedFlow):
    type_id, title = "flow.join", "Join parallel branches"
    inputs = tuple(Port(name, "any", required=False) for name in BRANCHES)
    outputs = (
        Port("group", "any"),
        *tuple(Port(n, "any") for n in BRANCHES),
        Port("queued", "number"),
    )
    defaults = {"branches": 2, **LIMITS}
    flow_primary = ("group",)

    def process(self, inputs):
        count = integer(self.parameters["branches"], "Branches", 2, 4)
        pending = self.runtime["pending"]
        for name in inputs:
            if name not in BRANCHES[:count]:
                raise ValueError("Join received an input outside its configured branch count")
            value = packet(self, name, inputs)
            key = identity(value)
            row = pending.setdefault(key, {})
            if key in self.runtime["seen"] or name in row:
                raise ValueError("Duplicate ticket on a join branch")
            row[name] = value
        self.bounded(len(pending))
        ready = next((k for k, row in pending.items() if len(row) == count), None)
        outputs = {"group": NO_VALUE, **dict.fromkeys(BRANCHES, NO_VALUE), "queued": len(pending)}
        contexts = {}
        if ready is not None:
            row = pending.pop(ready)
            entries = tuple(row[n] for n in BRANCHES[:count])
            from .recipes import unique_ref

            unique_ref(entries)
            outputs["group"] = FlowGroup(entries, BRANCHES[:count])
            contexts["group"] = {"ticket": entries[0].payload["ticket"]}
            for name, value in zip(BRANCHES, entries):
                outputs[name], contexts[name] = value.value, value.payload
            self.remember(ready)
        outputs["queued"] = len(pending)
        return Result(outputs, payloads=contexts, message=f"{len(pending)} unmatched tickets")


class Batch(BufferedFlow):
    type_id, title, subtype = "flow.batch", "Batch serial entries", "Groups"
    inputs = tuple(Port(f"item_{i}", "any", required=False) for i in range(1, 5))
    outputs = (Port("group", "any"), Port("queued", "number"))
    defaults = {"size": 3, "distinct_sources": False, **LIMITS}
    flow_primary = ("group",)

    def reset_runtime(self):
        super().reset_runtime()
        self.runtime["pending"] = []

    def process(self, inputs):
        size = integer(self.parameters["size"], "Group size", 1, 64)
        queue = self.runtime["pending"]
        for name in inputs:
            value = packet(self, name, inputs)
            identity(value)
            queue.append(value)
        self.bounded(len(queue))
        group, context = NO_VALUE, {}
        if len(queue) >= size:
            entries = tuple(queue[:size])
            if (
                self.parameters["distinct_sources"]
                and len({p.payload["ticket"]["source"] for p in entries}) != size
            ):
                raise ValueError("Batch requires distinct ticket sources")
            del queue[:size]
            group = FlowGroup(entries, tuple(p.payload["ticket"]["source"] for p in entries))
            ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(self.id)
            context = {"group": {"ticket": ticket}}
        return Result(
            {"group": group, "queued": len(queue)},
            payloads=context,
            message=f"{len(queue)}/{size} entries buffered",
        )


class Unbatch(FlowBlock):
    type_id, title, subtype = "flow.unbatch", "Unbatch serial entries", "Groups"
    inputs = (Port("group", "any"),)
    outputs = (Port("value", "any"), Port("count", "number"))
    defaults = {"order": "stored"}
    parameter_choices = {"order": ("stored", "sequence_ascending", "sequence_descending")}
    flow_primary = ("value",)

    def process(self, inputs):
        group = inputs["group"]
        if not isinstance(group, FlowGroup):
            raise ValueError("Unbatch requires a flow group")
        entries = list(group.items)
        if self.parameters["order"] != "stored":
            for entry in entries:
                identity(entry)
            entries.sort(
                key=lambda p: p.payload["ticket"]["sequence"],
                reverse=self.parameters["order"] == "sequence_descending",
            )
        return Result(
            {"value": entries[-1].value, "count": len(entries)},
            payloads={"value": entries[-1].payload},
            emissions={"value": entries},
        )


class SelectEntry(FlowBlock):
    type_id, title, subtype = "flow.select_entry", "Select group entry", "Groups"
    inputs = (Port("group", "any"),)
    outputs = (Port("value", "any"),)
    defaults = {"index": 0}

    def process(self, inputs):
        group = inputs["group"]
        index = integer(self.parameters["index"], "Entry index", 0, 63)
        if not isinstance(group, FlowGroup) or index >= len(group.items):
            raise ValueError("Entry index is outside this flow group")
        value = group.items[index]
        return Result({"value": value.value}, payloads={"value": value.payload})


class CaseSelect(FlowBlock):
    type_id, title = "flow.case", "Case select"
    inputs = (Port("value", "any"), Port("selector", "data", required=False))
    outputs = (*tuple(Port(f"case_{i}", "any") for i in range(1, 5)), Port("default", "any"))
    defaults = {"path": "ticket.value", "cases": [0, 1, 2, 3], "comparison": "equals", "modulo": 0}
    parameter_choices = {"comparison": ("equals", "less_than", "greater_than", "range")}
    parameter_help = {
        "path": "Dotted path under ticket or value, unless selector is connected.",
        "cases": "1..4 JSON values; range uses [minimum,maximum] pairs. First match wins.",
    }

    def process(self, inputs):
        value = packet(self, "value", inputs)
        selector = inputs.get("selector", NO_VALUE)
        if selector is NO_VALUE:
            selector = {"ticket": value.payload.get("ticket", {}), "value": value.value}
            for name in self.parameters["path"].split("."):
                if not isinstance(selector, dict) or name not in selector:
                    raise ValueError("Case selector path is missing")
                selector = selector[name]
        if isinstance(selector, dict) and selector.get("model") == "flow_ticket_v1":
            selector = selector["value"]
        modulo = integer(self.parameters["modulo"], "Modulo", 0, 1000000)
        if modulo:
            selector = finite(selector, "Numeric selector") % modulo
        cases = self.parameters["cases"]
        if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
            raise ValueError("Case select requires 1..4 cases")
        chosen = "default"
        for index, case in enumerate(cases, 1):
            mode = self.parameters["comparison"]
            if mode == "equals":
                match = selector == case and (isinstance(selector, bool) == isinstance(case, bool))
            elif mode == "range":
                if not isinstance(case, list) or len(case) != 2:
                    raise ValueError("Range cases need [minimum,maximum]")
                low, high = (finite(v, "Case range") for v in case)
                if low > high:
                    raise ValueError("Case range minimum exceeds maximum")
                match = low <= finite(selector, "Selector") <= high
            else:
                a, b = finite(selector, "Selector"), finite(case, "Case")
                match = a < b if mode == "less_than" else a > b
            if match:
                chosen = f"case_{index}"
                break
        return Result(
            {p.name: value.value if p.name == chosen else NO_VALUE for p in self.outputs},
            payloads={chosen: value.payload},
            message=f"Selected {chosen}",
        )


class Reorder(BufferedFlow):
    type_id, title = "flow.reorder", "FIFO reorder by ticket"
    inputs = (
        Port("expected", "any", required=False),
        *tuple(Port(f"value_{i}", "any", required=False) for i in range(1, 5)),
    )
    outputs = (Port("value", "any"), Port("queued", "number"))
    defaults = {"mode": "manifest", "start": 1, "step": 1, **LIMITS}
    parameter_choices = {"mode": ("manifest", "number")}
    flow_primary = ("value",)
    parameter_help = {
        "expected": "Wire the pre-branch stream to expected to order arbitrary tickets.",
        "start": "First numeric ticket expected; number mode only.",
    }

    def process(self, inputs):
        pending, order = self.runtime["pending"], self.runtime["order"]
        numeric = self.parameters["mode"] == "number"
        start = integer(self.parameters["start"], "First ticket", -(2**53), 2**53)
        step = integer(self.parameters["step"], "Ticket step", 1, 1000000)
        self.runtime.setdefault("next", start)
        for name in inputs:
            value = packet(self, name, inputs)
            key = identity(value)
            if not numeric:
                # Cameras may share a PLC part ID; FIFO orders individual acquisitions.
                key = (*key, value.payload["ticket"]["source"])
            if numeric:
                if name == "expected":
                    raise ValueError("Expected input is used only in manifest mode")
                ticket = value.payload["ticket"]
                if ticket["kind"] != "number" or type(ticket["value"]) is not int:
                    raise ValueError(
                        "Number reorder requires integer tickets; use manifest for text/time"
                    )
                key = ticket["value"]
                if key < self.runtime["next"] or (key - start) % step:
                    raise ValueError("Late or off-sequence FIFO ticket")
            if name == "expected":
                if key in order or key in self.runtime["seen"]:
                    raise ValueError("Duplicate FIFO expectation")
                order.append(key)
            else:
                if key in pending or key in self.runtime["seen"]:
                    raise ValueError("Duplicate FIFO result")
                pending[key] = value
        self.bounded(len(order) + len(pending))
        entries = []
        while True:
            key = self.runtime["next"] if numeric else (order[0] if order else None)
            if key is None or key not in pending:
                break
            entries.append(pending.pop(key))
            self.remember(key)
            if numeric:
                self.runtime["next"] += step
            else:
                order.pop(0)
        return Result(
            {
                "value": entries[-1].value if entries else NO_VALUE,
                "queued": max(len(order), len(pending)),
            },
            payloads={"value": entries[-1].payload} if entries else {},
            emissions={"value": entries},
            message=f"{len(pending)} results, {len(order)} expectations buffered",
        )


CONTROL_BLOCKS = (AssignTicket, Split, Join, Batch, Unbatch, SelectEntry, CaseSelect, Reorder)
