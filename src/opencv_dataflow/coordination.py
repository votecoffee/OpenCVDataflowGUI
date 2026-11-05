"""Explicit, nonblocking multi-source coordination at event ingress."""

import copy
import time
from collections import OrderedDict

from .control_flow import LIMITS, BufferedFlow, byte_size, identity, packet
from .core import Block, Port, Result
from .outcomes import failed_input, source_info
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .processing import finite, integer
from .tickets import DIRECT_CLOCK

INPUTS = tuple(f"input_{i}" for i in range(1, 5))


class CoordinateInputs(BufferedFlow):
    type_id, title = "flow.coordinate", "Coordinate inputs"
    inputs = tuple(Port(name, "any", required=False) for name in INPUTS)
    outputs = (Port("group", "any"), Port("value", "any"), Port("info", "data"))
    flow_primary = ("group",)
    poll_without_input = True
    tolerates_unavailable_inputs = True
    per_packet_outcomes = True
    defaults = {
        "inputs": 2,
        "mode": "ticket_all",
        "max_wait_ms": 1000.0,
        "timeout": "discard",
        "warmup": "all",
        "max_age_ms": 1000.0,
        "primary": "input_1",
        "duplicates": "error",
        "late": "ignore",
        "history_size": 4096,
        **LIMITS,
    }
    parameter_choices = {
        "mode": ("ticket_all", "latest", "trigger_latest", "first_arrival"),
        "timeout": ("discard", "emit_partial", "error"),
        "warmup": ("all", "any"),
        "primary": INPUTS,
        "duplicates": ("error", "replace"),
        "late": ("ignore", "error"),
    }
    parameter_help = {
        "max_wait_ms": "Ticket-all deadline from the first received member. 0 disables expiry.",
        "max_age_ms": "Latest modes: maximum cache age measured at this block; 0 disables expiry.",
        "warmup": "Latest modes require all fresh inputs, or explicitly allow partial snapshots.",
        "primary": "Preferred value output; trigger_latest emits only on this input.",
        "history_size": "Closed ticket keys remembered for late suppression; oldest keys age out.",
    }

    def reset_runtime(self):
        self.runtime = {
            "pending": OrderedDict(),
            "latest": {},
            "closed": OrderedDict(),
            "received": 0,
            "emitted": 0,
            "timeouts": 0,
            "late": 0,
            "replaced": 0,
            "not_ready": 0,
        }

    def now(self):
        return time.monotonic()

    def settings(self):
        p = self.parameters
        count = integer(p["inputs"], "Input count", 2, 4)
        names = INPUTS[:count]
        if p["primary"] not in names:
            raise ValueError("Primary input must be within the configured input count")
        return (
            names,
            finite(p["max_wait_ms"], "Wait time", 0) / 1000,
            finite(p["max_age_ms"], "Maximum age", 0) / 1000,
            integer(p["history_size"], "History size", 1, 4096),
        )

    def bound(self):
        p = self.parameters
        maximum = integer(p["max_pending"], "Maximum pending", 1, 4096)
        mb = finite(p["max_bytes_mb"], "Maximum buffer MB", 0)
        if not 0 < mb <= 4096:
            raise ValueError("Buffer size must be in (0,4096] MB")
        size = byte_size(self.runtime["pending"]) + byte_size(self.runtime["latest"])
        if len(self.runtime["pending"]) > maximum or size > mb * 1024 * 1024:
            raise ValueError("Coordination buffer limit exceeded")

    def close_key(self, key, history):
        self.runtime["closed"][key] = True
        while len(self.runtime["closed"]) > history:
            self.runtime["closed"].popitem(last=False)

    def report(self, row, kind, message, passed=None, data=None):
        consumer = self._ticket_results
        if consumer is None:
            return
        seen = set()
        for value, _ in row.values():
            key = identity(value)
            if key in seen:
                continue
            seen.add(key)
            consumer.consume(
                {
                    "ticket": value.payload["ticket"],
                    "source": source_info(self),
                    "kind": kind,
                    "passed": passed,
                    "chain_passed": passed,
                    "score": None,
                    "message": message,
                    "data": data or {},
                    "final_judgment": None,
                    "parent_ticket": None,
                }
            )

    def snapshot(self, row, names, now, reason, trigger=None):
        selected = [name for name in names if name in row]
        entries = tuple(row[name][0] for name in selected)
        group = FlowGroup(entries, tuple(selected))
        primary = self.parameters["primary"]
        chosen = primary if primary in row else (trigger if trigger in row else selected[0])
        value = row[chosen][0]
        mixed = len({identity(entry) for entry in entries}) > 1
        if mixed:
            ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(self.id)
        else:
            ticket = copy.deepcopy(entries[0].payload["ticket"])
        info = {
            "model": "input_coordination_v1",
            "mode": self.parameters["mode"],
            "reason": reason,
            "complete": len(selected) == len(names),
            "present": selected,
            "missing": [n for n in names if n not in row],
            "trigger": trigger,
            "primary": chosen,
            "mixed_tickets": mixed,
            "tickets": {n: row[n][0].payload["ticket"] for n in selected},
            "age_ms": {n: max(0.0, (now - row[n][1]) * 1000) for n in selected},
        }
        # Incomplete timeout/warmup snapshots are explicitly failed. A first-arrival
        # race intentionally needs only one member and is not a missing-input failure.
        passed = info["complete"] or reason == "first_arrival"

        def payload(base, item):
            result = copy.deepcopy(base)
            result["coordination"] = copy.deepcopy(info)
            result["outcome"] = {
                "model": "tool_outcome_v1",
                "passed": passed,
                "chain_passed": passed and not failed_input(item),
                "source": source_info(self),
                "score": None,
                "message": reason,
                "data": {},
            }
            return result

        group_payload = payload({"ticket": ticket}, group)
        value_payload = payload(value.payload, value)
        self.runtime["emitted"] += 1
        return {
            "group": ValuePacket(group, group_payload),
            "value": ValuePacket(value.value, value_payload),
            "info": ValuePacket(info, copy.deepcopy(group_payload)),
        }, passed

    def process(self, inputs):
        names, wait, age, history = self.settings()
        now, p, state = self.now(), self.parameters, self.runtime
        produced = []
        if p["mode"] == "ticket_all" and wait:
            for key, pending in list(state["pending"].items()):
                if now - pending["first"] < wait:
                    continue
                row = state["pending"].pop(key)["values"]
                self.close_key(key, history)
                state["timeouts"] += 1
                missing = [n for n in names if n not in row]
                self.report(
                    row,
                    "error" if p["timeout"] == "error" else "coordination_timeout",
                    "Inputs expired before completion",
                    False,
                    {"missing": missing, "timeout": p["timeout"]},
                )
                if p["timeout"] == "error":
                    raise ValueError(f"Coordination timeout; missing {', '.join(missing)}")
                if p["timeout"] == "emit_partial":
                    produced.append(self.snapshot(row, names, now, "timeout"))
        if p["mode"] in ("latest", "trigger_latest") and age:
            state["latest"] = {
                n: item for n, item in state["latest"].items() if now - item[1] <= age
            }
        for name in inputs:
            if name not in names:
                raise ValueError("Arrival is outside the configured input count")
            value = packet(self, name, inputs)
            key = identity(value)
            state["received"] += 1
            if p["mode"] in ("ticket_all", "first_arrival"):
                if key in state["closed"]:
                    state["late"] += 1
                    self.report({name: (value, now)}, "coordination_late", "Ignored closed ticket")
                    if p["late"] == "error":
                        raise ValueError("Late arrival for a closed coordination ticket")
                    continue
                if p["mode"] == "first_arrival":
                    self.close_key(key, history)
                    produced.append(
                        self.snapshot({name: (value, now)}, names, now, "first_arrival", name)
                    )
                    continue
                pending = state["pending"].setdefault(key, {"first": now, "values": {}})
                row = pending["values"]
                if name in row:
                    if p["duplicates"] == "error":
                        raise ValueError("Duplicate input for a pending coordination ticket")
                    state["replaced"] += 1
                row[name] = (value, now)
                self.bound()
                if len(row) == len(names):
                    del state["pending"][key]
                    self.close_key(key, history)
                    produced.append(self.snapshot(row, names, now, "complete", name))
            else:
                state["latest"][name] = (value, now)
                self.bound()
                if p["mode"] == "trigger_latest" and name != p["primary"]:
                    continue
                row = state["latest"]
                if len(row) < len(names) and p["warmup"] == "all":
                    state["not_ready"] += 1
                    if p["mode"] == "trigger_latest":
                        self.report(
                            {name: (value, now)},
                            "coordination_not_ready",
                            "Trigger arrived without all fresh inputs",
                            False,
                        )
                    continue
                if len(row) < len(names):
                    self.report(
                        row,
                        "coordination_partial",
                        "Emitted an incomplete latest snapshot",
                        False,
                        {"missing": [n for n in names if n not in row], "mode": p["mode"]},
                    )
                produced.append(self.snapshot(row, names, now, "latest", name))
        self.bound()
        emissions = {
            name: [row[name] for row, _ in produced] for name in ("group", "value", "info")
        }
        return Result(
            {
                name: packets[-1].value if packets else NO_VALUE
                for name, packets in emissions.items()
            },
            emissions=emissions,
            passed=all(passed for _, passed in produced),
            event_data={"mode": p["mode"], "waiting": len(state["pending"])},
            message=f"{len(state['pending'])} waiting · {len(state['latest'])} cached · "
            f"{state['emitted']} emitted · {state['timeouts']} expired · {state['late']} late",
        )


class CounterInput(Block):
    """Deterministic stand-in for an external PLC part counter, not a PLC driver."""

    type_id, title, category = "input.counter", "Counter input", "Input"
    block_type, subtype = "Input", "Synthetic"
    outputs = (Port("value", "data"),)
    defaults = {"start": 1, "step": 1}

    def reset_runtime(self):
        self.next_value = None

    def copy_runtime_from(self, other):
        self.next_value = getattr(other, "next_value", None)

    def process(self, inputs):
        start, step = (self.parameters[k] for k in ("start", "step"))
        if type(start) is not int or type(step) is not int or step == 0:
            raise ValueError("Counter start and nonzero step must be whole numbers")
        value = getattr(self, "next_value", None)
        value = int(start) if value is None else value
        self.next_value = value + int(step)
        return Result({"value": value})
