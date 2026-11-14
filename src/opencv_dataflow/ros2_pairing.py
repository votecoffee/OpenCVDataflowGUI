"""Bounded ordered timestamp pairing; temporal association is not hardware sync."""

import copy
import time
from collections import deque

from .control_flow import LIMITS, BufferedFlow, byte_size, packet
from .core import Port, Result
from .outcomes import failed_input, source_info
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .recipes import unique_ref
from .ros2_context import acquisition_header, check_current, scope_key, validate_context
from .ros2_poses import frame_name
from .tickets import DIRECT_CLOCK, validate_ticket
from .timing import merge_timing
from .visual_style import INDUSTRIAL

NAMES = ("input_1", "input_2")


class PairROSImages(BufferedFlow):
    type_id, title = "industrial.ros2_image_pair", "Pair ROS 2 images by timestamp"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (
        Port("context", "data", required=False),
        Port("reset", "boolean", required=False),
        *(Port(name, required=False) for name in NAMES),
    )
    outputs = (Port("group", "any"), *(Port(n) for n in NAMES), Port("info", "data"))
    flow_primary = ("group",)
    poll_without_input = True
    tolerates_unavailable_inputs = True
    requires_serial_execution = True
    per_packet_outcomes = True
    preview_safe = False
    payload_reference = ""
    defaults = {
        "tolerance_ms": 5.0,
        "max_wait_ms": 1000.0,
        "clock_domain": "ros_time",
        "require_same_frame": True,
        "require_context": False,
        **LIMITS,
    }
    parameter_ranges = {
        "tolerance_ms": NumericRange(0, 1000),
        "max_wait_ms": NumericRange(0, 3600000),
        "max_pending": NumericRange(1, 4096, whole=True),
        "max_bytes_mb": NumericRange(0.001, 4096),
    }
    parameter_help = {
        "tolerance_ms": "Inclusive maximum header-stamp spread; zero requires exact time.",
        "max_wait_ms": (
            "Monotonic receipt deadline per image; zero disables expiry. Polled on graph runs."
        ),
        "clock_domain": (
            "Declared common clock label, not verified synchronization or clock conversion."
        ),
        "require_same_frame": (
            "Require the same optical frame. Turning off does not register images."
        ),
        "require_context": (
            "Require current Run identity and tagged images; a context change clears queues."
        ),
        "max_pending": (
            "Maximum total queued images across both streams; overflow errors and clears queues."
        ),
        "max_bytes_mb": "Bound accounted packet bytes; not a hard process-heap limit.",
    }
    input_help = {
        "input_1": "Ticketed ROS image with a positive integer header stamp, in increasing order.",
        "input_2": "Second ordered image stream on the declared common clock.",
        "context": "Prime shared Run identity before tagged images; changes invalidate queues.",
        "reset": "Rising true clears queued images and a backward-time hold once.",
    }

    def reset_runtime(self):
        self.runtime = {
            "queues": {name: deque() for name in NAMES},
            "last": {name: None for name in NAMES},
            "context": None,
            "reset": False,
            "held": False,
            "needs_context": False,
            "bytes": 0,
            "matched": 0,
            "dropped": 0,
            "duplicates": 0,
            "settings": None,
        }

    def now(self):
        return time.monotonic()

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in self.parameter_ranges:
            if type(p[name]) not in (int, float):
                raise ValidationError("Use numeric queue/timing settings", name)
        for name in ("require_context", "require_same_frame"):
            if type(p[name]) is not bool:
                raise ValidationError("Use a boolean setting", name)
        if (
            not isinstance(p["clock_domain"], str)
            or not 1 <= len(p["clock_domain"]) <= 128
            or not p["clock_domain"].strip()
            or "\0" in p["clock_domain"]
        ):
            raise ValidationError(
                "Name the declared common clock (1..128 characters)", "clock_domain"
            )

    def notify(self, name, entry, kind, passed=False):
        value = entry["packet"]
        data = {"input": name, "stamp_ns": entry["ns"], "reason": kind}
        self.report_event(kind, data=data, values={name: value})
        if self._ticket_results is not None:
            self._ticket_results.consume(
                {
                    "ticket": value.payload["ticket"],
                    "source": source_info(self),
                    "kind": kind,
                    "passed": passed,
                    "chain_passed": passed,
                    "score": None,
                    "message": kind,
                    "data": data,
                    "final_judgment": None,
                    "parent_ticket": None,
                }
            )

    def pop(self, name, reason=None):
        entry = self.runtime["queues"][name].popleft()
        self.runtime["bytes"] -= entry["size"]
        if reason is not None:
            self.runtime["dropped"] += 1
            self.notify(name, entry, reason)
        return entry

    def make_pair(self, entries, now):
        state = self.runtime
        context = state["context"]
        values = [entry["packet"] for entry in entries]
        reference = unique_ref(values)
        if (
            self.parameters["require_same_frame"]
            and entries[0]["header"]["frame_id"] != entries[1]["header"]["frame_id"]
        ):
            raise ValidationError(
                "Paired optical frames differ; register images or explicitly allow different frames"
            )
        ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(self.id)
        if context is not None:
            ticket["session"] = scope_key(context)
        info = {
            "model": "ros_time_pair_v1",
            "policy": "ordered_fifo",
            "ticket": copy.deepcopy(ticket),
            "spread_ns": abs(entries[1]["ns"] - entries[0]["ns"]),
            "tolerance_ns": round(self.parameters["tolerance_ms"] * 1_000_000),
            "clock_domain": self.parameters["clock_domain"],
            "clock_domain_verified": False,
            "context": copy.deepcopy(context),
            "physical_sync_verified": False,
            "members": {
                name: {
                    "header": copy.deepcopy(entry["header"]),
                    "ticket": copy.deepcopy(entry["packet"].payload["ticket"]),
                    "acquisition": copy.deepcopy(entry["packet"].payload.get("acquisition")),
                    "receipt_age_ms": max(0, now - entry["received"]) * 1000,
                }
                for name, entry in zip(NAMES, entries)
            },
        }

        def payload(original):
            result = copy.deepcopy(original)
            result.update(ticket=copy.deepcopy(ticket), ros_time_pair=copy.deepcopy(info))
            if reference is not None:
                result["recipe_ref"] = copy.deepcopy(reference)
            if context is not None:
                result["acquisition"] = {
                    "model": "ros_acquisition_v1",
                    "context": copy.deepcopy(context),
                    "capture": {"kind": ticket["kind"], "value": ticket["value"]},
                }
            return result

        packets = [ValuePacket(v.value, payload(v.payload)) for v in values]
        group = FlowGroup(tuple(packets), NAMES)
        group_payload = payload({})
        for value in values:
            timing = value.payload.get("processing_time")
            if timing is not None:
                group_payload["processing_time"] = merge_timing(
                    group_payload.get("processing_time"), timing
                )
        passed = not failed_input(group)
        group_payload["outcome"] = {
            "model": "tool_outcome_v1",
            "passed": passed,
            "chain_passed": passed,
            "source": source_info(self),
            "score": None,
            "message": "Temporal association",
            "data": {},
        }
        state["matched"] += 1
        return {
            "group": ValuePacket(group, group_payload),
            "input_1": packets[0],
            "input_2": packets[1],
            "info": ValuePacket(info, copy.deepcopy(group_payload)),
        }

    def process(self, inputs):
        state, p, now = self.runtime, self.parameters, self.now()
        if state["settings"] is not None and state["settings"] != p:
            context, resetting = state["context"], state["reset"]
            self.reset_runtime()
            state = self.runtime
            state.update(context=context, reset=resetting)
        if "context" in inputs:
            context = validate_context(inputs["context"])
            if context != state["context"]:
                resetting = state["reset"]
                self.reset_runtime()
                state = self.runtime
                state.update(context=context, reset=resetting)
        if "reset" in inputs:
            if inputs["reset"] and not state["reset"]:
                context = state["context"]
                self.reset_runtime()
                state = self.runtime
                state["context"] = context
            state["reset"] = inputs["reset"]
        state["settings"] = copy.deepcopy(p)
        wait = p["max_wait_ms"] / 1000
        if wait:
            for name in NAMES:
                queue = state["queues"][name]
                while queue and now - queue[0]["received"] >= wait:
                    self.pop(name, "time_pair_timeout")
        for name in NAMES:
            if name not in inputs:
                continue
            value = packet(self, name, inputs)
            validate_ticket(value.payload.get("ticket"))
            if state["context"] is None and (
                p["require_context"] or "acquisition" in value.payload
            ):
                state["needs_context"] = True
                self.report_event("time_pair_no_context", values={name: value})
                continue  # Let a later Context event prime the block; never queue this image.
            if "acquisition" in value.payload or state["context"] is not None:
                check_current(value.value, value.payload, state["context"])
            state["needs_context"] = False
            if "ros2_static_binding" in value.payload:
                raise ValidationError("Pair image streams before binding static CameraInfo")
            header = acquisition_header(value.value, value.payload)
            frame_name(header["frame_id"])
            ns = header["stamp"]["sec"] * 1_000_000_000 + header["stamp"]["nanosec"]
            if ns <= 0:
                raise ValidationError(
                    "Pairing requires positive acquisition stamps; zero/receipt time is forbidden"
                )
            entry = {
                "packet": value,
                "header": header,
                "ns": ns,
                "received": now,
                "size": byte_size(value),
            }
            previous = state["last"][name]
            if state["held"]:
                self.notify(name, entry, "time_pair_reset_required")
                continue
            if previous is not None and ns < previous:
                for stream in NAMES:
                    while state["queues"][stream]:
                        self.pop(stream, "time_pair_clock_reset")
                state["held"] = True
                self.notify(name, entry, "time_pair_clock_reset")
                self.logger.warning(
                    "Image time moved backward; reset flow or supply a new run context"
                )
                continue
            if ns == previous:
                state["duplicates"] += 1
                self.notify(name, entry, "time_pair_duplicate", None)
                continue
            state["last"][name] = ns
            state["queues"][name].append(entry)
            state["bytes"] += entry["size"]
            if (
                sum(len(q) for q in state["queues"].values()) > p["max_pending"]
                or state["bytes"] > p["max_bytes_mb"] * 1024 * 1024
            ):
                raise ValidationError("Timestamp pairing buffer limit exceeded; queues cleared")
        produced = []
        first, second = (state["queues"][n] for n in NAMES)
        tolerance = round(p["tolerance_ms"] * 1_000_000)
        while first and second and not state["held"]:
            delta = second[0]["ns"] - first[0]["ns"]
            if abs(delta) <= tolerance:
                produced.append(self.make_pair([self.pop(n) for n in NAMES], now))
            else:
                self.pop("input_1" if delta > 0 else "input_2", "time_pair_outside_window")
        emissions = {
            name: [row[name] for row in produced] for name in (p.name for p in self.outputs)
        }
        return Result(
            {name: values[-1].value if values else NO_VALUE for name, values in emissions.items()},
            payloads={name: values[-1].payload for name, values in emissions.items() if values},
            emissions=emissions,
            passed=not state["held"],
            event_data={
                "policy": "ordered_fifo",
                "queued": sum(len(q) for q in state["queues"].values()),
                "matched": state["matched"],
                "dropped": state["dropped"],
                "reset_required": state["held"],
            },
            message=(
                "Reset required · "
                if state["held"]
                else "Waiting for current Run identity · "
                if state["needs_context"]
                else ""
            )
            + f"{sum(len(q) for q in state['queues'].values())} queued · "
            f"{state['matched']} paired · {state['dropped']} dropped · "
            f"{state['duplicates']} duplicate",
        )
