"""Connection outcomes and a bounded, thread-safe ticket event consumer (no Qt)."""

import copy
import json
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from .fail_reasons import merge_reasons, normalize_reasons, reasons_in
from .production import ProductionStatistics
from .tickets import ticket_key, validate_ticket

INPUT_FAILURE_POLICIES = ("continue", "reject_failed")


def validate_outcome(outcome):
    if (
        not isinstance(outcome, dict)
        or outcome.get("model") != "tool_outcome_v1"
        or type(outcome.get("passed")) is not bool
        or type(outcome.get("chain_passed")) is not bool
        or not isinstance(outcome.get("source"), dict)
        or not isinstance(outcome["source"].get("id"), str)
    ):
        raise ValueError("Invalid tool_outcome_v1 connection status")
    json.dumps(outcome, allow_nan=False)
    normalize_reasons(outcome.get("fail_reasons", []))


def packets_in(value):
    from .payloads import FlowGroup, ValuePacket, metadata

    if isinstance(value, ValuePacket):
        if not isinstance(value.value, FlowGroup):
            yield value
        value = value.value
    else:
        if not isinstance(value, FlowGroup):
            yield ValuePacket(value, metadata(value))
    if isinstance(value, FlowGroup):
        for packet in value.items:
            yield from packets_in(packet)


def failed_input(value):
    from .payloads import metadata

    failed = False
    outer = metadata(value).get("outcome")
    if outer is not None:
        validate_outcome(outer)
        failed = not outer["chain_passed"]
    for packet in packets_in(value):
        if "outcome" in packet.payload:
            validate_outcome(packet.payload["outcome"])
            failed |= not packet.payload["outcome"]["chain_passed"]
    return failed


def rejected_inputs(block, values):
    return [
        name
        for name, value in values.items()
        if block.input_failure_policies.get(name, "continue") == "reject_failed"
        and failed_input(value)
    ]


def source_info(block):
    return {
        "id": block.id,
        "name": block.name[:256],
        "type": block.type_id,
        "path": getattr(block, "_event_path", block.id)[:2048],
    }


def stamp_result(block, result):
    """Stamp each emitted packet, including members of groups, without an event log on wires."""
    from .payloads import NO_VALUE, FlowGroup, ImagePyramid, ValuePacket, image_with_payload
    from .timing import advance_timing, mark

    output_finished = mark()

    incoming_failed = any(
        failed_input(ValuePacket(value, block.input_payloads.get(name, {})))
        for name, value in block.last_inputs.items()
    )
    source = source_info(block)
    incoming_reasons = merge_reasons(
        *(
            reasons_in(ValuePacket(value, block.input_payloads.get(name, {})))
            for name, value in block.last_inputs.items()
        )
    )
    if (not result.passed or result.final_judgment is False) and not result.fail_reasons:
        result.fail_reasons = [
            {
                "code": f"{block.type_id}.failed"[:128],
                "message": str(result.message or f"{block.name} failed")[:512],
            }
        ]

    def stamp(packet):
        payload = copy.deepcopy(packet.payload)
        value = packet.value
        if isinstance(value, FlowGroup):
            value = FlowGroup(tuple(stamp(p) for p in value.items), value.labels)
        passed, score = result.passed, result.score
        # Composite calls may contain different tickets. Their per-ticket public
        # emissions carry the actual child's result, not the aggregate node color.
        if getattr(block, "per_packet_outcomes", False) and "outcome" in payload:
            passed = payload["outcome"]["passed"]
            score = payload["outcome"].get("score")
        if getattr(block, "owns_workers", False) and "inspection" in payload:
            if isinstance(value, dict) and "passed" in value:
                passed = value["passed"] is True
                score = value.get("score")
            elif "outcome" in payload:
                passed = payload["outcome"]["passed"]
                score = payload["outcome"].get("score")
        chain_passed = passed and not failed_input(ValuePacket(value, payload))
        if not getattr(block, "outcome_passthrough", False):
            chain_passed = chain_passed and not incoming_failed
        if "ticket" in payload:
            payload["processing_time"] = advance_timing(
                payload.get("processing_time"),
                block._work_started,
                output_finished,
                source=block.ticket_source,
            )
            payload["outcome"] = {
                "model": "tool_outcome_v1",
                "passed": passed,
                "chain_passed": chain_passed,
                "source": source,
                "score": score,
                "message": result.message,
                "data": copy.deepcopy(result.event_data),
                "fail_reasons": merge_reasons(
                    result.fail_reasons if not passed or result.final_judgment is False else [],
                    payload.get("outcome", {}).get("fail_reasons", []),
                    [] if getattr(block, "outcome_passthrough", False) else incoming_reasons,
                ),
            }
        if hasattr(value, "ndim"):
            value = image_with_payload(value, payload)
        elif isinstance(value, ImagePyramid):
            value = ImagePyramid(
                tuple(
                    image_with_payload(
                        level,
                        {
                            **level.metadata,
                            **{k: v for k, v in payload.items() if k not in ("spatial", "pyramid")},
                        },
                    )
                    for level in value.levels
                )
            )
        return ValuePacket(value, payload)

    for port in block.outputs:
        name = port.name
        packets = result.emissions.get(name)
        if packets is None:
            packets = (
                []
                if result.outputs[name] is NO_VALUE
                else [ValuePacket(result.outputs[name], result.payloads.get(name, {}))]
            )
        packets = [stamp(packet) for packet in packets]
        result.emissions[name] = packets
        if packets:
            result.outputs[name], result.payloads[name] = packets[-1].value, packets[-1].payload
        else:
            result.outputs[name] = NO_VALUE
    return result


def publish_result(block, values, result=None, kind="tool_result", message=""):
    from .payloads import metadata
    from .timing import advance_timing, mark

    consumer = getattr(block, "_ticket_results", None)
    if consumer is None:
        return
    if (
        not block.run_profile.ticket_details
        and kind == "tool_result"
        and result
        and result.passed
        and (not block.ticket_source or getattr(block, "_production_excluded", False))
        and not any(
            p.payload.get("outcome", {}).get("chain_passed") is False
            for packets in result.emissions.values()
            for p in packets
        )
    ):
        return
    candidates = []
    if result is not None:
        candidates.extend((p, True) for packets in result.emissions.values() for p in packets)
    primary = (
        getattr(block, "final_ticket_input", None)
        if result and result.final_judgment is not None
        else None
    )
    candidates.extend(
        (v, False) for name, v in values.items() if primary is None or name == primary
    )
    tickets = {}
    for value, stamped in candidates:
        context = metadata(value).get("coordination")
        recipe_context = metadata(value).get("recipe_ref")
        for packet in packets_in(value):
            ticket = packet.payload.get("ticket")
            if ticket is not None:
                tickets.setdefault(ticket_key(ticket), (packet, stamped, context, recipe_context))
    for packet, stamped, context, recipe_context in tickets.values():
        ticket = packet.payload["ticket"]
        outcome = packet.payload.get("outcome", {})
        handled = kind in ("tool_result", "final_judgment", "error")
        end = getattr(block, "_work_finished", None) if handled else None
        end = end or mark()
        start = (getattr(block, "_work_started", None) or end) if handled else end
        timing = (
            packet.payload["processing_time"]
            if stamped
            else advance_timing(
                packet.payload.get("processing_time"),
                start,
                end,
                source=handled
                and block.ticket_source
                and getattr(block, "_source_ticket", None) is not None,
            )
        )
        consumer.consume(
            {
                "ticket": ticket,
                "source": source_info(block),
                "kind": kind,
                "passed": (outcome.get("passed", result.passed) if stamped else result.passed)
                if result
                else None,
                "chain_passed": (
                    outcome.get("chain_passed", True)
                    if stamped
                    else result.passed and not failed_input(packet)
                )
                if result
                else None,
                "score": (outcome.get("score", result.score) if stamped else result.score)
                if result
                else None,
                "message": str(message or (result.message if result else ""))[:2048],
                "data": result.event_data if result else {},
                "final_judgment": result.final_judgment if result else None,
                "parent_ticket": packet.payload.get("inspection", {}).get("parent_ticket"),
                "production_excluded": bool(getattr(block, "_production_excluded", False)),
                "coordination": packet.payload.get("coordination", context),
                "recipe": packet.payload.get("recipe_ref", recipe_context),
                "processing_time": timing,
                "work_ms": block.timing.last_ms if handled and block._work_finished else None,
                "fail_reasons": []
                if result and result.final_judgment is True
                else merge_reasons(
                    result.fail_reasons if result else [],
                    outcome.get("fail_reasons", []),
                    [{"code": "execution.error", "message": str(message)[:512]}]
                    if kind == "error"
                    else [],
                ),
            }
        )


class TicketResults:
    """One shared consumer per root job; identity matches Join/FIFO ticket semantics."""

    def __init__(self, max_tickets=500, events_per_ticket=128, max_bytes=16 * 1024 * 1024):
        if min(max_tickets, events_per_ticket, max_bytes) < 1:
            raise ValueError("Ticket history limits must be positive")
        self.max_tickets, self.events_per_ticket = max_tickets, events_per_ticket
        self.max_bytes = max_bytes
        self.retained_bytes = 0
        self.lock = threading.RLock()
        self.rows = OrderedDict()
        self.sequence = 0
        self.evicted_tickets = 0
        self.production = ProductionStatistics()

    def consume(self, event):
        from .timing import merge_timing, validate_timing

        event = copy.deepcopy(event)
        validate_ticket(event["ticket"])
        event["fail_reasons"] = normalize_reasons(event.get("fail_reasons", []))
        if event.get("processing_time") is not None:
            validate_timing(event["processing_time"])
        encoded = json.dumps(event, allow_nan=False)
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError("Ticket event exceeds 16 KiB; publish a compact summary")
        key = ticket_key(event["ticket"])
        with self.lock:
            self.sequence += 1
            event.update(
                id=uuid.uuid4().hex,
                sequence=self.sequence,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            )
            if key not in self.rows:
                self.rows[key] = {
                    "ticket": event["ticket"],
                    "events": [],
                    "event_count": 0,
                    "discarded_events": 0,
                    "sources": [],
                    "judgments": [],
                    "had_failure": False,
                    "had_error": False,
                    "status": "in_progress",
                    "retained_bytes": 0,
                    "recipes": [],
                }
            row = self.rows[key]
            if event.get("parent_ticket"):
                row["parent_ticket"] = event["parent_ticket"]
            if event.get("production_excluded"):
                row["production_excluded"] = True
            row["fail_reasons"] = merge_reasons(row.get("fail_reasons", []), event["fail_reasons"])
            if event["final_judgment"] is False:
                row["final_fail_reasons"] = merge_reasons(
                    row.get("final_fail_reasons", []), event["fail_reasons"]
                )
            if event.get("processing_time") is not None:
                row["processing_time"] = merge_timing(
                    row.get("processing_time"), event["processing_time"]
                )
            if event["final_judgment"] is not None:
                row.setdefault(
                    "finalized_at",
                    (event.get("processing_time") or {}).get("finished_at", event["timestamp"]),
                )
            recipe = event.get("recipe")
            if recipe and recipe not in row["recipes"] and len(row["recipes"]) < 128:
                row["recipes"].append(copy.deepcopy(recipe))
            self.rows.move_to_end(key)
            row["event_count"] += 1
            row["events"].append(event)
            event_bytes = len(json.dumps(event, allow_nan=False).encode("utf-8"))
            row["retained_bytes"] += event_bytes
            self.retained_bytes += event_bytes
            if len(row["events"]) > self.events_per_ticket:
                removed = row["events"].pop(0)
                removed_bytes = len(json.dumps(removed, allow_nan=False).encode("utf-8"))
                row["retained_bytes"] -= removed_bytes
                self.retained_bytes -= removed_bytes
                row["discarded_events"] += 1
            source = event["ticket"]["source"]
            if source not in row["sources"] and len(row["sources"]) < 128:
                row["sources"].append(source)
            row["had_failure"] |= event["passed"] is False or event["chain_passed"] is False
            row["had_error"] |= event["kind"] == "error"
            judgment = event["final_judgment"]
            if judgment is not None and judgment not in row["judgments"]:
                row["judgments"].append(judgment)
            row["status"] = (
                "conflicting_final"
                if len(row["judgments"]) > 1
                else ("final_pass" if row["judgments"][0] else "final_fail")
                if row["judgments"]
                else "error"
                if row["had_error"]
                else "intermediate_fail"
                if row["had_failure"]
                else "in_progress"
            )
            if (
                judgment is not None
                and not row.get("parent_ticket")
                and not row.get("production_excluded")
            ):
                self.production.record(
                    row["ticket"],
                    bool(judgment),
                    row.get("processing_time", {}).get("elapsed_ms"),
                    final=bool(judgment),
                    conflict=len(row["judgments"]) > 1,
                )
            while len(self.rows) > self.max_tickets or self.retained_bytes > self.max_bytes:
                _, evicted = self.rows.popitem(last=False)
                self.retained_bytes -= evicted["retained_bytes"]
                self.evicted_tickets += 1

    def snapshot(self):
        with self.lock:
            return {
                "tickets": copy.deepcopy(list(self.rows.values())),
                "sequence": self.sequence,
                "evicted_tickets": self.evicted_tickets,
                "production_statistics": copy.deepcopy(self.production.snapshot()),
            }

    def revision(self):
        with self.lock:
            return self.sequence

    def restore(self, snapshot):
        with self.lock:
            self.rows = OrderedDict(
                (ticket_key(r["ticket"]), copy.deepcopy(r)) for r in snapshot["tickets"]
            )
            self.sequence = snapshot["sequence"]
            self.evicted_tickets = snapshot["evicted_tickets"]
            self.retained_bytes = sum(row["retained_bytes"] for row in self.rows.values())
            self.production = ProductionStatistics()
            if "production_statistics" in snapshot:
                self.production.restore(copy.deepcopy(snapshot["production_statistics"]))

    def statistics(self):
        with self.lock:
            return self.production.summary()

    def reset_statistics(self):
        with self.lock:
            self.production = ProductionStatistics()

    def complete(self, ticket, passed, timing=None, summary=False):
        from .timing import advance_timing, mark

        with self.lock:
            row = self.rows.get(ticket_key(ticket))
            if summary and row is None:
                now = mark()
                self.consume(
                    {
                        "ticket": ticket,
                        "source": {
                            "id": "job",
                            "name": "Job completion",
                            "type": "core.job",
                            "path": "",
                        },
                        "kind": "completion",
                        "passed": passed,
                        "chain_passed": passed,
                        "score": None,
                        "message": "Completed ticket",
                        "data": {},
                        "final_judgment": None,
                        "processing_time": advance_timing(timing, now, now),
                    }
                )
                row = self.rows.get(ticket_key(ticket))
            if row and (row.get("parent_ticket") or row.get("production_excluded")):
                return
            previous = row.get("processing_time", timing) if row else timing
            elapsed = advance_timing(previous, mark(), mark())["elapsed_ms"] if previous else None
            if summary and row and previous:
                row["processing_time"] = advance_timing(previous, mark(), mark())
            final = row["judgments"][0] if row and row["judgments"] else None
            accepted = passed and not (row and (row["had_error"] or row["had_failure"]))
            if row is not None:
                row["completion"] = {
                    "passed": bool(
                        (final if final is not None else accepted) and len(row["judgments"]) < 2
                    ),
                    "elapsed_ms": elapsed,
                }
            self.production.record(
                ticket,
                accepted,
                elapsed,
                final=final,
                conflict=bool(row and len(row["judgments"]) > 1),
            )

    def find(self, ticket):
        with self.lock:
            return copy.deepcopy(self.rows.get(ticket_key(ticket)))
