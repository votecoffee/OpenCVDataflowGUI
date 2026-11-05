"""Runtime invocation statistics and portable UTC/local-monotonic ticket timing."""

import copy
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

CLOCK_ID = uuid.uuid4().hex


def mark():
    return {
        "ns": time.perf_counter_ns(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "clock": CLOCK_ID,
    }


@dataclass
class WorkTiming:
    count: int = 0
    last_ms: float | None = None
    average_ms: float | None = None
    started: dict | None = None

    def begin(self):
        self.started = mark()
        return self.started

    def finish(self):
        end = mark()
        if self.started is not None:
            self.last_ms = max(0, end["ns"] - self.started["ns"]) / 1_000_000
            self.count += 1
            self.average_ms = (self.average_ms or 0) + (
                self.last_ms - (self.average_ms or 0)
            ) / self.count
            self.started = None
        return end

    def snapshot(self):
        started = self.started
        return {
            "running": started is not None,
            "current_ms": max(0, time.perf_counter_ns() - started["ns"]) / 1_000_000
            if started
            else None,
            "last_ms": self.last_ms,
            "average_ms": self.average_ms,
            "count": self.count,
        }

    def absorb(self, other):
        if other.count:
            count = self.count + other.count
            self.average_ms = (
                (self.average_ms or 0) * self.count + other.average_ms * other.count
            ) / count
            self.count = count
            self.last_ms = other.last_ms


def validate_timing(value):
    if not isinstance(value, dict) or value.get("model") != "processing_time_v1":
        raise ValueError("Expected processing_time_v1 timing")
    for key in ("started_at", "finished_at"):
        stamp = datetime.fromisoformat(value[key].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Processing timestamps require a timezone")
    for key in ("started_ns", "finished_ns"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("Processing monotonic timestamps must be nonnegative integers")
    for key in ("clock_id", "finished_clock_id"):
        if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 128:
            raise ValueError("Processing timing requires a clock identity")
    elapsed = value["elapsed_ms"]
    if elapsed is not None and (
        type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise ValueError("Processing duration must be finite and nonnegative")
    if value["clock_id"] != value["finished_clock_id"]:
        if elapsed is not None:
            raise ValueError("Cannot compare monotonic timestamps from different processes")
    elif (
        value["finished_ns"] < value["started_ns"]
        or elapsed != (value["finished_ns"] - value["started_ns"]) / 1_000_000
    ):
        raise ValueError("Processing duration does not match its monotonic timestamps")


def advance_timing(previous, start, finish, *, source=False):
    if previous is not None and not source:
        validate_timing(previous)
        result = copy.deepcopy(previous)
    else:
        result = {
            "model": "processing_time_v1",
            "started_at": start["utc"],
            "started_ns": start["ns"],
            "clock_id": start["clock"],
            "inferred_start": not source,
        }
    result.update(
        finished_at=finish["utc"],
        finished_ns=finish["ns"],
        finished_clock_id=finish["clock"],
        elapsed_ms=max(0, finish["ns"] - result["started_ns"]) / 1_000_000
        if finish["clock"] == result["clock_id"]
        else None,
    )
    return result


def merge_timing(previous, incoming):
    """Events can arrive out of order from parallel workers; never shorten local spans."""
    if previous is None:
        return copy.deepcopy(incoming)
    if previous["clock_id"] == incoming["clock_id"]:
        result = copy.deepcopy(min((previous, incoming), key=lambda t: t["started_ns"]))
        if previous["finished_clock_id"] == incoming["finished_clock_id"]:
            end = max((previous, incoming), key=lambda t: t["finished_ns"])
            result.update({k: end[k] for k in ("finished_at", "finished_ns", "finished_clock_id")})
            result["elapsed_ms"] = (
                (result["finished_ns"] - result["started_ns"]) / 1_000_000
                if result["clock_id"] == result["finished_clock_id"]
                else None
            )
        return result
    # Unrelated clocks cannot be ordered using their monotonic readings.
    return copy.deepcopy(incoming)
