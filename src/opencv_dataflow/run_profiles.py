"""Observation budgets for development and mature processing jobs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunProfile:
    name: str
    log_level: int
    activity_spans: bool
    parameter_snapshots: bool
    input_previews: bool
    graphics: bool
    output_previews: bool
    readonly_input_optimization: bool
    ticket_details: bool
    ui_interval_ms: int
    iteration_pause_ms: int


PROFILES = {
    "debug": RunProfile("debug", 5, True, True, True, True, True, False, True, 0, 50),
    "balanced": RunProfile("balanced", 20, True, False, False, False, True, True, True, 200, 0),
    "lean": RunProfile("lean", 30, False, False, False, False, False, True, False, 500, 0),
}


def profile(value="debug"):
    if isinstance(value, RunProfile):
        return value
    if not isinstance(value, str) or value not in PROFILES:
        raise ValueError("Run profile must be debug, balanced or lean")
    return PROFILES[value]


def event_level(kind, state=""):
    if kind in ("registered", "input", "ticket_assigned"):
        return 5
    if kind == "exception":
        return 40
    if kind in ("blocked", "rejected", "dropped"):
        return 30
    if kind == "finish":
        return {"error": 40, "fail": 30, "pass": 20}.get(state, 10)
    if kind in ("start", "unblocked", "queue_busy", "queue_wait", "queue_resume"):
        return 10
    return 20


def captures(block, kind, level=None):
    policy = getattr(block, "run_profile", PROFILES["debug"])
    if policy.activity_spans and kind in ("start", "finish"):
        return True
    return (
        event_level(kind, getattr(block, "state", "")) if level is None else level
    ) >= policy.log_level
