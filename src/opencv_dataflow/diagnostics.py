"""Per-block logger handles and bounded, structured diagnostic records (no Qt)."""

import json
import logging
import math
import sys
import threading
import traceback

from .activity import ActivityLog, report

LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
COLORS = {
    "TRACE": "#8998ac",
    "DEBUG": "#75b5df",
    "INFO": "#d9e3f0",
    "WARNING": "#efbe55",
    "ERROR": "#ff7b79",
    "CRITICAL": "#ee8df5",
}


def level_name(level):
    if isinstance(level, str):
        level = {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(level.upper(), level.upper())
        if level in LEVELS:
            return level
    elif type(level) is int and level in LEVELS.values():
        return next(name for name, value in LEVELS.items() if value == level)
    raise ValueError("Diagnostic level must be TRACE, DEBUG, INFO, WARNING, ERROR or CRITICAL")


def bounded_text(value, size):
    text = str(value)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) > size:
        text = raw[:size].decode("utf-8", errors="ignore") + "\n[truncated]"
    return text


def parameters_snapshot(value):
    def clean(item, depth=0):
        if depth > 8:
            return "[maximum nesting reached]"
        if item is None or type(item) in (bool, int):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else str(item)
        if isinstance(item, str):
            return bounded_text(item, 4096)
        if isinstance(item, dict):
            result = {str(k): clean(v, depth + 1) for k, v in list(item.items())[:128]}
            if len(item) > 128:
                result["[omitted]"] = len(item) - 128
            return result
        if isinstance(item, (list, tuple)):
            return [clean(v, depth + 1) for v in item[:128]] + (
                [f"[{len(item) - 128} items omitted]"] if len(item) > 128 else []
            )
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            if item.shape == () and hasattr(item, "item"):
                return clean(item.item(), depth + 1)
            return {"shape": list(item.shape), "dtype": str(item.dtype), "data": "[not retained]"}
        try:
            return bounded_text(repr(item), 512)
        except Exception:
            return f"[{type(item).__name__}]"

    result = clean(value)
    encoded = json.dumps(result, allow_nan=False, ensure_ascii=False)
    if len(encoded.encode()) > 16384:
        return {"truncated": True, "preview": bounded_text(encoded, 15000)}
    return result


def exception_details(exc_info):
    if exc_info is True:
        exc_info = sys.exc_info()
    if isinstance(exc_info, BaseException):
        exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
    if not exc_info or exc_info[1] is None:
        return None
    typ, exc, tb = exc_info
    formatted = "".join(
        traceback.TracebackException(typ, exc, tb, capture_locals=False).format(chain=True)
    )
    return {
        "type": f"{typ.__module__}.{typ.__qualname__}",
        "message": bounded_text(exc, 8192),
        "traceback": bounded_text(formatted, 32768),
        "truncated": len(formatted.encode("utf-8", errors="replace")) > 32768,
    }


def enrich(event):
    kind = event["kind"]
    severity = "DEBUG"
    if kind in ("input", "registered", "ticket_assigned"):
        severity = "TRACE"
    elif kind in ("exception",):
        severity = "ERROR"
    elif kind in ("blocked", "rejected", "dropped"):
        severity = "WARNING"
    elif kind == "finish":
        severity = {"error": "ERROR", "fail": "WARNING", "pass": "INFO"}.get(
            event.get("judgment"), "DEBUG"
        )
    elif kind not in ("start", "unblocked", "queue_busy", "queue_wait", "queue_resume"):
        severity = "INFO"
    event.setdefault("severity", severity)
    event.setdefault(
        "message",
        bounded_text(
            event.get("data", {}).get("message")
            or event.get("data", {}).get("reason")
            or event.get("result", {}).get("message")
            or kind.replace("_", " "),
            8192,
        ),
    )
    event.setdefault("parameters", {})
    event.setdefault("exception", None)
    event["severity"] = level_name(event["severity"])
    if not isinstance(event["message"], str) or not isinstance(event["parameters"], dict):
        raise ValueError("Diagnostics require a text message and parameter object")
    error = event["exception"]
    if error is not None and (
        not isinstance(error, dict)
        or any(not isinstance(error.get(k), str) for k in ("type", "message", "traceback"))
    ):
        raise ValueError("Invalid diagnostic exception record")


class BlockLogger:
    def __init__(self, block):
        self.block = block
        self.standalone = ActivityLog(max_events=2000, max_bytes=4 * 1024 * 1024)

    @property
    def name(self):
        return getattr(self.block, "_activity_logical_path", self.block.id)

    def bind(self, log):
        if log is self.standalone:
            return
        for event in self.standalone.snapshot()["events"]:
            event["source"].update(
                path=getattr(self.block, "_activity_path", self.block.id), logical_path=self.name
            )
            log.emit(event)
        self.standalone = ActivityLog(max_events=2000, max_bytes=4 * 1024 * 1024)

    def snapshot(self):
        log = self.block._activity or self.standalone
        snapshot = log.snapshot()
        snapshot["events"] = [
            e
            for e in snapshot["events"]
            if e["source"].get("logical_path", e["source"]["path"]) == self.name
        ]
        return snapshot

    def isEnabledFor(self, level):
        level_name(level)
        return True

    def log(self, level, message, *args, parameters=None, exc_info=None, values=None):
        if args:
            message = str(message) % args
        error = exception_details(exc_info)
        if error and getattr(self.block, "_execution_owner", None) == threading.get_ident():
            self.block._diagnostic_exception = error
        report(
            self.block,
            "exception" if error else "diagnostic",
            data={"message": bounded_text(message, 2048)} if error else {},
            values=values,
            diagnostic={
                "severity": level_name(level),
                "message": bounded_text(message, 8192),
                "parameters": parameters_snapshot(
                    self.block.parameters if parameters is None else parameters
                ),
                "exception": error,
            },
        )

    def trace(self, message, *args, **kwargs):
        self.log(5, message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.log(logging.WARNING, message, *args, **kwargs)

    warn = warning

    def error(self, message, *args, **kwargs):
        self.log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message, *args, exc_info=True, **kwargs):
        self.log(logging.ERROR, message, *args, exc_info=exc_info, **kwargs)
