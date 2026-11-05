"""Cumulative ticket statistics with a bounded recent identity ledger."""

import hashlib
import json
from collections import OrderedDict
from fractions import Fraction

from .tickets import ticket_key


class ProductionStatistics:
    def __init__(self):
        self.entries = OrderedDict()
        self.total = self.passed = self.failed = self.timed = 0
        self.elapsed_sum = 0.0

    def record(self, ticket, passed, elapsed, final=None, conflict=False):
        session, kind, value = ticket_key(ticket)
        if kind == "number":
            number = Fraction(value)
            value = [number.numerator, number.denominator]
        key = hashlib.sha256(json.dumps([session, kind, value]).encode()).hexdigest()
        old = self.entries.get(key)
        if old:
            self.passed -= int(old[0])
            self.failed -= int(not old[0])
            if old[1] is not None:
                self.elapsed_sum -= old[1]
                self.timed -= 1
            conflict |= old[3] or (old[2] is not None and final is not None and old[2] != final)
            final = old[2] if final is None else final
            if old[1] is not None:
                elapsed = max(old[1], elapsed) if elapsed is not None else old[1]
        else:
            self.total += 1
        passed = False if conflict else bool(final if final is not None else passed)
        self.entries[key] = [passed, elapsed, final, conflict]
        self.entries.move_to_end(key)
        self.passed += int(passed)
        self.failed += int(not passed)
        if elapsed is not None:
            self.timed += 1
            self.elapsed_sum += elapsed
        while len(self.entries) > 8192:
            self.entries.popitem(last=False)

    def summary(self):
        return {
            "total": self.total,
            "pass": self.passed,
            "fail": self.failed,
            "pass_rate": 100 * self.passed / self.total if self.total else None,
            "average_ms": self.elapsed_sum / self.timed if self.timed else None,
            "timed": self.timed,
        }

    def snapshot(self):
        return {
            "total": self.total,
            "pass": self.passed,
            "fail": self.failed,
            "timed": self.timed,
            "elapsed_sum": self.elapsed_sum,
            "entries": list(self.entries.items()),
        }

    def restore(self, value):
        self.total, self.passed, self.failed = value["total"], value["pass"], value["fail"]
        self.timed, self.elapsed_sum = value["timed"], value["elapsed_sum"]
        self.entries = OrderedDict(value["entries"][-8192:])
