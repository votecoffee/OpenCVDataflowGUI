"""Reusable parameter limits and expected input validation failures."""

import math
from dataclasses import dataclass

import numpy as np


class ValidationError(ValueError):
    def __init__(self, message, *fields):
        super().__init__(message)
        self.fields = fields


@dataclass(frozen=True)
class NumericRange:
    minimum: float | None = None
    maximum: float | None = None
    whole: bool = False
    odd: bool = False
    zero_allowed: bool = False

    def describe(self):
        if self.minimum is not None and self.maximum is not None:
            limit = f"between {self.minimum:g} and {self.maximum:g}"
        elif self.minimum is not None:
            limit = f"at least {self.minimum:g}"
        elif self.maximum is not None:
            limit = f"at most {self.maximum:g}"
        else:
            limit = ""
        if self.odd:
            positive = self.minimum is not None and self.minimum > 0
            kind = "a positive odd integer" if positive else "an odd integer"
        elif self.whole:
            kind = "a whole number"
        else:
            kind = "a finite number"
        constraint = f"{kind} {limit}".strip()
        return f"zero or {constraint}" if self.zero_allowed else constraint

    def check(self, value, label, *fields):
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            raise ValidationError(f"{label} must be a finite number", *fields) from None
        if not math.isfinite(number):
            raise ValidationError(f"{label} must be a finite number", *fields)
        if self.whole:
            if isinstance(value, (bool, np.bool_, str)) or not number.is_integer():
                raise ValidationError(f"{label} must be a whole number", *fields)
            number = int(value)
        if self.zero_allowed and number == 0:
            return number
        if (
            (self.minimum is not None and number < self.minimum)
            or (self.maximum is not None and number > self.maximum)
            or (self.odd and number % 2 == 0)
        ):
            raise ValidationError(f"{label} must be {self.describe()}", *fields)
        return number
