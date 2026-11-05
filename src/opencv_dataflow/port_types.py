"""Connection compatibility and checked scalar conversion at the receiving port."""

import numpy as np

NUMERIC_KINDS = frozenset(("number", "integer", "float"))
SCALAR_KINDS = NUMERIC_KINDS | {"boolean", "text"}


def connection_mode(source, target):
    if source == target or target == "any":
        return "direct"
    if source == "any":
        return "runtime type check"
    if source == "pyramid" and target == "image":
        return "select pyramid level"
    if source == "depth" and target == "image":
        return "use depth samples as an image"
    if source == "image" and target == "depth":
        return "check depth shape and dtype"
    if source in NUMERIC_KINDS and target == "number":
        return "numeric widening"
    if (
        source in NUMERIC_KINDS and target in NUMERIC_KINDS
        or source in SCALAR_KINDS and target == "data"
        or source == "data" and target in SCALAR_KINDS
    ):
        return "checked scalar conversion"
    return None


def compatible(out_kind, in_kind):
    return connection_mode(out_kind, in_kind) is not None


def is_integer(value):
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, np.integer) and value.dtype.kind in "iu"
    )


def exact_float(port, value):
    if is_integer(value):
        value = int(value)  # Compare in Python to avoid NumPy's implicit float promotion.
    elif not isinstance(value, (float, np.floating)):
        raise TypeError(f"{port.name} requires a finite float")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{port.name} exceeds the finite float range") from exc
    if not np.isfinite(converted) or converted != value:
        raise ValueError(f"{port.name} requires an exactly representable finite float")
    return converted


def validate_scalar(port, value):
    kind = port.kind
    if kind == "number":
        valid = isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
    elif kind == "integer":
        valid = is_integer(value)
    elif kind == "float" and isinstance(value, (float, np.floating)):
        exact_float(port, value)
        return
    elif kind == "float":
        valid = False
    elif kind == "boolean":
        valid = isinstance(value, (bool, np.bool_))
    else:
        valid = isinstance(value, str)
    if not valid:
        expected = {
            "number": "a number",
            "integer": "an integer",
            "float": "a finite float",
            "boolean": "a boolean",
            "text": "text",
        }[kind]
        raise TypeError(f"{port.name} requires {expected}")


def coerce_scalar_input(port, value):
    """Convert actual scalar values without rounding, truncating or parsing text."""
    kind = port.kind
    if kind == "integer":
        if is_integer(value):
            return int(value)
        if (
            isinstance(value, (float, np.floating))
            and np.isfinite(value)
            and value == np.trunc(value)
        ):
            return int(value)
        raise TypeError(f"{port.name} requires an integer; conversion must be exact")
    if kind == "float" or (kind == "data" and isinstance(value, np.floating)):
        return exact_float(port, value)
    if kind == "data" and isinstance(value, np.integer) and is_integer(value):
        return int(value)
    if kind in ("boolean", "data") and isinstance(value, np.bool_):
        return bool(value)
    return value
