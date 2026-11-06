"""Validation for numeric feature tables shared by graph blocks."""

import numpy as np

from .parameter_validation import ValidationError

MAX_SAMPLE_VALUES = 2_000_000


def read_sample_table(value, *, max_columns=128):
    try:
        samples = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Samples must be a rectangular numeric table") from exc
    if (
        samples.ndim != 2
        or not 1 <= samples.shape[0] <= 100_000
        or not 1 <= samples.shape[1] <= max_columns
        or samples.size > MAX_SAMPLE_VALUES
        or samples.dtype.kind not in "iuf"
    ):
        raise ValidationError(
            f"Use a numeric table with 1–100,000 rows, 1–{max_columns:,} features "
            "and at most 2M values"
        )
    return samples


def finite_features(value, limit=1e12, *, max_columns=128):
    samples = np.ascontiguousarray(
        read_sample_table(value, max_columns=max_columns), dtype=np.float64
    )
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > limit:
        raise ValidationError(f"Features must be finite and within ±{limit:.0e}".replace("+", ""))
    return samples
