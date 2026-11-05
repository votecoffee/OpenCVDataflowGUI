"""Validation for numeric feature tables shared by graph blocks."""

import numpy as np

from .parameter_validation import ValidationError


def read_sample_table(value):
    try:
        samples = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Samples must be a rectangular numeric table") from exc
    if (
        samples.ndim != 2
        or not 1 <= samples.shape[0] <= 100_000
        or not 1 <= samples.shape[1] <= 128
        or samples.size > 2_000_000
        or samples.dtype.kind not in "iuf"
    ):
        raise ValidationError(
            "Use a numeric table with 1–100,000 rows, 1–128 features and at most 2M values"
        )
    return samples
