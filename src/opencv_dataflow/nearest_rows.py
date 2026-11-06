"""Exact nearest-neighbor search over numeric feature tables."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .sample_tables import MAX_SAMPLE_VALUES, finite_features

_METRICS = {
    "euclidean": cv2.NORM_L2,
    "squared_euclidean": cv2.NORM_L2SQR,
    "manhattan": cv2.NORM_L1,
    "hamming": cv2.NORM_HAMMING,
    "hamming2": cv2.NORM_HAMMING2,
}
_FIELDS = ("queries", "references", "candidate_mask")
_MAX_FEATURE_COMPARISONS = 100_000_000


def _configured(value):
    return not isinstance(value, (list, tuple)) or bool(value)


def _features(value, field, packed):
    try:
        table = finite_features(value, max_columns=4_096)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid {field}: {exc}", field) from exc
    if packed and (np.any((table < 0) | (table > 255)) or np.any(table != np.trunc(table))):
        raise ValidationError(f"{field.capitalize()} must contain whole bytes for Hamming", field)
    return table.astype(np.uint8 if packed else np.float32)


def _candidate_mask(value):
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Use a rectangular candidate mask", "candidate_mask") from exc
    if (
        mask.ndim != 2
        or mask.size == 0
        or mask.size > MAX_SAMPLE_VALUES
        or mask.dtype.kind not in "biuf"
        or np.any((mask != 0) & (mask != 1))
    ):
        raise ValidationError("Use a nonempty mask table of zeros and ones", "candidate_mask")
    return mask.astype(np.uint8)


class FindNearestRows(Block):
    type_id, title = "cv.nearest_rows", "Find nearest rows"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = tuple(
        Port(key, "data", required=False, required_when_connected=True) for key in _FIELDS
    )
    outputs = (Port("indices", "data"), Port("distances", "data"))
    process_inputs_readonly = True
    defaults = {
        "queries": [],
        "references": [],
        "candidate_mask": [],
        "metric": "euclidean",
        "neighbors": 1,
        "crosscheck": False,
    }
    parameter_choices = {"metric": tuple(_METRICS)}
    parameter_ranges = {"neighbors": NumericRange(1, 256, whole=True)}
    parameter_help = {
        "queries": "One query vector per row. Wired data takes precedence.",
        "references": "One reference vector per row. Wired data takes precedence.",
        "candidate_mask": "Optional query-by-reference table. A 1 allows that pair to match.",
        "metric": "Hamming counts differing bits. Hamming2 counts differing two-bit groups.",
        "neighbors": "Maximum matches per query, capped at the number of reference rows.",
        "crosscheck": "Keep mutual nearest neighbors. Requires one neighbor and an empty mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["crosscheck"] and p["neighbors"] != 1:
            raise ValidationError("Crosscheck requires one neighbor", "crosscheck", "neighbors")
        packed = p["metric"] in ("hamming", "hamming2")
        for field in ("queries", "references"):
            if _configured(p[field]):
                _features(p[field], field, packed)
        if _configured(p["candidate_mask"]):
            if p["crosscheck"]:
                raise ValidationError(
                    "Crosscheck requires an empty candidate mask", "crosscheck", "candidate_mask"
                )
            _candidate_mask(p["candidate_mask"])

    def process(self, inputs):
        p = self.parameters
        packed = p["metric"] in ("hamming", "hamming2")
        queries = _features(inputs.get("queries", p["queries"]), "queries", packed)
        references = _features(inputs.get("references", p["references"]), "references", packed)
        if queries.shape[1] != references.shape[1]:
            raise ValidationError("Query and reference feature counts must match")
        shape = (len(queries), len(references))
        pairs = shape[0] * shape[1]
        if pairs > MAX_SAMPLE_VALUES or pairs * queries.shape[1] > _MAX_FEATURE_COMPARISONS:
            raise ValidationError("Search is too large. Use smaller query or reference batches.")
        mask = None
        if "candidate_mask" in inputs or _configured(p["candidate_mask"]):
            if p["crosscheck"]:
                raise ValidationError(
                    "Crosscheck requires an empty candidate mask", "crosscheck", "candidate_mask"
                )
            mask = _candidate_mask(inputs.get("candidate_mask", p["candidate_mask"]))
            if mask.shape != shape:
                raise ValidationError(
                    "Mask needs one row per query and one column per reference", "candidate_mask"
                )
        distances, indices = cv2.batchDistance(
            queries,
            references,
            cv2.CV_32S if packed else cv2.CV_32F,
            normType=_METRICS[p["metric"]],
            K=min(int(p["neighbors"]), len(references)),
            mask=mask,
            crosscheck=p["crosscheck"],
        )
        found = indices >= 0
        return Result(
            {
                "indices": np.where(found, indices, None).tolist(),
                "distances": np.where(found, distances, None).tolist(),
            },
            message=f"Found {np.count_nonzero(found)} matches across {len(queries)} queries",
        )
