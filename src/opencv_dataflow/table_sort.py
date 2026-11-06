"""Sort numeric tables while tracking the original position of each result."""

from numbers import Integral

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .sample_tables import read_sample_table


def _table(value):
    try:
        table = read_sample_table(value)
    except ValidationError as exc:
        raise ValidationError(str(exc), "table") from exc
    if table.dtype.itemsize > 8:
        raise ValidationError("Use table values with at most 64 bits", "table")
    if not np.isfinite(table).all():
        raise ValidationError("Table values must be finite", "table")
    if (
        table.dtype.kind == "f"
        and isinstance(value, (list, tuple))
        and np.max(np.abs(table)) >= 2**53
        and any(
            isinstance(item, Integral) and abs(int(item)) > 2**53 for row in value for item in row
        )
    ):
        raise ValidationError(
            "Large integer values need an integer table to preserve precision", "table"
        )
    return table


def _sort_indices(keys, axis, descending):
    # Keep integer keys outside float64's exact range in NumPy.
    if keys.dtype.kind in "iu" and (np.any(keys > 2**53) or np.any(keys < -(2**53))):
        return np.argsort(~keys if descending else keys, axis=axis, kind="stable")
    flags = cv2.SORT_EVERY_ROW if axis == 1 else cv2.SORT_EVERY_COLUMN
    flags |= cv2.SORT_DESCENDING if descending else cv2.SORT_ASCENDING
    order = cv2.sortIdx(np.ascontiguousarray(keys, dtype=np.float64), flags)
    sorted_keys = np.take_along_axis(keys, order, axis=axis)

    # SortIdx may reorder equal keys. Refine only when a tie is out of input order.
    ordered = np.moveaxis(order, axis, -1)
    sorted_keys = np.moveaxis(sorted_keys, axis, -1)
    changed = sorted_keys[..., 1:] != sorted_keys[..., :-1]
    if np.any(~changed & (ordered[..., 1:] < ordered[..., :-1])):
        groups = np.empty(ordered.shape, dtype=np.int32)
        groups[..., 0] = 0
        np.cumsum(changed, axis=-1, dtype=np.int32, out=groups[..., 1:])
        within_groups = np.lexsort((ordered, groups))
        order = np.moveaxis(np.take_along_axis(ordered, within_groups, axis=-1), -1, axis)
    return order


class SortNumericTable(Block):
    type_id, title = "cv.sort_numeric_table", "Sort numeric table"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (Port("table", "data", required=False, required_when_connected=True),)
    outputs = (Port("table", "data"), Port("indices", "data"))
    defaults = {"table": [], "mode": "rows_by_column", "order": "ascending", "key_index": 0}
    parameter_choices = {
        "mode": ("rows_by_column", "columns_by_row", "within_rows", "within_columns"),
        "order": ("ascending", "descending"),
    }
    parameter_ranges = {"key_index": NumericRange(0, 99_999, whole=True)}
    parameter_help = {
        "table": "Enter a rectangular numeric table or wire a table from another block.",
        "mode": (
            "Move whole rows or columns by a key, or sort values independently within each slice."
        ),
        "key_index": "Zero-based column for Rows by column or row for Columns by row.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        value = p["table"]
        if not (isinstance(value, (list, tuple)) and not value):
            _table(value)

    def process(self, inputs):
        p = self.parameters
        table = _table(inputs.get("table", p["table"]))
        mode = p["mode"]
        whole_records = mode in ("rows_by_column", "columns_by_row")
        axis = int(mode in ("columns_by_row", "within_rows"))
        keys = table
        if whole_records:
            key = int(p["key_index"])
            if key >= table.shape[1 - axis]:
                raise ValidationError("Key index is outside the table", "key_index")
            keys = table[:, key : key + 1] if axis == 0 else table[key : key + 1, :]
        indices = _sort_indices(keys, axis, p["order"] == "descending")
        if whole_records:
            indices = indices.ravel()
            result = np.take(table, indices, axis=axis)
        else:
            result = np.take_along_axis(table, indices, axis=axis)
        return Result({"table": result.tolist(), "indices": indices.tolist()})
