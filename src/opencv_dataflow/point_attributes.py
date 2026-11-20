"""Scalar inspection columns aligned with the retained XYZ rows of a point cloud."""

import numpy as np

POINT_ATTRIBUTE_DTYPES = {
    "intensity": np.dtype("<f8"),
    "confidence": np.dtype("<f8"),
    "source_index": np.dtype("<i4"),
    "source_scan": np.dtype("<i4"),
}


def missing_value(name):
    return np.nan if POINT_ATTRIBUTE_DTYPES[name].kind == "f" else -1


def known_rows(name, values):
    return ~np.isnan(values) if POINT_ATTRIBUTE_DTYPES[name].kind == "f" else values >= 0


def validate_point_attributes(attributes, count):
    if not isinstance(attributes, dict) or attributes.keys() - POINT_ATTRIBUTE_DTYPES.keys():
        raise ValueError("Unsupported point attribute names")
    columns = {}
    for name, samples in attributes.items():
        dtype = POINT_ATTRIBUTE_DTYPES[name]
        values = np.asarray(samples)
        if values.shape != (count,) or values.dtype.kind not in "iuf":
            raise ValueError(f"Point attribute {name} needs one numeric scalar per XYZ row")
        if dtype.kind == "i":
            within_range = (values >= -1) & (values <= np.iinfo(dtype).max)
            if values.dtype.kind == "f":
                within_range &= values == np.rint(values)
            if not within_range.all():
                raise ValueError(f"Point attribute {name} needs int32 indices or -1 for missing")
        elif np.isinf(values).any():
            raise ValueError(f"Point attribute {name} needs finite values or NaN for missing")
        elif values.dtype.kind in "iu" and (np.any(values > 2**53) or np.any(values < -(2**53))):
            raise ValueError(f"Point attribute {name} exceeds exact float64 integer precision")
        with np.errstate(over="ignore", invalid="ignore"):
            column = np.ascontiguousarray(values, dtype)
        if dtype.kind == "f" and np.isinf(column).any():
            raise ValueError(f"Point attribute {name} exceeds the float64 range")
        columns[name] = column
    return columns


def mean_point_attributes(attributes, groups, group_count):
    """Average measured scalars at voxel centroids, omitting individual row IDs."""
    output = {}
    for name, values in attributes.items():
        if POINT_ATTRIBUTE_DTYPES[name].kind != "f":
            continue
        valid = known_rows(name, values)
        samples, labels = values[valid], groups[valid]
        scale = np.zeros(group_count, np.float64)
        np.maximum.at(scale, labels, np.abs(samples))
        scale[scale == 0] = 1
        count = np.bincount(labels, minlength=group_count)
        total = np.bincount(labels, weights=samples / scale[labels], minlength=group_count)
        mean = np.divide(total, count, out=np.full(group_count, np.nan), where=count != 0)
        output[name] = np.clip(mean, -1, 1) * scale
    return output


def merge_point_attributes(selections):
    """Keep every available column and disambiguate scan-local row indices."""
    names = set().union(*(attrs.keys() for attrs, _ in selections))
    output = {}
    for name, dtype in POINT_ATTRIBUTE_DTYPES.items():
        if name not in names or name == "source_scan":
            continue
        output[name] = np.concatenate(
            [
                attrs[name][rows]
                if name in attrs
                else np.full(len(rows), missing_value(name), dtype)
                for attrs, rows in selections
            ]
        )
    spans = []
    if names & {"source_index", "source_scan"}:
        chunks = []
        start = 0
        for attrs, rows in selections:
            source = attrs.get("source_scan")
            if source is None:
                labels = np.zeros(len(rows), np.int32)
                count = 1
            else:
                identifiers, labels = np.unique(source[rows], return_inverse=True)
                count = len(identifiers)
            chunks.append((labels + start).astype(np.int32, copy=False))
            spans.append([start, start + count])
            start += count
        output["source_scan"] = np.concatenate(chunks)
    return output, spans
