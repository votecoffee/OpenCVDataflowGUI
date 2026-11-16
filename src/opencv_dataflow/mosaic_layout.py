"""Versioned planar mosaic geometry with stable source identity and bounded binding."""

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .image_stitching import _transform, mosaic_geometry
from .images import image_format
from .parameter_validation import ValidationError
from .payloads import check_spatial, frame

MODEL = "tiled_scan_mosaic_v2"
IDENTITIES = ("ordered", "labels", "payload", "explicit")


def source_ids(group, contexts, mode, field="source_id", explicit=None):
    if mode == "labels":
        values = list(group.labels)
    elif mode == "payload":
        values = [context.get(field) for context in contexts]
    elif mode == "explicit":
        values = explicit
    else:
        raise ValidationError("Choose a stable source identity mode")
    if not isinstance(values, (list, tuple)) or len(values) != len(group.items):
        raise ValidationError("Provide one source ID per received image", "source_ids")
    if any(
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(c) < 32 for c in value)
        for value in values
    ):
        raise ValidationError("Source IDs require 1..256 visible characters", "source_ids")
    if len(set(values)) != len(values):
        raise ValidationError("Source IDs must be unique within a group", "source_ids")
    return list(values)


@dataclass(frozen=True)
class GridExtent:
    # Geometry needs dimensions only: never allocate absent calibration images.
    shape: tuple


def geometry(layout):
    if not isinstance(layout, dict) or layout.get("model") not in (MODEL, "tiled_scan_mosaic_v1"):
        raise ValidationError("Choose a supported tiled mosaic layout")
    size = layout.get("image_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(x) is not int or not 1 <= x <= 1000000 for x in size)
    ):
        raise ValidationError("Invalid mosaic canvas size")
    records = layout.get("views")
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValidationError("A layout requires 1..64 source records")
    shapes, transforms, formats = [], [], set()
    for i, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or type(record.get("index")) is not int
            or record["index"] != i
        ):
            raise ValidationError("Layout records require contiguous calibrated indices")
        dims = record.get("source_size")
        if (
            not isinstance(dims, list)
            or len(dims) != 2
            or any(type(x) is not int or not 8 <= x < 32767 for x in dims)
            or dims[0] * dims[1] > 8388608
        ):
            raise ValidationError("Invalid calibrated source dimensions")
        depth, encoding = record.get("source_dtype"), record.get("source_format")
        if depth not in ("uint8", "uint16", "float32", "float64") or encoding not in (
            "gray",
            "rgb",
            "bgr",
            "rgba",
            "bgra",
        ):
            raise ValidationError("Unsupported calibrated source depth/encoding")
        if not isinstance(record.get("label"), str):
            raise ValidationError("Calibrated source display labels must be strings")
        try:
            check_spatial({"spatial": record.get("source_spatial")}, dims)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise ValidationError("Invalid calibrated source spatial metadata") from exc
        formats.add((depth, encoding))
        shapes.append(GridExtent((dims[1], dims[0])))
        transforms.append(_transform(record.get("source_to_mosaic")))
    if len(formats) != 1:
        raise ValidationError("Calibrated sources must use matching depths and encodings")
    actual, shift = mosaic_geometry(shapes, transforms)
    if list(actual) != size or not np.allclose(shift, np.eye(3), atol=1e-6, rtol=0):
        raise ValidationError("Full calibrated geometry disagrees with mosaic bounds")
    return tuple(size), records, transforms


def identity_core(layout):
    fields = (
        "index",
        "source_id",
        "source_size",
        "source_dtype",
        "source_format",
        "source_to_mosaic",
        "source_spatial",
    )
    return {
        "model": MODEL,
        "image_size": layout["image_size"],
        "source_identity": layout["source_identity"],
        "views": [{key: record.get(key) for key in fields} for record in layout["views"]],
    }


def layout_digest(layout):
    try:
        raw = json.dumps(
            identity_core(layout), allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    except (ValueError, TypeError, KeyError) as exc:
        raise ValidationError("Layout geometry must be finite JSON data") from exc
    return hashlib.sha256(raw).hexdigest()


def seal_layout(layout):
    """Validate a v2 copy; acquisition tickets and display labels do not define its identity."""
    output = copy.deepcopy(layout)
    output["layout_id"] = layout_digest(output)
    validate_layout(output)
    return output


def validate_layout(layout):
    size, records, transforms = geometry(layout)
    if layout["model"] == MODEL:
        identity = layout.get("source_identity")
        if (
            not isinstance(identity, dict)
            or identity.get("mode") not in IDENTITIES[1:]
            or not isinstance(identity.get("field"), str)
            or not identity["field"]
            or len(identity["field"]) > 128
        ):
            raise ValidationError("Invalid stable source identity definition")
        ids = [record.get("source_id") for record in records]
        if any(
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 256
            or any(ord(c) < 32 for c in value)
            for value in ids
        ) or len(set(ids)) != len(ids):
            raise ValidationError("Calibrated source IDs must be valid and unique")
        if layout.get("layout_id") != layout_digest(layout):
            raise ValidationError(
                "Layout identity differs from calibrated geometry; create a new revision"
            )
    return size, records, transforms


def bind_layout(layout, group, images, pixels, contexts, settings, explicit=None):
    size, records, all_transforms = validate_layout(layout)
    identity = layout["source_identity"]
    supplied = explicit if explicit is not None else settings["source_ids"]
    mode = "explicit" if explicit is not None or settings["source_ids"] else identity["mode"]
    ids = source_ids(group, contexts, mode, identity["field"], supplied)
    indices = {value: i for i, value in enumerate(ids)}
    registered = {record["source_id"] for record in records}
    unknown = sorted(set(ids) - registered)
    if unknown:
        raise ValidationError("Unrecognized mosaic source IDs: " + ", ".join(unknown))
    missing = [record["source_id"] for record in records if record["source_id"] not in indices]
    if missing and settings["missing_views"] == "error":
        raise ValidationError(
            "Missing calibrated mosaic sources: " + ", ".join(missing), "missing_views"
        )
    selected, grid_checks = [], []
    for record in records:
        source_id = record["source_id"]
        if source_id not in indices:
            continue
        current = indices[source_id]
        pixel, image, context = pixels[current], images[current], contexts[current]
        if (
            record["source_size"] != list(pixel.shape[1::-1])
            or record["source_dtype"] != pixel.dtype.name
            or record["source_format"] != image_format(image)
        ):
            raise ValidationError("Mosaic source grid, depth or encoding changed: " + source_id)
        before, after = record.get("source_spatial"), context.get("spatial")
        evidence = "dimensions_only"
        if settings["grid_policy"] == "strict" and before is not None:
            if (
                after is None
                or before["image_size"] != after["image_size"]
                or before["root_size"] != after["root_size"]
                or not np.allclose(before["to_root"], after["to_root"], atol=1e-9, rtol=0)
                or (
                    before.get("root_id") is not None
                    and before.get("root_id") != after.get("root_id")
                )
            ):
                raise ValidationError(
                    "Mosaic source coordinate grid changed: " + source_id, "grid_policy"
                )
            evidence = "matching_declared_grid"
        elif settings["grid_policy"] == "pixel_indices":
            evidence = "caller_asserted_pixel_indices"
        selected.append((record["index"], current))
        grid_checks.append({"source_id": source_id, "evidence": evidence})
    status = {
        "complete": not missing,
        "present_source_ids": [records[i]["source_id"] for i, _ in selected],
        "missing_source_ids": missing,
        "registered_source_ids": [record["source_id"] for record in records],
        "grid_checks": grid_checks,
        "missing_policy": settings["missing_views"],
    }
    return size, [all_transforms[i] for i, _ in selected], selected, status


def canvas_frame(layout):
    return frame(layout["image_size"], "mosaic:" + layout["layout_id"])
