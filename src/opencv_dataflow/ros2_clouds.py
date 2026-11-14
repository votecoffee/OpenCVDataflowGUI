"""Explicit PointCloud2 projections into the app's finite float32 PointCloud type."""

import copy

import numpy as np

from .parameter_validation import ValidationError
from .pointclouds import PointCloud
from .ros2_support.cloud_wire import FIELD_TYPES, MAX_CLOUD_BYTES, ROSPointCloud


def cloud_records(packet):
    """A private structured view, respecting record padding, row padding and endian."""
    if not isinstance(packet, ROSPointCloud):
        raise ValidationError("Expected a portable ROSPointCloud packet")
    packet.__post_init__()
    endian = ">" if packet.is_bigendian else "<"
    formats = []
    for field in packet.fields:
        dtype = np.dtype(FIELD_TYPES[field["datatype"]][0]).newbyteorder(endian)
        formats.append(dtype if field["count"] == 1 else (dtype, (field["count"],)))
    dtype = np.dtype(
        {
            "names": [f["name"] for f in packet.fields],
            "formats": formats,
            "offsets": [f["offset"] for f in packet.fields],
            "itemsize": packet.point_step,
        }
    )
    return np.ndarray(
        (packet.height, packet.width),
        dtype=dtype,
        buffer=packet.data,
        strides=(packet.row_step, packet.point_step),
    )


def decode_cloud(
    packet,
    *,
    units="m",
    invalid_points="drop",
    colors="auto",
    normals="normalize",
    empty_frame="ros_cloud",
):
    if units not in ("m", "mm") or invalid_points not in ("drop", "reject"):
        raise ValidationError("Choose m/mm and drop/reject for invalid points")
    if colors not in ("auto", "ignore") or normals not in ("normalize", "ignore"):
        raise ValidationError("Choose auto/ignore colors and normalize/ignore normals")
    if not isinstance(empty_frame, str) or not empty_frame or "\0" in empty_frame:
        raise ValidationError("Name a fallback frame for messages with empty frame_id")
    records = cloud_records(packet)
    fields = {f["name"]: f for f in packet.fields}

    def scalar(name, types):
        item = fields.get(name)
        if item is None or item["count"] != 1 or item["datatype"] not in types:
            raise ValidationError(f"PointCloud2 {name} needs a scalar field of a supported type")
        return records[name].reshape(-1)

    xyz = np.column_stack([scalar(k, (7, 8)) for k in ("x", "y", "z")])
    with np.errstate(over="ignore", invalid="ignore"):
        xyz = xyz.astype(np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    invalid_xyz = int(np.count_nonzero(~keep))
    if invalid_xyz and packet.is_dense:
        raise ValidationError("Dense PointCloud2 contains invalid XYZ or float32 overflow")
    used = {"x", "y", "z"}
    vectors, invalid_normals = None, 0
    if normals == "normalize":
        normal_names = next(
            (
                names
                for names in (("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz"))
                if all(name in fields for name in names)
            ),
            None,
        )
        if normal_names is not None:
            vectors = np.column_stack([scalar(k, (7, 8)) for k in normal_names]).astype(np.float64)
            lengths = np.linalg.norm(vectors, axis=1)
            valid = np.isfinite(vectors).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)
            invalid_normals = int(np.count_nonzero(keep & ~valid))
            keep &= valid
            vectors = vectors[keep] / lengths[keep, None]
            used.update(normal_names)
    dropped = int(np.count_nonzero(~keep))
    if dropped and invalid_points == "reject":
        raise ValidationError("PointCloud2 has invalid XYZ/normals; choose Drop to discard them")
    rgb = None
    if colors == "auto":
        packed_name = next((k for k in ("rgba", "rgb") if k in fields), None)
        if packed_name is not None:
            values = scalar(packed_name, (6, 7))
            if fields[packed_name]["datatype"] == 7:
                values = values.view(
                    np.dtype("u4").newbyteorder(">" if packet.is_bigendian else "<")
                )
            packed = values.astype(np.uint32)
            rgb = np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255])[keep]
            used.add(packed_name)
        elif all(k in fields for k in ("r", "g", "b")):
            rgb = np.column_stack([scalar(k, (2,)) for k in ("r", "g", "b")])[keep]
            used.update(("r", "g", "b"))
    pixels, source_size = None, None
    if packet.height > 1 and packet.width > 0:
        indices = np.flatnonzero(keep)
        pixels = np.column_stack([indices % packet.width, indices // packet.width])
        source_size = (packet.width, packet.height)
    payload = copy.deepcopy(packet.metadata)
    payload["ros2"] = {
        **payload.get("ros2", {}),
        **packet.describe(),
        "message_type": "sensor_msgs/msg/PointCloud2",
        "declared_units": units,
    }
    details = {
        **packet.describe(),
        "units": units,
        "decoded_points": int(keep.sum()),
        "invalid_xyz": invalid_xyz,
        "invalid_normals": invalid_normals,
        "dropped": dropped,
        "ignored_fields": sorted(fields.keys() - used),
        "xyz_storage": "float32",
        "alpha_projected": False,
    }
    cloud = PointCloud(
        xyz[keep],
        units,
        packet.header["frame_id"] or empty_frame,
        rgb,
        vectors,
        pixels,
        source_size,
        payload,
    )
    return cloud, details


def encode_cloud(cloud, *, organization="unorganized", byte_order="little", stamp_mode="preserve"):
    if not isinstance(cloud, PointCloud):
        raise ValidationError("Connect a typed PointCloud")
    # Plugins can mutate dataclass fields after construction; validate a private copy again.
    cloud = copy.deepcopy(cloud)
    cloud.__post_init__()
    if cloud.units not in ("m", "mm"):
        raise ValidationError("ROS publication requires physical m/mm units; unitless is ambiguous")
    if organization not in ("unorganized", "source_pixels") or byte_order not in ("little", "big"):
        raise ValidationError("Choose unorganized/source_pixels and little/big endian")
    if stamp_mode not in ("preserve", "zero", "node_clock"):
        raise ValidationError("Choose preserve/zero/node_clock timestamp mode")
    fields = []

    def field(name, datatype):
        offset = sum(FIELD_TYPES[f["datatype"]][1] for f in fields)
        fields.append({"name": name, "offset": offset, "datatype": datatype, "count": 1})

    for name in ("x", "y", "z"):
        field(name, 7)
    if cloud.normals is not None:
        for name in ("normal_x", "normal_y", "normal_z"):
            field(name, 7)
    if cloud.colors is not None:
        field("rgb", 6)
    width, height = len(cloud.points), 1
    indices = np.arange(width)
    if organization == "source_pixels":
        if cloud.pixels is None or cloud.source_size is None:
            raise ValidationError(
                "Organized output requires retained unique source pixels and size"
            )
        width, height = cloud.source_size
        indices = cloud.pixels[:, 1].astype(np.int64) * width + cloud.pixels[:, 0]
        if len(np.unique(indices)) != len(indices):
            raise ValidationError("Organized output cannot map several points to the same pixel")
    step = sum(FIELD_TYPES[f["datatype"]][1] for f in fields)
    if width * height > 2_000_000 or step * width * height > MAX_CLOUD_BYTES:
        raise ValidationError("Encoded PointCloud2 exceeds 16 MiB or 2M points")
    endian = ">" if byte_order == "big" else "<"
    dtype = np.dtype([(f["name"], endian + FIELD_TYPES[f["datatype"]][0]) for f in fields])
    records = np.zeros(width * height, dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = np.nan
        records[name][indices] = cloud.points[:, axis] * (0.001 if cloud.units == "mm" else 1.0)
    if cloud.normals is not None:
        for axis, name in enumerate(("normal_x", "normal_y", "normal_z")):
            records[name] = np.nan
            records[name][indices] = cloud.normals[:, axis]
    if cloud.colors is not None:
        rgb = cloud.colors.astype(np.uint32)
        records["rgb"][indices] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    header = copy.deepcopy(
        cloud.metadata.get("ros2", {}).get(
            "header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": cloud.frame}
        )
    )
    header["frame_id"] = cloud.frame
    if stamp_mode == "zero":
        header["stamp"] = {"sec": 0, "nanosec": 0}
    return ROSPointCloud(
        height,
        width,
        fields,
        byte_order == "big",
        step,
        step * width,
        records.tobytes(),
        len(indices) == width * height,
        header,
        cloud.metadata,
    )
