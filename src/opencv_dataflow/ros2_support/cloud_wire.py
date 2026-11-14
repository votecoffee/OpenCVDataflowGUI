"""Portable, bounded PointCloud2 packets; stdlib only for the ROS interpreter."""

import base64
import copy
from dataclasses import dataclass, field

try:
    from .wire import MAX_DATA_BYTES, header_value, json_bytes
except ImportError:  # Fixed standalone ROS worker.
    from wire import MAX_DATA_BYTES, header_value, json_bytes

MAX_CLOUD_BYTES = 16 * 1024 * 1024
MAX_POINTS = 2_000_000
# Lyrical PointField datatypes, including its int64/uint64/bool additions.
FIELD_TYPES = {
    1: ("i1", 1),
    2: ("u1", 1),
    3: ("i2", 2),
    4: ("u2", 2),
    5: ("i4", 4),
    6: ("u4", 4),
    7: ("f4", 4),
    8: ("f8", 8),
    9: ("i8", 8),
    10: ("u8", 8),
    11: ("?", 1),
}


@dataclass
class ROSPointCloud:
    height: int
    width: int
    fields: list
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool
    header: dict
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or not 1 <= self.height <= MAX_POINTS
            or not 0 <= self.width <= MAX_POINTS
            or self.height * self.width > MAX_POINTS
            or type(self.point_step) is not int
            or not 1 <= self.point_step <= 65536
            or type(self.row_step) is not int
            or self.row_step < self.width * self.point_step
            or (not self.width and self.row_step != 0)
            or type(self.is_bigendian) is not bool
            or type(self.is_dense) is not bool
            or not isinstance(self.data, bytes)
            or len(self.data) != self.height * self.row_step
            or len(self.data) > MAX_CLOUD_BYTES
        ):
            raise ValueError(
                "PointCloud2 dimensions/stride/buffer invalid or exceed 16 MiB/2M points"
            )
        if not isinstance(self.fields, list) or not 1 <= len(self.fields) <= 64:
            raise ValueError("PointCloud2 requires 1..64 named fields")
        names = set()
        for item in self.fields:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "offset", "datatype", "count"}
                or not isinstance(item["name"], str)
                or not 1 <= len(item["name"]) <= 128
                or "\0" in item["name"]
                or item["name"] in names
                or type(item["datatype"]) is not int
                or item["datatype"] not in FIELD_TYPES
                or type(item["count"]) is not int
                or not 1 <= item["count"] <= 1024
                or type(item["offset"]) is not int
                or item["offset"] < 0
            ):
                raise ValueError(
                    "Invalid/duplicate PointCloud2 field name, datatype, count or offset"
                )
            if item["offset"] + item["count"] * FIELD_TYPES[item["datatype"]][1] > self.point_step:
                raise ValueError("PointCloud2 field extends beyond point_step")
            names.add(item["name"])
        header_value(self.header)
        if not isinstance(self.metadata, dict):
            raise ValueError("PointCloud2 local metadata must be an object")
        json_bytes(self.metadata, MAX_DATA_BYTES)

    def describe(self):
        return {
            "model": "ros2_pointcloud2_v1",
            "height": self.height,
            "width": self.width,
            "fields": copy.deepcopy(self.fields),
            "is_bigendian": self.is_bigendian,
            "point_step": self.point_step,
            "row_step": self.row_step,
            "is_dense": self.is_dense,
            "header": copy.deepcopy(self.header),
            "point_count": self.height * self.width,
            "byte_count": len(self.data),
        }

    def to_wire(self):
        self.__post_init__()
        result = self.describe()
        result.pop("point_count")
        result.pop("byte_count")
        result["data_base64"] = base64.b64encode(self.data).decode("ascii")
        return result

    @classmethod
    def from_wire(cls, value):
        if not isinstance(value, dict) or value.get("model") != "ros2_pointcloud2_v1":
            raise ValueError("Expected a portable ROS PointCloud2 packet")
        encoded = value["data_base64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_CLOUD_BYTES + 2) // 3):
            raise ValueError("PointCloud2 buffer exceeds 16 MiB")
        return cls(
            **{
                key: copy.deepcopy(value[key])
                for key in (
                    "height",
                    "width",
                    "fields",
                    "is_bigendian",
                    "point_step",
                    "row_step",
                    "is_dense",
                    "header",
                )
            },
            data=base64.b64decode(encoded, validate=True),
        )
