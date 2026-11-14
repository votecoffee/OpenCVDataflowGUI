"""Bounded local framing and portable ROS Image packets (Python 3.10+)."""

import base64
import copy
import json
import math
import struct
from dataclasses import dataclass, field

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_DATA_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 24 * 1024 * 1024
ENCODINGS = {
    "mono8": ("uint8", 1, "gray"),
    "mono16": ("uint16", 1, "gray"),
    "rgb8": ("uint8", 3, "rgb"),
    "bgr8": ("uint8", 3, "bgr"),
    "rgba8": ("uint8", 4, "rgba"),
    "bgra8": ("uint8", 4, "bgra"),
    "rgb16": ("uint16", 3, "rgb"),
    "bgr16": ("uint16", 3, "bgr"),
    "rgba16": ("uint16", 4, "rgba"),
    "bgra16": ("uint16", 4, "bgra"),
    "8UC1": ("uint8", 1, "gray"),
    "16UC1": ("uint16", 1, "gray"),
    "16SC1": ("int16", 1, "gray"),
    "32SC1": ("int32", 1, "gray"),
    "32FC1": ("float32", 1, "gray"),
    "64FC1": ("float64", 1, "gray"),
}


def json_bytes(value, limit=MAX_FRAME_BYTES):
    raw = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(raw) > limit:
        raise ValueError(f"JSON exceeds {limit} bytes")
    return raw


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key!r}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value}")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        part = stream.read(count - len(result))
        if not part:
            raise EOFError("ROS worker pipe closed")
        result.extend(part)
    return bytes(result)


def read_frame(stream):
    length = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ValueError("Invalid ROS worker frame length")
    return parse_json(read_exact(stream, length))


def write_frame(stream, value):
    raw = json_bytes(value)
    for part in (struct.pack("!I", len(raw)), raw):
        view = memoryview(part)
        while view:
            written = stream.write(view)
            if not written:
                raise EOFError("ROS worker pipe write failed")
            view = view[written:]
    stream.flush()


def header_value(header):
    if not isinstance(header, dict) or set(header) != {"stamp", "frame_id"}:
        raise ValueError("ROS header requires stamp and frame_id")
    stamp, frame_id = header["stamp"], header["frame_id"]
    if (
        not isinstance(stamp, dict)
        or set(stamp) != {"sec", "nanosec"}
        or type(stamp["sec"]) is not int
        or not -(2**31) <= stamp["sec"] < 2**31
        or type(stamp["nanosec"]) is not int
        or not 0 <= stamp["nanosec"] < 1_000_000_000
        or not isinstance(frame_id, str)
        or "\0" in frame_id
        or len(frame_id) > 1024
    ):
        raise ValueError("Invalid ROS header timestamp or frame_id")
    return copy.deepcopy(header)


@dataclass
class ROSImage:
    height: int
    width: int
    encoding: str
    step: int
    is_bigendian: int
    data: bytes
    header: dict
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.encoding not in ENCODINGS:
            raise ValueError(
                "Unsupported ROS Image encoding; convert Bayer/YUV/vector images explicitly"
            )
        dtype, channels, _ = ENCODINGS[self.encoding]
        itemsize = int("".join(c for c in dtype if c.isdigit())) // 8
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or not 1 <= self.height <= 16384
            or not 1 <= self.width <= 16384
            or type(self.step) is not int
            or self.step < self.width * channels * itemsize
            or type(self.is_bigendian) is not int
            or self.is_bigendian not in (0, 1)
            or not isinstance(self.data, bytes)
            or len(self.data) != self.step * self.height
            or len(self.data) > MAX_IMAGE_BYTES
        ):
            raise ValueError("ROS Image dimensions/stride/buffer are invalid or exceed 16 MiB")
        header_value(self.header)
        if not isinstance(self.metadata, dict):
            raise ValueError("ROS Image metadata must be an object")
        json_bytes(self.metadata, MAX_DATA_BYTES)

    def describe(self):
        return {
            "model": "ros2_image_v1",
            "height": self.height,
            "width": self.width,
            "encoding": self.encoding,
            "step": self.step,
            "is_bigendian": self.is_bigendian,
            "header": self.header,
            "byte_count": len(self.data),
        }

    def to_wire(self):
        self.__post_init__()
        result = self.describe()
        result.pop("byte_count")
        result["data_base64"] = base64.b64encode(self.data).decode("ascii")
        return result

    @classmethod
    def from_wire(cls, value):
        if not isinstance(value, dict) or value.get("model") != "ros2_image_v1":
            raise ValueError("Expected a ROS Image wire packet")
        encoded = value["data_base64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("ROS image buffer exceeds the limit")
        raw = base64.b64decode(encoded, validate=True)
        return cls(
            value["height"],
            value["width"],
            value["encoding"],
            value["step"],
            value["is_bigendian"],
            raw,
            header_value(value["header"]),
        )
