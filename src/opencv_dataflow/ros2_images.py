"""ROS Image conversion without cv_bridge, rclpy or an OpenCV ABI in the worker."""

import copy
import sys

import numpy as np

from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .payloads import metadata
from .ros2_support.wire import ENCODINGS, MAX_IMAGE_BYTES, ROSImage, header_value


def encode_image(image, *, encoding="auto", frame_id="", stamp_mode="preserve"):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim not in (2, 3)
        or not image.size
        or (image.ndim == 3 and image.shape[2] not in (3, 4))
    ):
        raise ValidationError("ROS Image publishing needs a 2D gray or RGB/BGR(A) image")
    payload = metadata(image)
    dtype = image.dtype.newbyteorder("=")
    color = image_format(image)
    if encoding == "auto":
        if image.ndim == 2:
            encoding = {
                "uint8": "mono8",
                "uint16": "mono16",
                "int16": "16SC1",
                "int32": "32SC1",
                "float32": "32FC1",
                "float64": "64FC1",
            }.get(dtype.name)
        elif dtype in (np.dtype("uint8"), np.dtype("uint16")) and color in (
            "rgb",
            "bgr",
            "rgba",
            "bgra",
        ):
            encoding = color + ("8" if dtype.itemsize == 1 else "16")
    if encoding not in ENCODINGS:
        raise ValidationError(
            "Choose a supported ROS encoding; convert HSV/Lab or other channel types first"
        )
    expected_dtype, channels, expected_color = ENCODINGS[encoding]
    if dtype != np.dtype(expected_dtype) or channels != (1 if image.ndim == 2 else image.shape[2]):
        raise ValidationError(
            "ROS encoding must match sample depth and channel count; convert explicitly"
        )
    if image.nbytes > MAX_IMAGE_BYTES:
        raise ValidationError("ROS Image exceeds 16 MiB")
    pixels = np.asarray(image)
    if channels > 1 and color != expected_color:
        if {color, expected_color} in ({"rgb", "bgr"}, {"rgba", "bgra"}):
            pixels = pixels[..., [2, 1, 0] + ([3] if channels == 4 else [])]
        else:
            raise ValidationError("Convert the image's color encoding before ROS publication")
    pixels = np.ascontiguousarray(pixels, dtype=dtype)
    header = copy.deepcopy(
        payload.get("ros2", {}).get(
            "header",
            {
                "stamp": {"sec": 0, "nanosec": 0},
                "frame_id": "",
            },
        )
    )
    if stamp_mode == "zero":
        header["stamp"] = {"sec": 0, "nanosec": 0}
    elif stamp_mode not in ("preserve", "node_clock"):
        raise ValidationError("Stamp mode must be preserve, zero or node_clock")
    if frame_id:
        header["frame_id"] = frame_id
    return ROSImage(
        image.shape[0],
        image.shape[1],
        encoding,
        image.shape[1] * channels * dtype.itemsize,
        int(sys.byteorder == "big"),
        pixels.tobytes(),
        header_value(header),
        payload,
    )


def decode_image(packet):
    if not isinstance(packet, ROSImage):
        raise ValidationError("Expected a portable ROSImage packet")
    packet.__post_init__()
    dtype, channels, color = ENCODINGS[packet.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if packet.is_bigendian else "<")
    shape = (
        (packet.height, packet.width) if channels == 1 else (packet.height, packet.width, channels)
    )
    strides = (
        (packet.step, dtype.itemsize)
        if channels == 1
        else (packet.step, channels * dtype.itemsize, dtype.itemsize)
    )
    view = np.ndarray(shape, dtype=dtype, buffer=packet.data, strides=strides)
    pixels = np.array(view, dtype=dtype.newbyteorder("="), order="C", copy=True)
    payload = copy.deepcopy(packet.metadata)
    payload["ros2"] = {
        **payload.get("ros2", {}),
        **packet.describe(),
        "message_type": "sensor_msgs/msg/Image",
    }
    return ImageArray(pixels, color, payload)
