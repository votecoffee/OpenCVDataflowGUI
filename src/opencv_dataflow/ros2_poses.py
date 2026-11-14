"""Strict ROS stamped poses and transforms using parent-from-child conventions."""

import copy

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import pose_record, rigid
from .parameter_validation import NumericRange, ValidationError
from .ros2_support.wire import header_value
from .visual_style import INDUSTRIAL

KINDS = ("transform_stamped", "pose_stamped")


def frame_name(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 1024
        or value.startswith("/")
        or "\0" in value
        or any(c.isspace() for c in value)
    ):
        raise ValidationError("Frame names must be nonempty, without whitespace or a leading slash")
    return value


def vector(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValidationError("ROS vector fields must match " + "/".join(names))
    if any(type(value[k]) not in (int, float) for k in names):
        raise ValidationError("ROS vectors require finite numeric values")
    result = np.array([value[k] for k in names], np.float64)
    if not np.isfinite(result).all():
        raise ValidationError("ROS vectors require finite numeric values")
    return result


def quaternion_matrix(q, policy="require_unit", tolerance=1e-6):
    if type(tolerance) not in (int, float) or not 1e-12 <= tolerance <= 0.01:
        raise ValidationError("Quaternion tolerance must be numeric in [1e-12,0.01]")
    length = float(np.linalg.norm(q))
    if not np.isfinite(length) or length < 1e-12:
        raise ValidationError("ROS quaternion must have a finite nonzero norm")
    if policy not in ("require_unit", "normalize"):
        raise ValidationError("Choose require_unit or normalize quaternion handling")
    if policy == "require_unit" and abs(length - 1) > tolerance:
        raise ValidationError(
            "ROS quaternion is not unit length; opt in to normalization explicitly"
        )
    x, y, z, w = q / length
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    ), length


def matrix_quaternion(rotation):
    # Stable near half turns, using the largest of the four quaternion components.
    m = rotation
    candidates = np.array(
        [
            1 + m[0, 0] - m[1, 1] - m[2, 2],
            1 - m[0, 0] + m[1, 1] - m[2, 2],
            1 - m[0, 0] - m[1, 1] + m[2, 2],
            1 + np.trace(m),
        ]
    )
    index = int(np.argmax(candidates))
    q = np.zeros(4)
    q[index] = 0.5 * np.sqrt(max(0.0, candidates[index]))
    divisor = 4 * q[index]
    if index == 3:
        q[:3] = [
            (m[2, 1] - m[1, 2]) / divisor,
            (m[0, 2] - m[2, 0]) / divisor,
            (m[1, 0] - m[0, 1]) / divisor,
        ]
    else:
        j, k = (index + 1) % 3, (index + 2) % 3
        q[j] = (m[j, index] + m[index, j]) / divisor
        q[k] = (m[k, index] + m[index, k]) / divisor
        q[3] = (m[k, j] - m[j, k]) / divisor
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def pose_from_ros(
    message,
    *,
    message_kind="transform_stamped",
    child_frame="object",
    expected_parent="",
    expected_child="",
    invert=False,
    units="m",
    quaternion_policy="require_unit",
    quaternion_tolerance=1e-6,
):
    if message_kind not in KINDS or units not in ("m", "mm") or type(invert) is not bool:
        raise ValidationError(
            "Choose a supported stamped message, physical units and inversion flag"
        )
    key = "transform" if message_kind == "transform_stamped" else "pose"
    fields = {"header", key} | (
        {"child_frame_id"} if message_kind == "transform_stamped" else set()
    )
    if not isinstance(message, dict) or set(message) != fields:
        raise ValidationError("Message fields do not match the chosen stamped pose/transform")
    header = header_value(message["header"])
    parent = frame_name(header["frame_id"])
    child = frame_name(
        message["child_frame_id"] if message_kind == "transform_stamped" else child_frame
    )
    if expected_parent and parent != expected_parent or expected_child and child != expected_child:
        raise ValidationError("ROS parent/child frame does not match the configured expectation")
    body = message[key]
    position, orientation = (
        ("translation", "rotation") if key == "transform" else ("position", "orientation")
    )
    if not isinstance(body, dict) or set(body) != {position, orientation}:
        raise ValidationError("ROS pose requires position/translation and orientation/rotation")
    translation = vector(body[position], ("x", "y", "z"))
    rotation, norm = quaternion_matrix(
        vector(body[orientation], ("x", "y", "z", "w")), quaternion_policy, quaternion_tolerance
    )
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    if parent == child and not np.allclose(matrix, np.eye(4), atol=1e-9, rtol=0):
        raise ValidationError("A frame cannot have a nonidentity transform to itself")
    if invert:
        matrix, child, parent = np.linalg.inv(matrix), parent, child
    if units == "mm":
        matrix[:3, 3] *= 1000
    pose_header = {**copy.deepcopy(header), "frame_id": parent}
    ros = {
        "header": pose_header,
        "source_header": header,
        "message_kind": message_kind,
        "convention": "parent_from_child",
    }
    result = pose_record(matrix, child, parent, units, ros2=ros)
    return result, {
        "parent_frame": parent,
        "child_frame": child,
        "units": units,
        "quaternion_norm": norm,
        "inverted": invert,
        "header": header,
        "quaternion_order": "xyzw",
        "convention": "parent_from_child",
    }


def pose_to_ros(pose, *, message_kind="transform_stamped", invert=False, stamp_mode="preserve"):
    if (
        message_kind not in KINDS
        or type(invert) is not bool
        or stamp_mode not in ("preserve", "zero")
    ):
        raise ValidationError("Choose a stamped message kind, inversion and timestamp mode")
    if not isinstance(pose, dict) or pose.get("units") not in ("m", "mm"):
        raise ValidationError("ROS pose encoding needs physical m/mm units")
    matrix = rigid(pose).copy()
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0):
        raise ValidationError("ROS pose rotation must be orthonormal")
    parent, child = frame_name(pose.get("target_frame")), frame_name(pose.get("source_frame"))
    if parent == child and not np.allclose(matrix, np.eye(4), atol=1e-9, rtol=0):
        raise ValidationError("A frame cannot have a nonidentity transform to itself")
    if pose["units"] == "mm":
        matrix[:3, 3] *= 0.001
    if invert:
        matrix, child, parent = np.linalg.inv(matrix), parent, child
    header = copy.deepcopy(
        pose.get("ros2", {}).get("header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": parent})
    )
    header["frame_id"] = parent
    if stamp_mode == "zero":
        header["stamp"] = {"sec": 0, "nanosec": 0}
    translation = dict(zip(("x", "y", "z"), matrix[:3, 3].tolist()))
    quaternion = dict(zip(("x", "y", "z", "w"), matrix_quaternion(matrix[:3, :3]).tolist()))
    if message_kind == "transform_stamped":
        return {
            "header": header_value(header),
            "child_frame_id": child,
            "transform": {"translation": translation, "rotation": quaternion},
        }
    return {
        "header": header_value(header),
        "pose": {"position": translation, "orientation": quaternion},
    }


class DecodeROSPose(Block):
    type_id, title = "industrial.ros2_pose_decode", "Decode ROS 2 pose / transform"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("message", "data"),)
    outputs = (
        Port("pose", "data", visual_type="pose"),
        Port("message", "data"),
        Port("details", "data"),
    )
    defaults = {
        "message_kind": "transform_stamped",
        "child_frame": "object",
        "expected_parent": "",
        "expected_child": "",
        "invert": False,
        "units": "m",
        "quaternion_policy": "require_unit",
        "quaternion_tolerance": 1e-6,
    }
    parameter_choices = {
        "message_kind": KINDS,
        "units": ("m", "mm"),
        "quaternion_policy": ("require_unit", "normalize"),
    }
    parameter_ranges = {"quaternion_tolerance": NumericRange(1e-12, 0.01)}
    parameter_help = {
        "message_kind": (
            "TransformStamped includes child_frame_id; PoseStamped needs the configured "
            "child frame."
        ),
        "child_frame": (
            "PoseStamped only: explicit object/tool frame whose pose is expressed in "
            "header.frame_id."
        ),
        "expected_parent": "Optional required header.frame_id, checked before inversion.",
        "expected_child": "Optional required child frame, checked before inversion.",
        "invert": "Invert the rigid transform and swap source/target frames.",
        "units": "ROS translations are meters; optionally convert the app pose to millimeters.",
        "quaternion_policy": (
            "Reject nonunit quaternions or explicitly normalize finite nonzero values."
        ),
        "quaternion_tolerance": "Allowed quaternion norm error in Require unit mode.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["invert"]) is not bool or type(p["quaternion_tolerance"]) not in (int, float):
            raise ValidationError("Use a boolean inversion and numeric quaternion tolerance")
        frame_name(p["child_frame"])
        for key in ("expected_parent", "expected_child"):
            if p[key] != "":
                frame_name(p[key])

    def process(self, inputs):
        pose, details = pose_from_ros(inputs["message"], **self.parameters)
        return Result(
            {"pose": pose, "message": copy.deepcopy(inputs["message"]), "details": details}
        )


class EncodeROSPose(Block):
    type_id, title = "industrial.ros2_pose_encode", "Encode ROS 2 pose / transform"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("pose", "data", visual_type="pose"),)
    outputs = (Port("message", "data"),)
    defaults = {"message_kind": "transform_stamped", "invert": False, "stamp_mode": "preserve"}
    parameter_choices = {"message_kind": KINDS, "stamp_mode": ("preserve", "zero")}
    parameter_help = {
        "message_kind": DecodeROSPose.parameter_help["message_kind"],
        "invert": DecodeROSPose.parameter_help["invert"],
        "stamp_mode": (
            "Keep a ROS acquisition stamp (otherwise zero) or clear it. Header frame is "
            "the output parent."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["invert"]) is not bool:
            raise ValidationError("Invert must be boolean", "invert")

    def process(self, inputs):
        return Result({"message": pose_to_ros(inputs["pose"], **self.parameters)})


ROS2_POSE_BLOCKS = (DecodeROSPose, EncodeROSPose)
