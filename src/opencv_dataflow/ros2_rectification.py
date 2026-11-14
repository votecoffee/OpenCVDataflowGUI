"""Explicit CameraInfo intensity remapping with a new optical and pixel frame."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import camera, camera_record, pose_record, rigid
from .images import DepthArray, ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, scalar_value
from .payloads import ValuePacket, frame
from .recipes import unique_ref
from .ros2_calibration import camera_from_info, camera_to_info, numeric
from .ros2_camera_cache import calibration_revision, static_binding_header
from .ros2_context import check_current
from .ros2_poses import frame_name
from .ros2_support.wire import ENCODINGS, MAX_IMAGE_BYTES, header_value
from .tickets import ticket_key
from .visual_style import INDUSTRIAL

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
MAX_PIXELS = 4_194_304


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RectifyROSImage(Block):
    type_id, title = "industrial.ros2_rectify_image", "Rectify ROS 2 intensity image"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (
        Port("image"),
        Port("camera_info", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("camera", "data"),
        Port("camera_info", "data"),
        Port("pose", "data", visual_type="pose"),
        Port("valid_mask"),
        Port("map_x"),
        Port("map_y"),
        Port("info", "data"),
    )
    defaults = {
        "camera_info": {},
        "rectified_frame": "",
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0,
        "header_mode": "require",
        "scalar_semantics": "auto",
        "require_matching_ticket": False,
    }
    parameter_choices = {
        "interpolation": tuple(INTERPOLATIONS),
        "border": tuple(BORDERS),
        "header_mode": ("require", "zero_if_missing"),
        "scalar_semantics": ("auto", "intensity"),
    }
    parameter_ranges = {
        "width": NumericRange(0, 16384, whole=True),
        "height": NumericRange(0, 16384, whole=True),
    }
    parameter_help = {
        "camera_info": "Fallback complete CameraInfo JSON when the input is unwired.",
        "rectified_frame": (
            "Empty derives <source>_rectified. Nonidentity R needs a distinct optical frame."
        ),
        "width": (
            "Both dimensions zero keep source size; otherwise scale P intrinsics to this width."
        ),
        "height": (
            "Both dimensions must be zero or positive. Output is bounded to 4M pixels/16 MiB."
        ),
        "interpolation": "OpenCV remap sampling; nearest preserves discrete labels.",
        "border": "Sampling outside source pixels; this does not extend the validity mask.",
        "border_value": "One to four finite channel values, in the image's existing channel order.",
        "header_mode": (
            "Require an image header, or explicitly use zero time if absent. "
            "Calibration time is never substituted."
        ),
        "require_matching_ticket": (
            "Require image and wired calibration to have the same acquisition ticket."
        ),
        "scalar_semantics": (
            "Auto rejects ambiguous float/16UC1 planes. Intensity explicitly declares "
            "brightness or labels; physical DepthArray input is always rejected."
        ),
    }
    input_help = {
        "image": (
            "Raw intensity image on the calibrated ROS grid. "
            "Physical/axial depth is not an intensity image."
        ),
        "camera_info": (
            "Complete CameraInfo, or decoded/bound CameraInfo camera model. "
            "R/P select output geometry."
        ),
    }
    parallel_safe = preview_safe = True
    payload_reference = "image"
    opencv_functions = ("initUndistortRectifyMap", "fisheye.initUndistortRectifyMap", "remap")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "camera", "camera_info", "pose", "valid_mask", "info"]
        self.reset_runtime()

    def reset_runtime(self):
        self._maps = self._map_key = None

    def copy_runtime_from(self, other):
        self.reset_runtime()  # Display/execution snapshots never share writable map buffers.

    def close(self):
        self.reset_runtime()

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if (p["width"] == 0) != (p["height"] == 0):
            raise ValidationError(
                "Set both dimensions to zero, or both positive", "width", "height"
            )
        if any(type(p[k]) is not int for k in ("width", "height")):
            raise ValidationError("Use integer dimensions", "width", "height")
        if p["rectified_frame"] != "":
            frame_name(p["rectified_frame"])
        if type(p["require_matching_ticket"]) is not bool:
            raise ValidationError("Use a boolean ticket setting", "require_matching_ticket")
        scalar_value(p["border_value"])

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        payload = copy.deepcopy(self.input_payloads["image"])
        calibration_payload = self.input_payloads.get("camera_info", {})
        value = inputs.get("camera_info", p["camera_info"])
        if isinstance(value, dict) and value.get("model") == "camera_model_v1":
            if value.get("ros2", {}).get("space") != "raw":
                raise ValidationError("Connect raw CameraInfo; this image is not already rectified")
            try:
                calibration_revision(value)
            except ValueError as exc:
                raise ValidationError(str(exc), "camera_info") from exc
            if "ros2_static_binding" in value:
                static_binding_header(value, calibration_payload)
                if payload.get("ticket") is None or ticket_key(
                    calibration_payload.get("ticket")
                ) != ticket_key(payload["ticket"]):
                    raise ValidationError("Bound calibration belongs to a different image ticket")
                if calibration_payload.get("spatial") != payload.get("spatial"):
                    raise ValidationError("Bound calibration belongs to a different pixel grid")
            value = value["ros2"]["camera_info"]
        original = copy.deepcopy(value)
        try:
            model, _ = camera_from_info(value, space="raw")
        except ValueError as exc:
            raise ValidationError(str(exc), "camera_info") from exc
        k, d, size = camera(model)
        raw_frame = frame_name(model["frame"])
        roi = value["roi"]
        if (
            value["binning_x"] not in (0, 1)
            or value["binning_y"] not in (0, 1)
            or roi["do_rectify"]
            or [roi["x_offset"], roi["y_offset"]] != [0, 0]
            or [roi["width"], roi["height"]] not in ([0, 0], list(size))
        ):
            raise ValidationError("This rectifier requires a full, unbinned CameraInfo grid")
        if isinstance(image, DepthArray):
            raise ValidationError(
                "Axial depth needs 3D reprojection; do not rectify it as intensity"
            )
        if (
            image.ndim not in (2, 3)
            or image.shape[:2] != (size[1], size[0])
            or image.ndim == 3
            and image.shape[2] not in (3, 4)
        ):
            raise ValidationError("Raw intensity dimensions must match CameraInfo")
        color = image_format(image)
        channels = 1 if image.ndim == 2 else image.shape[2]
        encoding = next(
            (
                name
                for name, (dtype, count, fmt) in ENCODINGS.items()
                if dtype == image.dtype.name and count == channels and fmt == color
            ),
            None,
        )
        if encoding is None or image.dtype == np.int32:
            raise ValidationError(
                "Use supported ROS gray or RGB/BGR(A) intensity samples for remap"
            )
        if not np.allclose(payload["spatial"]["to_root"], np.eye(3), atol=1e-9, rtol=0):
            raise ValidationError("Image pixels no longer match the original calibrated grid")
        image_ros = payload.get("ros2", {})
        if (
            p["scalar_semantics"] == "auto"
            and channels == 1
            and (image.dtype in (np.float32, np.float64) or image_ros.get("encoding") == "16UC1")
        ):
            raise ValidationError(
                "Declare Intensity for ambiguous scalar samples; axial depth needs reprojection",
                "scalar_semantics",
            )
        if ("width" in image_ros or "height" in image_ros) and [
            image_ros.get("width"),
            image_ros.get("height"),
        ] != list(size):
            raise ValidationError("ROS image dimensions no longer match the calibrated grid")
        if "header" in image_ros:
            try:
                header = header_value(image_ros["header"])
            except ValueError as exc:
                raise ValidationError("Invalid ROS image acquisition header") from exc
            if header["frame_id"] != raw_frame:
                raise ValidationError("Image and CameraInfo optical frames differ")
        elif p["header_mode"] == "zero_if_missing":
            header = {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": raw_frame}
        else:
            raise ValidationError(
                "Require an image header or explicitly choose Zero if missing", "header_mode"
            )
        if p["require_matching_ticket"] and (
            payload.get("ticket") is None
            or calibration_payload.get("ticket") is None
            or ticket_key(payload["ticket"]) != ticket_key(calibration_payload["ticket"])
        ):
            raise ValidationError("Image and calibration acquisition tickets differ")
        if "acquisition" in calibration_payload:
            context = payload.get("acquisition", {}).get("context")
            check_current(image, payload, context)
            check_current(value, calibration_payload, context)
        try:
            reference = unique_ref(
                [ValuePacket(image, payload), ValuePacket(value, calibration_payload)]
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if reference is not None:
            payload["recipe_ref"] = copy.deepcopy(reference)
        rotation = numeric(value["r"], 9, "R").reshape(3, 3)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        try:
            rigid(matrix)
        except ValueError as exc:
            raise ValidationError(
                "CameraInfo R must be a proper orthonormal rotation", "camera_info"
            ) from exc
        projection = numeric(value["p"], 12, "P").reshape(3, 4)
        if not np.allclose(projection[:, 3], 0, atol=1e-10, rtol=0):
            raise ValidationError(
                "Stereo projection translation requires explicit baseline handling"
            )
        target_frame = frame_name(p["rectified_frame"] or raw_frame + "_rectified")
        if target_frame == raw_frame and not np.allclose(rotation, np.eye(3), atol=1e-9, rtol=0):
            raise ValidationError(
                "Nonidentity R requires a different rectified optical frame", "rectified_frame"
            )
        out_size = (p["width"], p["height"]) if p["width"] else size
        if (
            np.prod(out_size) > MAX_PIXELS
            or np.prod(out_size) * channels * image.dtype.itemsize > MAX_IMAGE_BYTES
        ):
            raise ValidationError("Rectified output exceeds 4M pixels or 16 MiB", "width", "height")
        new_k = projection[:, :3].copy()
        new_k[:2] *= (np.array(out_size) / size)[:, None]
        try:
            target = camera_record(new_k, [], out_size, frame=target_frame)
        except ValueError as exc:
            raise ValidationError(str(exc), "camera_info") from exc
        key = signature(
            {
                "K": k.tolist(),
                "D": d.tolist(),
                "R": rotation.tolist(),
                "P": new_k.tolist(),
                "size": out_size,
                "lens": model["lens"],
            }
        )
        if self._map_key != key:
            build = (
                cv2.fisheye.initUndistortRectifyMap
                if model["lens"] == "fisheye"
                else cv2.initUndistortRectifyMap
            )
            mx, my = build(k, d, rotation, new_k, out_size, cv2.CV_32FC1)
            if not np.isfinite(mx).all() or not np.isfinite(my).all():
                raise ValidationError("Rectification produced nonfinite map coordinates")
            yy, xx = np.indices((out_size[1], out_size[0]), dtype=np.float64)
            ray_z = (np.linalg.inv(rotation) @ np.linalg.inv(new_k))[2]
            front = ray_z[0] * xx + ray_z[1] * yy + ray_z[2] > 1e-12
            epsilon = 1e-4  # Float map roundoff at an otherwise exact image boundary.
            valid = (
                front
                & (mx >= -epsilon)
                & (my >= -epsilon)
                & (mx <= size[0] - 1 + epsilon)
                & (my <= size[1] - 1 + epsilon)
            )
            mx[~front], my[~front] = -1, -1
            self._maps, self._map_key = (mx, my, front, valid.astype(np.uint8) * 255), key
        mx, my, front, valid = self._maps
        border = scalar_value(p["border_value"])
        output = cv2.remap(
            np.asarray(image).astype(image.dtype.newbyteorder("="), copy=False),
            mx,
            my,
            INTERPOLATIONS[p["interpolation"]],
            borderMode=BORDERS[p["border"]],
            borderValue=border,
        )
        fill = np.asarray(border[0] if channels == 1 else border[:channels])
        if np.issubdtype(image.dtype, np.integer):
            fill = np.rint(np.clip(fill, np.iinfo(image.dtype).min, np.iinfo(image.dtype).max))
        output[~front] = fill.astype(image.dtype)
        output_header = {**copy.deepcopy(header), "frame_id": target_frame}
        target["ros2"] = {"header": output_header}
        message = camera_to_info(target)
        target, _ = camera_from_info(message)
        pose = pose_record(matrix, raw_frame, target_frame, "m", ros2={"header": output_header})
        output_spatial = frame(
            list(out_size),
            "ros_rectification:"
            + signature({"source": payload["spatial"], "map": key, "target_frame": target_frame}),
        )
        info = {
            "model": "ros_image_rectification_v1",
            "id": key,
            "source_header": header,
            "output_header": output_header,
            "source_camera_info": original,
            "source_calibration_binding": copy.deepcopy(
                calibration_payload.get("ros2_static_binding")
            ),
            "calibration_ticket": copy.deepcopy(calibration_payload.get("ticket")),
            "source_size": list(size),
            "output_size": list(out_size),
            "valid_pixels": int(np.count_nonzero(valid)),
            "valid_fraction": float(np.count_nonzero(valid) / valid.size),
            "validity": "source_pixel_center_in_front_and_inside_grid_not_full_kernel_support",
            "image_semantics": "intensity_not_axial_depth",
            "pose_convention": "rectified_from_raw",
            "source_spatial": copy.deepcopy(payload["spatial"]),
        }
        for field in (
            "pyramid",
            "ros2_static_binding",
            "ros2_rgbd",
            "coordinate_remap",
            "lens_remap_map",
        ):
            payload.pop(field, None)
        payload.update(
            spatial=output_spatial,
            ros2_rectification=info,
            ros2={
                "header": output_header,
                "encoding": encoding,
                "width": out_size[0],
                "height": out_size[1],
                "message_type": "sensor_msgs/msg/Image",
            },
        )
        map_info = {
            "model": "image_coordinate_map_v1",
            "id": key,
            "source_size": list(size),
            "output_size": list(out_size),
            "source_spatial": info["source_spatial"],
            "output_spatial": output_spatial,
        }
        payloads = {
            name: copy.deepcopy(payload)
            for name in ("image", "camera", "camera_info", "pose", "valid_mask", "info")
        }
        payloads["valid_mask"]["ros2"]["encoding"] = "mono8"
        for name in ("camera", "camera_info"):
            payloads[name]["ros2"] = {
                **copy.deepcopy(target["ros2"]),
                "message_type": "sensor_msgs/msg/CameraInfo",
            }
        payloads["pose"]["ros2"] = {"header": copy.deepcopy(output_header)}
        for axis in ("x", "y"):
            payloads["map_" + axis] = {
                "spatial": output_spatial,
                "coordinate_remap": {**map_info, "axis": axis},
            }
        return Result(
            {
                "image": ImageArray(output, color),
                "camera": target,
                "camera_info": message,
                "pose": pose,
                "valid_mask": valid.copy(),
                "map_x": mx.copy(),
                "map_y": my.copy(),
                "info": info,
            },
            payloads=payloads,
            passed=bool(info["valid_pixels"]),
            score=info["valid_fraction"],
            fail_reasons=[] if info["valid_pixels"] else ["no_valid_rectified_pixels"],
        )
