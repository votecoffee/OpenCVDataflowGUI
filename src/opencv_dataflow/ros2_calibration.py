"""ROS CameraInfo conversion and strict, already-registered RGB-D group validation."""

import copy

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import camera, camera_record
from .images import DepthArray, ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, frame, metadata
from .ros2_support.wire import MAX_DATA_BYTES, header_value, json_bytes
from .tickets import ticket_key
from .visual_style import INDUSTRIAL

INFO_FIELDS = {
    "header",
    "height",
    "width",
    "distortion_model",
    "d",
    "k",
    "r",
    "p",
    "binning_x",
    "binning_y",
    "roi",
}
DISTORTION_MODELS = {
    "plumb_bob": ("pinhole", 5),
    "rational_polynomial": ("pinhole", 8),
    "equidistant": ("fisheye", 4),
    "": ("pinhole", 0),
}


def numeric(values, count, name):
    if (
        not isinstance(values, (list, tuple))
        or len(values) != count
        or any(type(v) not in (int, float) for v in values)
    ):
        raise ValidationError(f"CameraInfo {name} requires {count} numeric values")
    result = np.asarray(values, np.float64)
    if not np.isfinite(result).all():
        raise ValidationError(f"CameraInfo {name} must be finite")
    return result


def camera_from_info(value, *, space="raw"):
    if space not in ("raw", "rectified"):
        raise ValidationError("Choose raw or rectified CameraInfo image space")
    if not isinstance(value, dict) or set(value) != INFO_FIELDS:
        raise ValidationError("Connect complete sensor_msgs/msg/CameraInfo fields")
    json_bytes(value, MAX_DATA_BYTES)
    header = header_value(value["header"])
    if not header["frame_id"]:
        raise ValidationError("CameraInfo needs a named optical frame_id")
    for key in ("width", "height", "binning_x", "binning_y"):
        if (
            type(value[key]) is not int
            or not (1 if key in ("width", "height") else 0) <= value[key] <= 16384
        ):
            raise ValidationError(f"Invalid CameraInfo {key}")
    roi = value["roi"]
    if not isinstance(roi, dict) or set(roi) != {
        "x_offset",
        "y_offset",
        "width",
        "height",
        "do_rectify",
    }:
        raise ValidationError("CameraInfo ROI needs x/y offsets, width/height and do_rectify")
    if type(roi["do_rectify"]) is not bool or any(
        type(roi[k]) is not int or roi[k] < 0 for k in ("x_offset", "y_offset", "width", "height")
    ):
        raise ValidationError("Invalid CameraInfo ROI fields")
    x, y, width, height = (roi[k] for k in ("x_offset", "y_offset", "width", "height"))
    if (x, y, width, height) == (0, 0, 0, 0):
        width, height = value["width"], value["height"]
    if not width or not height or x + width > value["width"] or y + height > value["height"]:
        raise ValidationError("CameraInfo ROI must fit the full calibrated image")
    bx, by = max(1, value["binning_x"]), max(1, value["binning_y"])
    size = (width // bx, height // by)
    if not all(size):
        raise ValidationError("CameraInfo binning leaves no complete output pixels")
    k = numeric(value["k"], 9, "K").reshape(3, 3)
    r = numeric(value["r"], 9, "R").reshape(3, 3)
    p = numeric(value["p"], 12, "P").reshape(3, 4)
    if k[0, 0] == 0:
        raise ValidationError("CameraInfo is uncalibrated (K[0] is zero)")
    model = value["distortion_model"]
    if not isinstance(model, str) or model not in DISTORTION_MODELS:
        raise ValidationError("Unsupported CameraInfo distortion model")
    lens, count = DISTORTION_MODELS[model]
    empty_d = isinstance(value["d"], (list, tuple)) and not value["d"]
    d = np.zeros(count) if empty_d else numeric(value["d"], count, "D")
    # Validate the original raw calibration even when selecting rectified intrinsics.
    camera_record(k, d, (value["width"], value["height"]), frame=header["frame_id"], lens=lens)
    if space == "rectified":
        if roi["do_rectify"]:
            raise ValidationError(
                "Rectified ROI recomputation is not supported; supply an already rectified ROI"
            )
        if not np.allclose(r, np.eye(3), atol=1e-8, rtol=0):
            raise ValidationError(
                "Nonidentity rectification changes the optical frame; "
                "use explicit stereo/TF processing"
            )
        if not np.allclose(p[:, 3], 0, atol=1e-10, rtol=0):
            raise ValidationError("Stereo P translation requires explicit baseline/frame handling")
        k, d, lens = p[:, :3].copy(), np.empty(0), "pinhole"
    adjusted = k.copy()
    adjusted[0] /= bx
    adjusted[1] /= by
    adjusted[0, 2] -= x / bx
    adjusted[1, 2] -= y / by
    transform = [[bx, 0, x], [0, by, y], [0, 0, 1]]
    ros = {
        "header": header,
        "message_type": "sensor_msgs/msg/CameraInfo",
        "space": space,
        "camera_info": copy.deepcopy(value),
        "output_to_full": transform,
    }
    result = camera_record(adjusted, d, size, frame=header["frame_id"], lens=lens, ros2=ros)
    details = {
        "space": space,
        "full_size": [value["width"], value["height"]],
        "image_size": list(size),
        "binning": [bx, by],
        "roi": [x, y, width, height],
        "discarded_partial_bins": [width % bx, height % by],
        "output_to_full": transform,
        "header": header,
        "distortion_model": model,
    }
    return result, details


def camera_to_info(model, *, stamp_mode="preserve"):
    k, d, size = camera(model)
    if not isinstance(model.get("frame"), str) or not model["frame"]:
        raise ValidationError("Name the camera's optical frame before ROS encoding")
    if stamp_mode not in ("preserve", "zero"):
        raise ValidationError("Choose preserve or zero CameraInfo timestamp mode")
    lens = model.get("lens", "pinhole")
    if lens == "fisheye":
        distortion_model = "equidistant"
    elif lens == "pinhole" and len(d) in (0, 4, 5, 8):
        distortion_model = "rational_polynomial" if len(d) == 8 else "plumb_bob"
        if len(d) == 0:
            d = np.zeros(5)
        elif len(d) == 4:
            d = np.r_[d, 0.0]
    else:
        raise ValidationError(
            "ROS CameraInfo encoding supports pinhole 0/4/5/8 or equidistant 4 coefficients"
        )
    header = copy.deepcopy(
        model.get("ros2", {}).get(
            "header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": model["frame"]}
        )
    )
    header["frame_id"] = model["frame"]
    if stamp_mode == "zero":
        header["stamp"] = {"sec": 0, "nanosec": 0}
    result = {
        "header": header_value(header),
        "width": size[0],
        "height": size[1],
        "distortion_model": distortion_model,
        "d": d.tolist(),
        "k": k.reshape(-1).tolist(),
        "r": np.eye(3).reshape(-1).tolist(),
        "p": np.c_[k, np.zeros(3)].reshape(-1).tolist(),
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "width": 0, "height": 0, "do_rectify": False},
    }
    camera_from_info(result)
    return result


class ROSCalibrationBlock(Block):
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"


class DecodeROSCamera(ROSCalibrationBlock):
    type_id, title = "industrial.ros2_camera_decode", "Decode ROS 2 CameraInfo"
    inputs = (Port("message", "data"),)
    outputs = (Port("camera", "data"), Port("message", "data"), Port("details", "data"))
    defaults = {"space": "raw"}
    parameter_choices = {"space": ("raw", "rectified")}
    parameter_help = {
        "space": (
            "Use raw K/D or rectified P intrinsics for an already rectified image. No "
            "image warping occurs."
        )
    }

    def process(self, inputs):
        model, details = camera_from_info(inputs["message"], **self.parameters)
        return Result(
            {"camera": model, "message": copy.deepcopy(inputs["message"]), "details": details}
        )


class EncodeROSCamera(ROSCalibrationBlock):
    type_id, title = "industrial.ros2_camera_encode", "Encode ROS 2 CameraInfo"
    inputs = (Port("camera", "data"),)
    outputs = (Port("message", "data"),)
    defaults = {"stamp_mode": "preserve"}
    parameter_choices = {"stamp_mode": ("preserve", "zero")}
    parameter_help = {
        "stamp_mode": (
            "Preserve a ROS acquisition stamp, otherwise zero. Builds canonical "
            "calibration for the current grid."
        )
    }

    def process(self, inputs):
        return Result({"message": camera_to_info(inputs["camera"], **self.parameters)})


class AssociateROSRGBD(ROSCalibrationBlock):
    type_id, title = "industrial.ros2_rgbd", "Validate registered ROS RGB-D"
    inputs = (Port("group", "any"),)
    outputs = (Port("color"), Port("depth", "depth"), Port("camera", "data"), Port("info", "data"))
    flow_primary = ("depth",)
    defaults = {
        "use_color": True,
        "color_label": "input_1",
        "depth_label": "input_2",
        "camera_label": "input_3",
        "calibration_timing": "per_acquisition",
        "max_skew_ms": 0.0,
        "allow_zero_stamps": False,
        "uint16_m_per_sample": 0.001,
        "float_depth_units": "m",
        "display_max_m": 5.0,
    }
    parameter_choices = {
        "float_depth_units": ("m", "mm"),
        "calibration_timing": ("per_acquisition", "bound_static"),
    }
    parameter_ranges = {
        "max_skew_ms": NumericRange(0, 1000),
        "uint16_m_per_sample": NumericRange(1e-9, 1000),
        "display_max_m": NumericRange(0.001, 1e6),
    }
    parameter_help = {
        "use_color": (
            "Require a registered color image; otherwise only depth and calibration are required."
        ),
        "color_label": "Color member label in the completed coordinator FlowGroup.",
        "depth_label": (
            "Depth member label; use input_1 for a two-input depth/calibration coordinator."
        ),
        "camera_label": "Decoded CameraInfo model member; use input_2 in depth-only groups.",
        "calibration_timing": (
            "Per acquisition checks the original CameraInfo stamp. Bound static requires "
            "an explicit Cache ROS 2 CameraInfo binding to this image ticket and grid."
        ),
        "max_skew_ms": (
            "Maximum header stamp spread. Zero requires exact integer nanosecond equality."
        ),
        "allow_zero_stamps": (
            "Explicit offline/unknown-time mode. Zero stamps never verify acquisition timing."
        ),
        "uint16_m_per_sample": (
            "Explicit meters per uint16 depth sample, e.g. 0.001 for millimeters."
        ),
        "float_depth_units": "Declared 32FC1/64FC1 depth units. Output is physical meters.",
        "display_max_m": (
            "Depth preview upper display bound in meters; does not clip measured values."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("max_skew_ms", "uint16_m_per_sample", "display_max_m"):
            if type(p[name]) not in (int, float):
                raise ValidationError(f"{name} must be numeric", name)
        for name in ("use_color", "allow_zero_stamps"):
            if type(p[name]) is not bool:
                raise ValidationError(f"{name} must be boolean", name)
        names = ["depth_label", "camera_label"] + (["color_label"] if p["use_color"] else [])
        if any(not isinstance(p[k], str) or not p[k] or len(p[k]) > 128 for k in names):
            raise ValidationError("Name each required group member", *names)
        if len({p[k] for k in names}) != len(names):
            raise ValidationError("RGB-D member labels must be distinct", *names)

    def process(self, inputs):
        p, group = self.parameters, inputs["group"]
        if not isinstance(group, FlowGroup) or len(set(group.labels)) != len(group.labels):
            raise ValidationError("Connect a completed FlowGroup with unique labels")
        members = dict(zip(group.labels, group.items))
        labels = [p["depth_label"], p["camera_label"]] + (
            [p["color_label"]] if p["use_color"] else []
        )
        if any(name not in members for name in labels):
            raise ValidationError("Incomplete RGB-D group; a required member is missing")
        packets = [members[k] for k in labels]
        tickets = [metadata(v).get("ticket") for v in packets]
        if any(t is None for t in tickets) or len({ticket_key(t) for t in tickets}) != 1:
            raise ValidationError(
                "RGB-D images and calibration require the same acquisition ticket"
            )
        model = copy.deepcopy(members[p["camera_label"]].value)
        k, d, size = camera(model)
        if model.get("lens", "pinhole") != "pinhole" or np.any(d):
            raise ValidationError(
                "Registered depth projection requires undistorted pinhole intrinsics"
            )
        headers = []
        images = {}
        for label in labels:
            packet = members[label]
            value = packet.value
            if label == p["camera_label"]:
                ros = model.get("ros2", {})
                if p["calibration_timing"] == "bound_static":
                    from .ros2_camera_cache import static_binding_header

                    ros = {"header": static_binding_header(model, metadata(packet))}
            else:
                if not isinstance(value, np.ndarray) or value.ndim not in (2, 3):
                    raise ValidationError("RGB-D members must contain images")
                ros = metadata(packet).get("ros2", {})
                if [value.shape[1], value.shape[0]] != list(size) or [
                    ros.get("width"),
                    ros.get("height"),
                ] != list(size):
                    raise ValidationError(
                        "RGB-D dimensions must match CameraInfo and the original ROS image grid"
                    )
                spatial = metadata(packet).get("spatial")
                if spatial and not np.allclose(spatial["to_root"], np.eye(3), atol=1e-9, rtol=0):
                    raise ValidationError(
                        "Transformed image pixels require corresponding updated ROS calibration"
                    )
                images[label] = value
            header = header_value(ros.get("header"))
            if header["frame_id"] != model["frame"]:
                raise ValidationError("RGB-D and CameraInfo optical frame_id values must match")
            headers.append(header)
        stamps = [h["stamp"]["sec"] * 1_000_000_000 + h["stamp"]["nanosec"] for h in headers]
        zero = any(s == 0 for s in stamps)
        if zero and not p["allow_zero_stamps"]:
            raise ValidationError("RGB-D acquisition timestamps must be nonzero")
        skew = max(stamps) - min(stamps)
        if skew > round(p["max_skew_ms"] * 1_000_000):
            raise ValidationError("RGB-D header timestamps exceed the configured tolerance")
        depth_packet = members[p["depth_label"]]
        image = images[p["depth_label"]]
        encoding = metadata(depth_packet).get("ros2", {}).get("encoding")
        if image.ndim != 2:
            raise ValidationError("ROS depth must be a scalar image")
        if image.dtype == np.uint16 and encoding in ("mono16", "16UC1"):
            scale = p["uint16_m_per_sample"]
        elif image.dtype in (np.float32, np.float64) and encoding in ("32FC1", "64FC1"):
            scale = 0.001 if p["float_depth_units"] == "mm" else 1.0
        else:
            raise ValidationError(
                "Choose uint16 or floating-point ROS depth with an explicit supported encoding"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(image, np.float32) * scale
        valid = np.isfinite(values) & (values > 0)
        values[~valid] = np.nan
        depth = DepthArray(values, "m", 0, p["display_max_m"])
        info = {
            "model": "ros2_rgbd_v1",
            "frame": model["frame"],
            "image_size": list(size),
            "header_spread_ns": skew,
            "timestamps_nonzero": not zero,
            "clock_domain_verified": False,
            "ticket": copy.deepcopy(tickets[0]),
            "valid_depth_samples": int(valid.sum()),
            "depth_units": "m",
            "registration": "caller_supplied_same_grid",
            "calibration_timing": p["calibration_timing"],
        }
        if p["calibration_timing"] == "bound_static":
            info["static_calibration"] = copy.deepcopy(model["ros2_static_binding"])
        base = metadata(depth_packet)
        base.setdefault("spatial", frame(list(size), f"{self.id}:registered_pixels"))
        base["ros2_rgbd"] = copy.deepcopy(info)
        depth.metadata = copy.deepcopy(base)
        color = NO_VALUE
        if p["use_color"]:
            original = members[p["color_label"]]
            payload = metadata(original)
            payload.update(spatial=copy.deepcopy(base["spatial"]), ros2_rgbd=copy.deepcopy(info))
            color = ImageArray(original.value.copy(), image_format(original.value), payload)
        return Result(
            {"color": color, "depth": depth, "camera": model, "info": info},
            passed=bool(valid.any()),
            fail_reasons=[] if valid.any() else ["no_valid_depth"],
        )


ROS2_CALIBRATION_BLOCKS = (DecodeROSCamera, EncodeROSCamera, AssociateROSRGBD)
