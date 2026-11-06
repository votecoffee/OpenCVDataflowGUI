"""Fisheye/omnidirectional rectification, ArUco pose and structured-light wrappers."""

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .fiducial_detection import DICTIONARIES, MARKER_CHOICES, MARKER_OPTIONS, detector_parameters
from .geometry3d_common import (
    Tool3D,
    camera_for_image,
    camera_record,
    matrix,
    pose_matrix,
    pose_record,
    require_pinhole,
)
from .images import ImageArray, image_format
from .payloads import NO_VALUE, FlowGroup, ValuePacket, frame, metadata
from .processing import finite, gray_image, integer


class RectifyCamera3D(Tool3D):
    type_id, title, subtype = (
        "3d.rectify_camera",
        "Rectify pinhole / fisheye / omnidir",
        "Camera models",
    )
    inputs = (Port("image"), Port("camera", "data"))
    outputs = (Port("image"), Port("camera", "data"))
    defaults = {
        "width": 0,
        "height": 0,
        "focal_scale": 1.0,
        "rotation_vector": [0.0, 0.0, 0.0],
        "interpolation": "linear",
        "border_value": 0.0,
    }
    parameter_choices = {"interpolation": ("nearest", "linear", "cubic", "lanczos4")}

    def process(self, inputs):
        model, source, p = inputs["camera"], inputs["image"], self.parameters
        k, d, size = camera_for_image(model, source)
        output_size = (
            integer(p["width"], "Width", 0, 4096) or size[0],
            integer(p["height"], "Height", 0, 4096) or size[1],
        )
        scale = finite(p["focal_scale"], "Focal scale", 0.0001)
        new_k = k.copy()
        new_k[0] *= output_size[0] / size[0]
        new_k[1] *= output_size[1] / size[1]
        new_k[0, 0] *= scale
        new_k[1, 1] *= scale
        rotation = cv2.Rodrigues(matrix([p["rotation_vector"]], (1, 3), "Rotation vector"))[0]
        lens = model.get("lens", "pinhole")
        if lens == "fisheye":
            mx, my = cv2.fisheye.initUndistortRectifyMap(
                k, d, rotation, new_k, output_size, cv2.CV_32FC1
            )
        elif lens == "omnidir":
            mx, my = cv2.omnidir.initUndistortRectifyMap(
                k,
                d,
                np.array([model["xi"]]),
                rotation,
                new_k,
                output_size,
                cv2.CV_32FC1,
                cv2.omnidir.RECTIFY_PERSPECTIVE,
            )
        else:
            mx, my = cv2.initUndistortRectifyMap(k, d, rotation, new_k, output_size, cv2.CV_32FC1)
        image = cv2.remap(
            np.asarray(source),
            mx,
            my,
            {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "lanczos4": cv2.INTER_LANCZOS4,
            }[p["interpolation"]],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=finite(p["border_value"], "Border value"),
        )
        output = ImageArray(image, image_format(source))
        payload = {**metadata(source), "spatial": frame(output_size, f"{self.id}:rectified")}
        payload.pop("pyramid", None)
        cam = camera_record(
            new_k,
            [],
            output_size,
            model.get("units", "m"),
            f"{model.get('frame', 'camera')}:rectified",
            rectification_rotation=rotation.tolist(),
            source_frame=model.get("frame", "camera"),
        )
        return Result({"image": output, "camera": cam}, payloads={"image": payload})


class ArucoPose3D(Tool3D):
    type_id, title, subtype = "3d.aruco_pose", "ArUco 3D marker poses", "Fiducials"
    inputs = (Port("image"), Port("camera", "data"))
    outputs = (
        Port("poses", "data"),
        Port("pose", "data"),
        Port("count", "number"),
        Port("overlay"),
    )
    defaults = {
        "dictionary": "4x4_50",
        "marker_length": 0.04,
        "minimum_markers": 1,
        **MARKER_OPTIONS,
        "axis_length": 0.02,
        "max_reprojection_error": 3.0,
    }
    parameter_choices = MARKER_CHOICES

    def process(self, inputs):
        model, p = inputs["camera"], self.parameters
        k, d, _ = require_pinhole(model)
        camera_for_image(model, inputs["image"])
        gray = gray_image(inputs["image"], (np.uint8,))
        parameters = detector_parameters(p)
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]])
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
        overlay = overlay_image(inputs["image"])
        length = finite(p["marker_length"], "Marker length", 1e-9)
        axis = finite(p["axis_length"], "Axis length", 1e-9)
        max_error = finite(p["max_reprojection_error"], "Max reprojection error", 0)
        obj = np.array(
            [
                [-length / 2, length / 2, 0],
                [length / 2, length / 2, 0],
                [length / 2, -length / 2, 0],
                [-length / 2, -length / 2, 0],
            ]
        )
        poses = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            for corner, marker_id in zip(corners, ids.reshape(-1)):
                ok, rv, tv = cv2.solvePnP(
                    obj, corner.reshape(4, 2), k, d, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue
                error = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    cv2.projectPoints(obj, rv, tv, k, d)[0].reshape(4, 2)
                                    - corner.reshape(4, 2)
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                if error > max_error or tv[2, 0] <= 0:
                    continue
                poses.append(
                    pose_record(
                        pose_matrix(cv2.Rodrigues(rv)[0], tv),
                        f"aruco:{marker_id}",
                        model.get("frame", "camera"),
                        model.get("units", "m"),
                        marker_id=int(marker_id),
                        rms_pixels=error,
                    )
                )
                cv2.drawFrameAxes(overlay, k, d, rv, tv, axis)
        passed = len(poses) >= integer(p["minimum_markers"], "Minimum markers", 1, 10000)
        return Result(
            {
                "poses": poses,
                "pose": poses[0] if poses else NO_VALUE,
                "count": len(poses),
                "overlay": overlay,
            },
            passed=passed,
            fail_reasons=[] if passed else ["aruco_markers"],
        )


def gray_code(parameters):
    module = getattr(cv2, "structured_light", None)
    if module is None:
        raise RuntimeError("This OpenCV build does not expose structured light")
    width = integer(parameters["width"], "Projector width", 2, 4096)
    height = integer(parameters["height"], "Projector height", 2, 4096)
    pattern = module.GrayCodePattern.create(width, height)
    pattern.setWhiteThreshold(
        integer(parameters.get("white_threshold", 5), "White threshold", 0, 255)
    )
    pattern.setBlackThreshold(
        integer(parameters.get("black_threshold", 40), "Black threshold", 0, 255)
    )
    return pattern, width, height


class GrayCodePatterns(Tool3D):
    type_id, title, subtype = "3d.graycode", "Generate Gray-code patterns", "Structured light"
    outputs = (Port("patterns", "any"), Port("white"), Port("black"), Port("count", "number"))
    defaults = {"width": 640, "height": 480}

    def process(self, inputs):
        pattern, width, height = gray_code(self.parameters)
        count = pattern.getNumberOfPatternImages()
        if count * width * height > 128_000_000:
            raise ValueError("Pattern stack exceeds 128 million pixels; reduce projector size")
        ok, images = pattern.generate()
        if not ok:
            raise ValueError("OpenCV could not generate Gray-code patterns")
        packets = tuple(
            ValuePacket(image, {"pattern_index": i, "projector_size": [width, height]})
            for i, image in enumerate(images)
        )
        return Result(
            {
                "patterns": FlowGroup(packets, tuple(f"pattern_{i}" for i in range(len(packets)))),
                "white": np.full((height, width), 255, np.uint8),
                "black": np.zeros((height, width), np.uint8),
                "count": len(images),
            }
        )


class DecodeGrayCode(Tool3D):
    type_id, title, subtype = (
        "3d.graycode_decode",
        "Decode Gray-code correspondences",
        "Structured light",
    )
    inputs = (
        Port("patterns", "any"),
        Port("points", "data", required=False),
        Port("white", required=False),
        Port("black", required=False),
    )
    outputs = (
        Port("camera_points", "data"),
        Port("projector_points", "data"),
        Port("valid", "data"),
    )
    defaults = {
        "width": 640,
        "height": 480,
        "sample_step": 16,
        "white_threshold": 5,
        "black_threshold": 40,
    }

    def process(self, inputs):
        pattern, width, height = gray_code(self.parameters)
        group = inputs["patterns"]
        if (
            not isinstance(group, FlowGroup)
            or len(group.items) != pattern.getNumberOfPatternImages()
        ):
            raise ValueError("Connect a complete captured/generated Gray-code pattern group")
        images = [gray_image(packet.value, (np.uint8,)) for packet in group.items]
        if any(
            packet.payload.get("pattern_index", i) != i
            or packet.payload.get("projector_size", [width, height]) != [width, height]
            for i, packet in enumerate(group.items)
        ):
            raise ValueError("Gray-code pattern order/projector size metadata do not match")
        shape = images[0].shape
        if any(image.shape != shape for image in images):
            raise ValueError("Captured pattern image dimensions must match each other")
        points = inputs.get("points")
        if points is None:
            step = integer(self.parameters["sample_step"], "Sample step", 1, 1000)
            yy, xx = np.mgrid[0 : shape[0] : step, 0 : shape[1] : step]
            points = np.c_[xx.reshape(-1), yy.reshape(-1)].tolist()
        xy = np.asarray(points)
        if (
            xy.ndim != 2
            or xy.shape[1] != 2
            or len(xy) > 10000
            or not np.isfinite(xy).all()
            or np.any(xy != np.rint(xy))
            or np.any(xy < 0)
            or np.any(xy >= [shape[1], shape[0]])
        ):
            raise ValueError(
                "Decode points must be at most 10000 camera pixels inside the captures"
            )
        lit = np.ones(shape, bool)
        if ("white" in inputs) != ("black" in inputs):
            raise ValueError("Supply both white and black shadow captures")
        if "white" in inputs:
            white, black = (gray_image(inputs[k], (np.uint8,)) for k in ("white", "black"))
            if white.shape != shape or black.shape != shape:
                raise ValueError("Shadow captures must match pattern dimensions")
            lit = white.astype(int) - black.astype(int) > self.parameters["black_threshold"]
        projector, valid = [], []
        for x, y in xy.astype(int):
            error, pixel = pattern.getProjPixel(images, int(x), int(y))
            ok = not error and bool(lit[y, x]) and 0 <= pixel[0] < width and 0 <= pixel[1] < height
            projector.append(list(pixel) if ok else None)
            valid.append(ok)
        return Result(
            {"camera_points": xy.tolist(), "projector_points": projector, "valid": valid},
            passed=any(valid),
            message="Camera/projector correspondences; use calibrated projections for 3D",
        )


class Capabilities3D(Tool3D):
    type_id, title, subtype = "3d.capabilities", "OpenCV 3D capabilities", "Diagnostics"
    outputs = (Port("capabilities", "data"),)

    def process(self, inputs):
        from .stereo3d import rgbd_function

        rgbd = {}
        for name in ("depthTo3d", "registerDepth", "RgbdNormals_create"):
            try:
                rgbd_function(name)
                rgbd[name] = True
            except RuntimeError:
                rgbd[name] = False
        return Result(
            {
                "capabilities": {
                    "opencv": cv2.__version__,
                    "rgbd": rgbd,
                    "modules": {
                        name: hasattr(cv2, name)
                        for name in (
                            "fisheye",
                            "omnidir",
                            "ppf_match_3d",
                            "structured_light",
                            "stereo",
                            "sfm",
                            "viz",
                            "aruco",
                        )
                    },
                    "hand_eye": hasattr(cv2, "calibrateHandEye"),
                    "robot_world_hand_eye": hasattr(cv2, "calibrateRobotWorldHandEye"),
                }
            }
        )


VISION3D_MODULE_BLOCKS = (
    RectifyCamera3D,
    ArucoPose3D,
    GrayCodePatterns,
    DecodeGrayCode,
    Capabilities3D,
)
