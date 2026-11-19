"""Fit a reusable zero-skew fisheye camera from offline calibration targets."""

import cv2
import numpy as np

from .core import Port
from .geometry3d_common import camera, camera_record
from .parameter_validation import ValidationError
from .wide_angle_calibration import WideAngleCalibration

FLAGS = {
    "recompute_extrinsics": cv2.CALIB_RECOMPUTE_EXTRINSIC,
    "check_condition": cv2.CALIB_CHECK_COND,
    "fix_focal_length": cv2.CALIB_FIX_FOCAL_LENGTH,
    "fix_principal_point": cv2.CALIB_FIX_PRINCIPAL_POINT,
    **{f"fix_k{i}": getattr(cv2, f"CALIB_FIX_K{i}") for i in range(1, 5)},
}


class FisheyeCalibration(WideAngleCalibration):
    type_id, title = "3d.fisheye_calibrate", "Calibrate fisheye camera"
    inputs = (
        *WideAngleCalibration.inputs,
        Port("initial_camera", "data", required=False, required_when_connected=True),
    )
    defaults = {
        **WideAngleCalibration.defaults,
        **dict.fromkeys(FLAGS, False),
        "recompute_extrinsics": True,
        "check_condition": True,
    }
    parameter_help = {
        **WideAngleCalibration.parameter_help,
        "recompute_extrinsics": "Refit target-to-camera poses during intrinsic optimization.",
        "check_condition": "Have OpenCV reject ill-conditioned extrinsic updates.",
        "fix_focal_length": "Keep both focal lengths from the connected initial camera.",
        "fix_principal_point": "Keep the principal point from the connected initial camera.",
        **{f"fix_k{i}": f"Fix fisheye coefficient k{i} at zero." for i in range(1, 5)},
        "max_iterations": "Maximum iterations for the OpenCV solve.",
        "epsilon": (
            "Relative focal/principal-point change. An epsilon of zero or locking both uses "
            "the iteration limit."
        ),
    }
    input_help = {
        **WideAngleCalibration.input_help,
        "initial_camera": "Optional fisheye seed with four coefficients on the same image grid.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in FLAGS:
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox", key)
        if (
            p["fix_focal_length"]
            and p["fix_principal_point"]
            and all(p[f"fix_k{i}"] for i in range(1, 5))
        ):
            raise ValidationError("Leave at least one intrinsic parameter free", "fix_focal_length")

    def process(self, inputs):
        p = self.parameters
        views = self.read_views(inputs)
        world_arrays, pixel_arrays, size, _, units = views
        flags = cv2.CALIB_FIX_SKEW | sum(flag for key, flag in FLAGS.items() if p[key])
        initial = inputs.get("initial_camera")
        if initial is None:
            if p["fix_focal_length"] or p["fix_principal_point"]:
                raise ValidationError(
                    "Connect an initial camera to fix focal length or principal point"
                )
            k, d = np.eye(3, dtype=np.float64), np.zeros((4, 1), np.float64)
        else:
            k, d, initial_size = camera(initial)
            if initial.get("lens") != "fisheye" or initial_size != size:
                raise ValidationError("Initial camera must be fisheye on the same pixel grid")
            k, d = k.copy(), d.reshape(4, 1).copy()
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # Native epsilon compares only focal lengths and principal point. With
        # both fixed it would stop after one iteration, even if distortion is moving.
        use_epsilon = p["epsilon"] > 0 and not (p["fix_focal_length"] and p["fix_principal_point"])
        criteria = (
            cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if use_epsilon else 0),
            p["max_iterations"],
            p["epsilon"] if use_epsilon else 0.0,
        )
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            world_arrays, pixel_arrays, size, k, d, flags=flags, criteria=criteria
        )
        if not np.isfinite(rms) or rms < 0:
            raise ValidationError("Fisheye solver returned an invalid RMS")
        profile = camera_record(
            k,
            d,
            size,
            units,
            p["camera_frame"],
            "fisheye",
            rms_pixels=float(rms),
            views=len(world_arrays),
        )

        def project(world, rvec, tvec):
            return cv2.fisheye.projectPoints(world, rvec, tvec, k, d)[0]

        return self.fit_result(views, profile, rvecs, tvecs, project)
