"""Fit Mei-model omnidirectional camera intrinsics from planar target views."""

import cv2
import numpy as np

from .geometry3d_common import camera_record
from .parameter_validation import NumericRange, ValidationError
from .wide_angle_calibration import WideAngleCalibration

FLAGS = {
    "fix_focal_length": "CALIB_FIX_GAMMA",
    "fix_principal_point": "CALIB_FIX_CENTER",
    "fix_xi": "CALIB_FIX_XI",
    "fix_k1": "CALIB_FIX_K1",
    "fix_k2": "CALIB_FIX_K2",
    "fix_p1": "CALIB_FIX_P1",
    "fix_p2": "CALIB_FIX_P2",
}


class OmnidirectionalCalibration(WideAngleCalibration):
    type_id, title = "3d.omnidir_calibrate", "Calibrate omnidirectional camera"
    defaults = {
        **WideAngleCalibration.defaults,
        **dict.fromkeys(FLAGS, False),
        "minimum_views": 3,
        "view_policy": "require_all",
        "max_iterations": 300,
    }
    parameter_choices = {
        **WideAngleCalibration.parameter_choices,
        "view_policy": ("require_all", "allow_subset"),
    }
    parameter_ranges = {
        **WideAngleCalibration.parameter_ranges,
        "minimum_views": NumericRange(3, 64, whole=True),
    }
    parameter_help = {
        **WideAngleCalibration.parameter_help,
        "fix_focal_length": "Keep the focal lengths estimated by native initialization.",
        "fix_principal_point": "Keep the native initial principal point at the sensor center.",
        "fix_xi": "Keep the native initial mirror parameter xi.",
        **{f"fix_{term}": f"Hold coefficient {term} at zero." for term in ("k1", "k2", "p1", "p2")},
        "view_policy": "Require every view or accept those that pass initialization.",
        "minimum_views": "Smallest accepted number of fitted views, including after exclusions.",
        "epsilon": "Relative parameter-update threshold. Zero uses the iteration limit.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in FLAGS:
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox", key)
        if all(p[key] for key in FLAGS):
            raise ValidationError("Leave at least one intrinsic parameter free", "fix_xi")
        if type(p["minimum_views"]) is not int:
            raise ValidationError("Use an integer", "minimum_views")

    def process(self, inputs):
        p = self.parameters
        native = getattr(cv2, "omnidir", None)
        if native is None:
            raise ValidationError("This OpenCV build does not provide omnidirectional calibration")
        views = self.read_views(inputs)
        world_arrays, pixel_arrays, size, names, units = views
        flags = native.CALIB_FIX_SKEW | sum(
            getattr(native, flag) for key, flag in FLAGS.items() if p[key]
        )
        criteria = (
            cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if p["epsilon"] else 0),
            p["max_iterations"],
            p["epsilon"],
        )
        rms, k, xi_value, d, rvecs, tvecs, indices = cv2.omnidir.calibrate(
            world_arrays, pixel_arrays, size, None, None, None, flags, criteria
        )
        xi = float(np.asarray(xi_value).item())
        if not np.isfinite(rms) or rms < 0 or not np.isfinite(xi) or xi < 0:
            raise ValidationError("Omnidirectional solver returned an invalid RMS or xi")
        indices = np.asarray(indices).reshape(-1)
        if (
            indices.dtype.kind not in "iu"
            or len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(world_arrays))
            or len(np.unique(indices)) != len(indices)
            or len(rvecs) != len(tvecs)
            or len(indices) != len(rvecs)
        ):
            raise ValidationError("OpenCV returned invalid omnidirectional view indices")
        used = indices.tolist()
        profile = camera_record(
            k,
            d,
            size,
            units,
            p["camera_frame"],
            "omnidir",
            xi=xi,
            rms_pixels=float(rms),
            views=len(used),
        )

        def project(world, rvec, tvec):
            return native.projectPoints(world, rvec, tvec, k, xi, d)[0]

        reasons = []
        if len(used) < p["minimum_views"]:
            reasons.append(
                {
                    "code": "omnidir.insufficient_views",
                    "message": f"Kept {len(used)} views. {p['minimum_views']} are required.",
                }
            )
        if p["view_policy"] == "require_all" and len(used) != len(names):
            reasons.append(
                {
                    "code": "omnidir.excluded_views",
                    "message": f"Initialization excluded {len(names) - len(used)} target views.",
                }
            )
        result = self.fit_result(views, profile, rvecs, tvecs, project, used=used, reasons=reasons)
        result.outputs["fit_report"]["xi"] = xi
        return result
