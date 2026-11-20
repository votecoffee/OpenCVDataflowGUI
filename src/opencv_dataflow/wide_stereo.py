"""Native fisheye and Mei stereo fitting and perspective rectification."""

import json
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera_record,
    points,
    pose_matrix,
    pose_record,
    stereo_cameras,
)
from .omnidir_calibration import FLAGS as OMNIDIR_FLAGS
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import finite

_VIEWS = ("object_views", "left_views", "right_views")
_MAX_OBSERVATIONS_BYTES = 2 * 1024 * 1024


def read_views(block, inputs, size, lens):
    p = block.parameters
    if p["path"]:
        if any(name in inputs for name in _VIEWS):
            raise ValidationError("Use an observations file or the three view connectors")
        path = Path(p["path"])
        if p["path_base"] == "project" and not path.is_absolute():
            if block.project_dir is None:
                raise ValidationError("Open or save the project before using relative view files")
            path = block.project_dir / path
        with path.open("rb") as stream:
            raw = stream.read(_MAX_OBSERVATIONS_BYTES + 1)
        if len(raw) > _MAX_OBSERVATIONS_BYTES:
            raise ValidationError("Stereo observations file exceeds 2 MiB")
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != set(_VIEWS):
            raise ValidationError("Expected object_views, left_views and right_views in the file")
    else:
        data = inputs
    groups = [data.get(name) for name in _VIEWS]
    if (
        any(not isinstance(group, list) for group in groups)
        or not 3 <= len(groups[0]) <= 32
        or len({len(g) for g in groups}) != 1
    ):
        raise ValidationError("Supply 3 to 32 matched object, left and right views")
    output = tuple([] for _ in _VIEWS)
    for index, view in enumerate(zip(*groups, strict=True), start=1):
        if any(not isinstance(rows, list) or not 6 <= len(rows) <= 256 for rows in view):
            raise ValidationError(f"View {index} needs 6 to 256 matched points")
        xyz, a, b = (points(rows, count, 6) for rows, count in zip(view, (3, 2, 2), strict=True))
        if len(xyz) != len(a) or len(xyz) != len(b):
            raise ValidationError(f"View {index} has different point counts")
        if len(xyz) != len(groups[0][0]):
            raise ValidationError("Native stereo fitting requires equal point counts across views")
        if lens == "omnidir" and not np.allclose(xyz[:, 2], 0, rtol=0, atol=1e-9):
            raise ValidationError("Omnidirectional fitting requires XY-plane targets")
        if any(
            np.linalg.matrix_rank(v - v[0]) < 2 or len(np.unique(v, axis=0)) != len(v)
            for v in (xyz, a, b)
        ):
            raise ValidationError(f"View {index} needs distinct, noncollinear points")
        if any(np.any(v < 0) or np.any(v >= size) for v in (a, b)):
            raise ValidationError(f"View {index} has observations outside the camera pixel grid")
        for dest, values in zip(output, (xyz, a, b), strict=True):
            dest.append(values[None])
    return output


def selected_indices(raw, count, rvecs, tvecs):
    indices = np.asarray(raw).reshape(-1)
    if (
        indices.dtype.kind not in "iu"
        or len(indices) == 0
        or len(indices) != len(rvecs)
        or len(indices) != len(tvecs)
        or np.any(indices < 0)
        or np.any(indices >= count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValidationError("OpenCV returned inconsistent stereo view indices")
    return indices.tolist()


def view_report(lens, views, indices, rvecs, tvecs, left, right, rotation, translation):
    objects, a, b = views
    report = []
    for index, lvec, lt in zip(indices, rvecs, tvecs, strict=True):
        lt = np.asarray(lt).reshape(3)
        lr = cv2.Rodrigues(lvec)[0]
        right_rvec = cv2.Rodrigues(rotation @ lr)[0]
        right_translation = rotation @ lt + translation
        world = objects[index]
        rows = {"index": index, "points": world.shape[1]}
        for side, record, vector, offset, observed in (
            ("left", left, lvec, lt, a[index]),
            ("right", right, right_rvec, right_translation, b[index]),
        ):
            matrix = np.array(record["camera_matrix"])
            distortion = np.array(record["distortion"])
            if np.any((world.reshape(-1, 3) @ cv2.Rodrigues(vector)[0].T + offset)[:, 2] <= 0):
                raise ValidationError("Stereo fit placed a target behind a camera")
            if lens == "fisheye":
                projected = cv2.fisheye.projectPoints(world, vector, offset, matrix, distortion)[0]
            else:
                projected = cv2.omnidir.projectPoints(
                    world, vector, offset, matrix, record["xi"], distortion
                )[0]
            residual = projected.reshape(-1, 2) - observed.reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=-1))))
            if not np.isfinite(rms):
                raise ValidationError("Stereo reprojection returned invalid residuals")
            rows[f"{side}_rms_pixels"] = rms
        report.append(rows)
    return report


class CalibrateWideStereo(Tool3D):
    type_id, title, subtype = (
        "3d.wide_stereo_calibrate",
        "Calibrate wide-angle stereo",
        "Calibration",
    )
    inputs = (
        Port("left_camera", "data"),
        Port("right_camera", "data"),
        *(Port(name, "data", required=False, required_when_connected=True) for name in _VIEWS),
    )
    outputs = (
        Port("rig", "data", required=False),
        Port("rms", "number"),
        Port("fit_report", "data"),
    )
    defaults = {
        "path": "",
        "path_base": "project",
        "fisheye_intrinsics": "fixed",
        "fisheye_check_condition": False,
        "fisheye_recompute_extrinsics": True,
        **{f"fisheye_fix_k{i}": False for i in range(1, 5)},
        **{f"omni_{key}": False for key in OMNIDIR_FLAGS},
        "view_policy": "require_all",
        "minimum_views": 3,
        "iterations": 200,
        "epsilon": 1e-9,
        "max_rms": 1.0,
        "max_view_rms": 0.0,
    }
    parameter_choices = {
        "path_base": ("project", "working_directory"),
        "fisheye_intrinsics": ("fixed", "refine"),
        "view_policy": ("require_all", "allow_subset"),
    }
    parameter_ranges = {
        "minimum_views": NumericRange(3, 32, whole=True),
        "iterations": NumericRange(1, 1000, whole=True),
        "epsilon": NumericRange(0, 1),
        "max_rms": NumericRange(1e-9),
        "max_view_rms": NumericRange(0),
    }
    parameter_help = {
        "path": "JSON object with object_views, left_views and right_views.",
        "path_base": "Base for a relative observations file.",
        "fisheye_intrinsics": "Keep the supplied intrinsics or refine the starting values.",
        "fisheye_check_condition": "Have OpenCV check the fisheye fit conditioning.",
        "fisheye_recompute_extrinsics": "Refit poses while initializing fisheye refinement.",
        **{
            f"fisheye_fix_k{i}": f"Fix fisheye k{i} at zero during refinement." for i in range(1, 5)
        },
        **{
            f"omni_{key}": f"Lock {key.removeprefix('fix_')} at OpenCV's omnidir initialization."
            for key in OMNIDIR_FLAGS
        },
        "view_policy": "Require every pair or allow the subset OpenCV kept during initialization.",
        "minimum_views": "Minimum number of fitted stereo pairs.",
        "max_rms": "Overall OpenCV reprojection RMS limit, in pixels.",
        "max_view_rms": "Worst single-camera view RMS limit, in pixels. Zero disables this check.",
    }
    input_help = {
        "left_camera": "Fisheye seed or omnidirectional frame, image grid and units.",
        "right_camera": "Same lens family, dimensions and units as Left camera.",
        "object_views": "Corresponding target XYZ arrays, in the camera model's units.",
        "left_views": "Left distorted pixel arrays in target point order.",
        "right_views": "Right distorted pixel arrays in the same target point order.",
    }
    process_inputs_readonly = parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Use a JSON observations file or leave the path empty", "path")
        for name in self.defaults:
            if isinstance(self.defaults[name], bool) and type(p[name]) is not bool:
                raise ValidationError("Use a checkbox", name)
        for name, limit in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limit.whole and type(p[name]) is not int:
                raise ValidationError("Use a numeric setting", name)

    def process(self, inputs):
        from .stereo3d import stereo_rig_record

        p = self.parameters
        a, b = inputs["left_camera"], inputs["right_camera"]
        (k1, d1, size), (k2, d2, _), lens = stereo_cameras(a, b)
        if lens not in ("fisheye", "omnidir"):
            raise ValidationError("Connect two fisheye cameras or two omnidirectional cameras")
        views = read_views(self, inputs, size, lens)
        eps = p["epsilon"]
        if lens == "fisheye":
            if p["fisheye_intrinsics"] == "fixed":
                flags = cv2.CALIB_FIX_INTRINSIC
            else:
                flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_SKEW
                options = {
                    "fisheye_recompute_extrinsics": cv2.CALIB_RECOMPUTE_EXTRINSIC,
                    **{f"fisheye_fix_k{i}": getattr(cv2, f"CALIB_FIX_K{i}") for i in range(1, 5)},
                }
                flags |= sum(flag for key, flag in options.items() if p[key])
            if p["fisheye_check_condition"]:
                flags |= cv2.CALIB_CHECK_COND
            criteria = (
                cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if eps else 0),
                p["iterations"],
                eps,
            )
            rms, k1, d1, k2, d2, r, t, rv, tv = cv2.fisheye.stereoCalibrate(
                *views,
                k1.copy(),
                d1.copy(),
                k2.copy(),
                d2.copy(),
                size,
                flags=flags,
                criteria=criteria,
            )
            used = selected_indices(np.arange(len(views[0])), len(views[0]), rv, tv)
            extras = ({}, {})
        else:
            if not hasattr(cv2, "omnidir"):
                raise ValidationError("Omnidirectional stereo requires OpenCV contrib")
            flags = cv2.omnidir.CALIB_FIX_SKEW | sum(
                getattr(cv2.omnidir, flag)
                for key, flag in OMNIDIR_FLAGS.items()
                if p[f"omni_{key}"]
            )
            criteria = (
                cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if eps else 0),
                p["iterations"],
                eps,
            )
            # This implementation initializes each camera internally and returns the selected pairs.
            rms, _, _, _, k1, xi1, d1, k2, xi2, d2, rod, t, rv, tv, kept = (
                cv2.omnidir.stereoCalibrate(
                    *views,
                    size,
                    size,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    flags,
                    criteria,
                )
            )
            r = cv2.Rodrigues(rod)[0]
            used = selected_indices(kept, len(views[0]), rv, tv)
            xis = [float(np.asarray(xi).item()) for xi in (xi1, xi2)]
            if any(not np.isfinite(x) or x < 0 for x in xis):
                raise ValidationError("Native stereo calibration returned invalid xi")
            extras = tuple({"xi": x} for x in xis)
        if not np.isfinite(rms) or rms < 0:
            raise ValidationError("Stereo calibration returned invalid reprojection RMS")
        left = camera_record(k1, d1, size, a["units"], a["frame"], lens, **extras[0])
        right = camera_record(k2, d2, size, b["units"], b["frame"], lens, **extras[1])
        t = t.reshape(3)
        pose = pose_record(pose_matrix(r, t), a["frame"], b["frame"], a["units"])
        rig = stereo_rig_record(left, right, pose)
        errors = view_report(lens, views, used, rv, tv, left, right, r, t)
        excluded = sorted(set(range(len(views[0]))) - set(used))
        worst = max(max(v["left_rms_pixels"], v["right_rms_pixels"]) for v in errors)
        reasons = []
        if len(used) < p["minimum_views"]:
            reasons.append("wide_stereo.insufficient_views")
        if excluded and p["view_policy"] == "require_all":
            reasons.append("wide_stereo.excluded_views")
        if rms > p["max_rms"]:
            reasons.append("wide_stereo.rms")
        if p["max_view_rms"] and worst > p["max_view_rms"]:
            reasons.append("wide_stereo.view_rms")
        report = {
            "model": "wide_stereo_fit_report_v1",
            "lens": lens,
            "baseline": float(np.linalg.norm(t)),
            "units": rig["units"],
            "used_view_indices": used,
            "excluded_view_indices": excluded,
            "views": errors,
            "worst_view_rms_pixels": worst,
        }
        return Result(
            {"rig": NO_VALUE if reasons else rig, "rms": float(rms), "fit_report": report},
            passed=not reasons,
            message=f"{len(used)}/{len(views[0])} views, RMS {rms:.4g} px",
            fail_reasons=reasons,
        )


def geometry(rig, calibrated, pose, output_size, parameters):
    (k1, d1, size), (k2, d2, _) = calibrated
    lens = rig["left"].get("lens", "pinhole")
    r, t = pose[:3, :3].copy(), pose[:3, 3].copy().reshape(3, 1)
    flags = cv2.CALIB_ZERO_DISPARITY if parameters["zero_disparity"] else 0
    if lens == "pinhole":
        alpha = finite(parameters["alpha"], "Alpha")
        if not -1 <= alpha <= 1:
            raise ValidationError("Rectification alpha must be in [-1,1]")
        return cv2.stereoRectify(
            k1, d1, k2, d2, size, r, t, flags=flags, alpha=alpha, newImageSize=output_size
        )
    if lens == "fisheye":
        args = cv2.fisheye.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            size,
            r,
            t,
            flags,
            newImageSize=output_size,
            balance=parameters["fisheye_balance"],
            fov_scale=parameters["fisheye_fov_scale"],
        )
        if not all(np.isfinite(value).all() for value in args):
            raise ValidationError("Fisheye stereo rectification returned nonfinite geometry")
        if abs(args[3][0, 3]) <= 1e-9 or abs(args[3][1, 3]) > 1e-9:
            raise ValidationError("This fisheye fit does not produce horizontal disparity")
        return (*args, None, None)
    r1, r2 = cv2.omnidir.stereoRectify(r, t)
    tx, ty, tz = (r2 @ t).reshape(3)
    if not np.isfinite([tx, ty, tz]).all() or tx >= -1e-10 or abs(ty) + abs(tz) > -tx * 1e-7:
        raise ValidationError(
            "Omnidirectional rig does not provide horizontal perspective geometry"
        )
    width, height = output_size
    focal = parameters["omnidir_focal_length"] or min(
        min(k1[0, 0], k1[1, 1]) / (1 + rig["left"]["xi"]),
        min(k2[0, 0], k2[1, 1]) / (1 + rig["right"]["xi"]),
    ) * min(width / size[0], height / size[1])
    cx, cy = (width - 1) / 2, (height - 1) / 2
    p1 = np.array([[focal, 0, cx, 0], [0, focal, cy, 0], [0, 0, 1, 0]])
    p2 = p1.copy()
    p2[0, 3] = focal * tx
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal], [0, 0, -1 / tx, 0]])
    return r1, r2, p1, p2, q, None, None


def remap_for_camera(model, k, d, r, p, size):
    lens = model.get("lens", "pinhole")
    if lens == "fisheye":
        return cv2.fisheye.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
    if lens == "omnidir":
        return cv2.omnidir.initUndistortRectifyMap(
            k,
            d,
            np.array([model["xi"]]),
            r,
            p[:, :3].copy(),
            size,
            cv2.CV_32FC1,
            cv2.omnidir.RECTIFY_PERSPECTIVE,
        )
    return cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
