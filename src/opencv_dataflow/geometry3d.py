"""Camera projection, PnP, epipolar geometry, triangulation and rigid pose blocks."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera,
    camera_record,
    matrix,
    points,
    pose_matrix,
    pose_record,
    require_pinhole,
    rigid,
)
from .payloads import NO_VALUE
from .pointclouds import PointCloud
from .processing import finite, integer


class CameraModel(Tool3D):
    type_id, title, subtype = "3d.camera", "Camera intrinsic model", "Camera models"
    inputs = (Port("calibration", "data", required=False),)
    outputs = (Port("camera", "data"),)
    defaults = {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "distortion": [],
        "lens": "pinhole",
        "xi": 1.0,
        "units": "m",
        "frame": "camera",
    }
    parameter_choices = {
        "lens": ("pinhole", "fisheye", "omnidir"),
        "units": ("m", "mm", "unitless"),
    }

    def process(self, inputs):
        p = self.parameters
        if "calibration" in inputs:
            model = inputs["calibration"]
            k, d, size = camera(model)
            lens = model.get("lens", "pinhole")
            return Result(
                {
                    "camera": camera_record(
                        k,
                        d,
                        size,
                        model.get("units", p["units"]),
                        p["frame"],
                        lens,
                        **({"xi": model["xi"]} if lens == "omnidir" else {}),
                    )
                }
            )
        k = [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]]
        return Result(
            {
                "camera": camera_record(
                    k,
                    p["distortion"],
                    [p["width"], p["height"]],
                    p["units"],
                    p["frame"],
                    p["lens"],
                    **({"xi": p["xi"]} if p["lens"] == "omnidir" else {}),
                )
            }
        )


class PointData3D(Tool3D):
    type_id, title = "3d.points", "3D point data"
    outputs = (Port("points", "data"), Port("cloud", "cloud"))
    defaults = {
        "points": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        "units": "m",
        "frame": "object",
    }
    parameter_choices = {"units": ("m", "mm", "unitless")}

    def process(self, inputs):
        xyz = points(self.parameters["points"])
        return Result(
            {
                "points": xyz.tolist(),
                "cloud": PointCloud(xyz, self.parameters["units"], self.parameters["frame"]),
            }
        )


class Pose3D(Tool3D):
    type_id, title = "3d.pose", "Define rigid 3D pose"
    outputs = (Port("transform", "data"),)
    defaults = {
        "rvec": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
        "source_frame": "object",
        "target_frame": "camera",
        "units": "m",
    }
    parameter_choices = {"units": ("m", "mm", "unitless")}
    parameter_help = {
        "rvec": "Rodrigues axis-angle rotation vector, in radians.",
        "translation": "XYZ translation, expressed in the selected length units.",
    }

    def process(self, inputs):
        p = self.parameters
        rvec = matrix([p["rvec"]], (1, 3), "Rodrigues rotation").reshape(3)
        translation = matrix([p["translation"]], (1, 3), "Translation").reshape(3)
        transform = pose_matrix(cv2.Rodrigues(rvec)[0], translation)
        return Result(
            {"transform": pose_record(transform, p["source_frame"], p["target_frame"], p["units"])}
        )


class ComposePose3D(Tool3D):
    type_id, title = "3d.compose", "Compose / invert 3D poses"
    inputs = (Port("first", "data"), Port("second", "data", required=False))
    outputs = (Port("transform", "data"),)
    defaults = {"operation": "compose"}
    parameter_choices = {"operation": ("compose", "inverse")}

    def process(self, inputs):
        a = inputs["first"]
        if self.parameters["operation"] == "inverse":
            result = pose_record(
                np.linalg.inv(rigid(a)), a["target_frame"], a["source_frame"], a["units"]
            )
        else:
            b = inputs.get("second")
            if not b or a["target_frame"] != b["source_frame"] or a["units"] != b["units"]:
                raise ValueError(
                    "Compose requires first target = second source and identical units"
                )
            result = pose_record(
                rigid(b) @ rigid(a), a["source_frame"], b["target_frame"], a["units"]
            )
        return Result({"transform": result})


class ProjectPoints3D(Tool3D):
    type_id, title = "3d.project", "Project 3D points"
    inputs = (Port("points", "data"), Port("camera", "data"), Port("transform", "data"))
    outputs = (Port("points", "data"), Port("visible", "data"), Port("projection", "data"))
    defaults = {}

    def process(self, inputs):
        model = inputs["camera"]
        k, d, size = camera(model)
        xyz = points(inputs["points"])
        pose = inputs["transform"]
        if pose["units"] != model["units"] or pose["target_frame"] != model["frame"]:
            raise ValueError("Pose units/target frame do not match the camera")
        t = rigid(pose)
        lens = model.get("lens", "pinhole")
        rvec = cv2.Rodrigues(t[:3, :3])[0]
        if lens == "fisheye":
            pixels, _ = cv2.fisheye.projectPoints(xyz[:, None], rvec, t[:3, 3], k, d)
        elif lens == "omnidir":
            pixels, _ = cv2.omnidir.projectPoints(xyz[:, None], rvec, t[:3, 3], k, model["xi"], d)
        else:
            pixels, _ = cv2.projectPoints(xyz, rvec, t[:3, 3], k, d)
        pixels = pixels.reshape(-1, 2)
        z = (xyz @ t[:3, :3].T + t[:3, 3])[:, 2]
        visible = (
            (z > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < size[0])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < size[1])
        )
        if not np.isfinite(pixels).all():
            raise ValueError("Projection contains points at infinity")
        return Result(
            {
                "points": pixels.tolist(),
                "visible": visible.tolist(),
                "projection": (k @ t[:3]).tolist()
                if lens == "pinhole" and not np.any(d)
                else NO_VALUE,
            }
        )


class UndistortPoints3D(Tool3D):
    type_id, title, subtype = "3d.undistort_points", "Undistort camera points", "Camera models"
    inputs = (Port("points", "data"), Port("camera", "data"))
    outputs = (Port("points", "data"),)
    defaults = {"output": "normalized"}
    parameter_choices = {"output": ("normalized", "pixels")}

    def process(self, inputs):
        model = inputs["camera"]
        k, d, _ = camera(model)
        xy = points(inputs["points"], 2).reshape(-1, 1, 2)
        projection = k if self.parameters["output"] == "pixels" else None
        lens = model.get("lens", "pinhole")
        if lens == "fisheye":
            result = cv2.fisheye.undistortPoints(xy, k, d, P=projection)
        elif lens == "omnidir":
            result = cv2.omnidir.undistortPoints(xy, k, d, np.array([model["xi"]]), np.eye(3))
            if projection is not None:
                result = cv2.perspectiveTransform(result, k)
        else:
            result = cv2.undistortPoints(xy, k, d, P=projection)
        return Result({"points": result.reshape(-1, 2).tolist()})


PNP = {
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "epnp": cv2.SOLVEPNP_EPNP,
    "sqpnp": cv2.SOLVEPNP_SQPNP,
    "ippe": cv2.SOLVEPNP_IPPE,
}


class SolvePnP3D(Tool3D):
    type_id, title = "3d.pnp", "3D object pose (PnP)"
    inputs = (
        Port("object_points", "data"),
        Port("image_points", "data"),
        Port("camera", "data"),
        Port("initial_pose", "data", required=False),
    )
    outputs = (Port("pose", "data"), Port("inliers", "data"), Port("rms", "number"))
    defaults = {
        "method": "iterative",
        "ransac": True,
        "iterations": 100,
        "reprojection_error": 3.0,
        "confidence": 0.99,
        "refinement": "lm",
        "refine_iterations": 20,
        "refine_epsilon": 1e-6,
        "source_frame": "object",
        "minimum_inliers": 4,
        "max_rms": 3.0,
    }
    parameter_choices = {"method": tuple(PNP), "refinement": ("none", "lm", "vvs")}

    def process(self, inputs):
        p = self.parameters
        model = inputs["camera"]
        k, d, _ = require_pinhole(model)
        xyz, xy = points(inputs["object_points"], 3, 4), points(inputs["image_points"], 2, 4)
        if len(xyz) != len(xy):
            raise ValueError("PnP object/image point counts differ")
        guess = inputs.get("initial_pose")
        rv, tv = None, None
        if guess:
            if (
                guess["units"] != model["units"]
                or guess["target_frame"] != model["frame"]
                or guess["source_frame"] != p["source_frame"]
            ):
                raise ValueError("Initial pose units/frame do not match camera")
            t = rigid(guess)
            rv, tv = cv2.Rodrigues(t[:3, :3])[0], t[:3, 3].copy()
        minimum = integer(p["minimum_inliers"], "Minimum inliers", 4, len(xyz))
        threshold = finite(p["reprojection_error"], "Reprojection error", 0.0001)
        confidence = finite(p["confidence"], "Confidence", 0.0001)
        if confidence >= 1:
            raise ValueError("Confidence must be below one")
        if p["ransac"]:
            ok, rv, tv, indices = cv2.solvePnPRansac(
                xyz,
                xy,
                k,
                d,
                rvec=rv,
                tvec=tv,
                useExtrinsicGuess=bool(guess),
                iterationsCount=integer(p["iterations"], "Iterations", 1, 100000),
                reprojectionError=threshold,
                confidence=confidence,
                flags=PNP[p["method"]],
            )
            selected = indices.reshape(-1) if indices is not None else np.empty(0, int)
        else:
            ok, rv, tv = cv2.solvePnP(xyz, xy, k, d, rv, tv, bool(guess), PNP[p["method"]])
            selected = np.arange(len(xyz)) if ok else np.empty(0, int)
        if not ok or len(selected) < minimum:
            return Result(
                {"pose": NO_VALUE, "inliers": selected.tolist(), "rms": 0.0},
                passed=False,
                message="Insufficient PnP inliers",
                fail_reasons=["pnp_inliers"],
            )
        if p["refinement"] != "none":
            criteria = (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                integer(p["refine_iterations"], "Refine iterations", 1, 10000),
                finite(p["refine_epsilon"], "Refine epsilon", 0),
            )
            refine = cv2.solvePnPRefineLM if p["refinement"] == "lm" else cv2.solvePnPRefineVVS
            rv, tv = refine(xyz[selected], xy[selected], k, d, rv, tv, criteria=criteria)
        projected = cv2.projectPoints(xyz[selected], rv, tv, k, d)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - xy[selected]) ** 2, axis=1))))
        passed = rms <= finite(p["max_rms"], "Maximum RMS", 0)
        pose = pose_record(
            pose_matrix(cv2.Rodrigues(rv)[0], tv), p["source_frame"], model["frame"], model["units"]
        )
        return Result(
            {"pose": pose if passed else NO_VALUE, "inliers": selected.tolist(), "rms": rms},
            passed=passed,
            message=f"PnP RMS {rms:.4g} pixels",
            fail_reasons=[] if passed else ["pnp_rms"],
        )


class Fundamental3D(Tool3D):
    type_id, title, subtype = "3d.fundamental", "Fundamental matrix", "Reconstruction"
    inputs = (Port("points_left", "data"), Port("points_right", "data"))
    outputs = (Port("matrix", "data"), Port("inliers", "data"))
    defaults = {"method": "ransac", "threshold": 1.0, "confidence": 0.99, "iterations": 2000}
    parameter_choices = {"method": ("ransac", "lmeds", "eight_point", "usac_magsac")}

    def process(self, inputs):
        a, b = points(inputs["points_left"], 2, 8), points(inputs["points_right"], 2, 8)
        if len(a) != len(b):
            raise ValueError("Point correspondence counts differ")
        p = self.parameters
        confidence = finite(p["confidence"], "Confidence", 0.0001)
        if confidence >= 1:
            raise ValueError("Confidence must be below one")
        f, mask = cv2.findFundamentalMat(
            a,
            b,
            {
                "ransac": cv2.FM_RANSAC,
                "lmeds": cv2.FM_LMEDS,
                "eight_point": cv2.FM_8POINT,
                "usac_magsac": cv2.USAC_MAGSAC,
            }[p["method"]],
            finite(p["threshold"], "Threshold", 0.0001),
            confidence,
            integer(p["iterations"], "Iterations", 1, 100000),
        )
        ok = f is not None and f.shape == (3, 3) and np.isfinite(f).all()
        return Result(
            {
                "matrix": f.tolist() if ok else NO_VALUE,
                "inliers": mask.reshape(-1).astype(bool).tolist() if mask is not None else [],
            },
            passed=ok,
            fail_reasons=[] if ok else ["fundamental_matrix"],
        )


class EssentialPose3D(Tool3D):
    type_id, title, subtype = (
        "3d.essential_pose",
        "Essential matrix / relative pose",
        "Reconstruction",
    )
    inputs = (*Fundamental3D.inputs, Port("camera", "data"))
    outputs = (Port("essential", "data"), Port("pose", "data"), Port("inliers", "data"))
    defaults = {
        "method": "ransac",
        "threshold": 1.0,
        "confidence": 0.999,
        "iterations": 2000,
        "left_frame": "left_camera",
        "right_frame": "right_camera",
        "minimum_inliers": 8,
    }
    parameter_choices = {"method": ("ransac", "lmeds")}

    def process(self, inputs):
        p = self.parameters
        a, b = points(inputs["points_left"], 2, 8), points(inputs["points_right"], 2, 8)
        k, d, _ = require_pinhole(inputs["camera"])
        if len(a) != len(b):
            raise ValueError("Point correspondence counts differ")
        a = cv2.undistortPoints(a[:, None], k, d, P=k).reshape(-1, 2)
        b = cv2.undistortPoints(b[:, None], k, d, P=k).reshape(-1, 2)
        confidence = finite(p["confidence"], "Confidence", 0.0001)
        if confidence >= 1:
            raise ValueError("Confidence must be below one")
        e, mask = cv2.findEssentialMat(
            a,
            b,
            k,
            method=cv2.RANSAC if p["method"] == "ransac" else cv2.LMEDS,
            prob=confidence,
            threshold=finite(p["threshold"], "Threshold", 0.0001),
            maxIters=integer(p["iterations"], "Iterations", 1, 100000),
        )
        ok = e is not None and e.shape == (3, 3)
        pose = NO_VALUE
        good = 0
        if ok:
            good, r, t, mask = cv2.recoverPose(e, a, b, k, mask=mask)
            ok = good >= integer(p["minimum_inliers"], "Minimum inliers", 5, len(a))
            if ok:
                pose = pose_record(
                    pose_matrix(r, t),
                    p["left_frame"],
                    p["right_frame"],
                    "unitless",
                    scale="unknown",
                )
        return Result(
            {
                "essential": e.tolist() if e is not None else NO_VALUE,
                "pose": pose,
                "inliers": mask.reshape(-1).astype(bool).tolist() if mask is not None else [],
            },
            passed=ok,
            message=f"{good} cheirality inliers; translation scale is unknown",
            fail_reasons=[] if ok else ["relative_pose"],
        )


class Triangulate3D(Tool3D):
    type_id, title, subtype = "3d.triangulate", "Triangulate 3D points", "Reconstruction"
    inputs = (
        *Fundamental3D.inputs,
        Port("projection_left", "data"),
        Port("projection_right", "data"),
    )
    outputs = (Port("cloud", "cloud"), Port("valid", "data"), Port("rms", "number"))
    defaults = {
        "units": "m",
        "frame": "left_camera",
        "max_reprojection_error": 3.0,
        "positive_depth": True,
    }
    parameter_choices = {"units": ("m", "mm", "unitless")}

    def process(self, inputs):
        a, b = points(inputs["points_left"], 2), points(inputs["points_right"], 2)
        if len(a) != len(b):
            raise ValueError("Point correspondence counts differ")
        p1, p2 = (matrix(inputs[k], (3, 4), k) for k in ("projection_left", "projection_right"))
        h = cv2.triangulatePoints(p1, p2, a.T, b.T).T
        valid = np.abs(h[:, 3]) > 1e-12
        xyz = h[:, :3] / np.where(valid, h[:, 3], 1)[:, None]
        errors = []
        for projection, xy in ((p1, a), (p2, b)):
            reproj = np.c_[xyz, np.ones(len(xyz))] @ projection.T
            valid &= np.abs(reproj[:, 2]) > 1e-12
            if self.parameters["positive_depth"]:
                valid &= reproj[:, 2] > 0
            pixels = (
                reproj[:, :2] / np.where(np.abs(reproj[:, 2]) > 1e-12, reproj[:, 2], 1)[:, None]
            )
            errors.append(np.linalg.norm(pixels - xy, axis=1))
        error = np.maximum(*errors)
        valid &= np.isfinite(xyz).all(axis=1) & (
            error <= finite(self.parameters["max_reprojection_error"], "Max reprojection error", 0)
        )
        cloud = PointCloud(xyz[valid], self.parameters["units"], self.parameters["frame"])
        return Result(
            {
                "cloud": cloud,
                "valid": valid.tolist(),
                "rms": float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else 0.0,
            },
            passed=bool(valid.any()),
            fail_reasons=[] if valid.any() else ["triangulation"],
        )


GEOMETRY3D_BLOCKS = (
    CameraModel,
    PointData3D,
    Pose3D,
    ComposePose3D,
    ProjectPoints3D,
    UndistortPoints3D,
    SolvePnP3D,
    Fundamental3D,
    EssentialPose3D,
    Triangulate3D,
)
