"""Common-row rectification and two metric disparity models for three cameras."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, applied_payload, digest
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, rigid
from .images import ImageArray, image_format
from .multicamera_calibration import RIG, _observations
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, metadata
from .stereo3d import STEREO_GRID
from .wide_stereo import geometry as stereo_geometry
from .wide_stereo import remap_for_camera

_VIEWS = ("reference", "second", "third")
_INTERPOLATION = {
    "nearest": (cv2.INTER_NEAREST, None),
    "linear": (cv2.INTER_LINEAR, 1),
    "cubic": (cv2.INTER_CUBIC, 2),
    "lanczos4": (cv2.INTER_LANCZOS4, 4),
}
_MIN_BASELINE_MM = 1e-4


def _rig_inputs(inputs):
    camera_keys = tuple(f"{name}_camera" for name in _VIEWS)
    pose_keys = ("reference_to_second", "reference_to_third")
    explicit_keys = (*camera_keys, *pose_keys)
    rig = inputs.get("rig")
    if rig is not None:
        if any(key in inputs for key in explicit_keys):
            raise ValidationError(
                "Connect either a three-camera rig or the five camera and pose inputs"
            )
        if not isinstance(rig, dict) or rig.get("model") != RIG or rig.get("lens") != "pinhole":
            raise ValidationError("Connect a pinhole multicamera_rig_v1 record", "rig")
        cameras = rig.get("cameras")
        poses = rig.get("anchor_to_camera")
        if not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in (cameras, poses)):
            raise ValidationError("Rig must contain exactly three ordered cameras and poses", "rig")
    else:
        if not all(key in inputs for key in explicit_keys):
            raise ValidationError("Connect a three-camera rig or three camera models and two poses")
        cameras = [inputs[key] for key in camera_keys]
        poses = [inputs[key] for key in pose_keys]

    if any(
        not isinstance(cam, dict)
        or cam.get("model") != "camera_model_v1"
        or cam.get("lens", "pinhole") != "pinhole"
        or not isinstance(cam.get("frame"), str)
        or not cam["frame"].strip()
        for cam in cameras
    ):
        raise ValidationError("Supply three named pinhole camera models")
    names = tuple(cam["frame"] for cam in cameras)
    units = cameras[0].get("units")
    if (
        len(set(names)) != 3
        or units not in ("mm", "m")
        or any(cam.get("units") != units for cam in cameras)
    ):
        raise ValidationError("Camera frames must be distinct and use the same metric units")
    if rig is not None and (rig.get("units") != units or rig.get("anchor_frame") != names[0]):
        raise ValidationError("Rig anchor and units must match its reference camera", "rig")

    transforms = []
    for name, pose in zip(names if rig is not None else names[1:], poses, strict=True):
        if (
            not isinstance(pose, dict)
            or pose.get("model") != "pose_3d_v1"
            or pose.get("source_frame") != names[0]
            or pose.get("target_frame") != name
            or pose.get("units") != units
        ):
            raise ValidationError("Poses must map reference to camera frames in rig units")
        transforms.append(rigid(pose))
    if rig is not None:
        if not np.allclose(transforms[0], np.eye(4), atol=1e-10, rtol=0):
            raise ValidationError("Rig anchor transform must be identity", "rig")
        transforms = transforms[1:]
    return cameras, names, units, transforms


def _parallel_rectification(cameras, optics, poses, size, alpha):
    # Use the reference/second pair to choose the camera orientation and shared zoom.
    native = stereo_geometry(
        {"left": cameras[0]}, optics[:2], poses[0], size, {"alpha": alpha, "zero_disparity": True}
    )
    r1, _, p1, p2, _, _, _ = native
    vertical = bool(abs(p2[1, 3]) > abs(p2[0, 3]))
    k = p1[:, :3].copy()
    if vertical:
        # Run the native crop calculation in the transposed canvas before rotating it.
        r1, _, p1, _, _, _, _ = stereo_geometry(
            {"left": cameras[0]},
            optics[:2],
            poses[0],
            size[::-1],
            {"alpha": alpha, "zero_disparity": True},
        )
        r1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ r1
        k = p1[:, :3].copy()
        k[0, 2], k[1, 2] = k[1, 2], size[1] - 1 - k[0, 2]
    rotations = [r1, *(r1 @ pose[:3, :3].T for pose in poses)]
    translations = [r @ pose[:3, 3] for r, pose in zip(rotations[1:], poses, strict=True)]
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValidationError("OpenCV could not create a finite rectification grid", "alpha")
    return k, rotations, translations, vertical


def _pair_geometry(k, rotations, tx, index, size, units, anchor, partner):
    f, cx, cy = k[0, 0], k[0, 2], k[1, 2]
    q = [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1 / tx, 0]]
    return {
        "model": "stereo_rectification_v1",
        "Q": q,
        "P1": np.column_stack((k, np.zeros(3))).tolist(),
        "P2": np.column_stack((k, k[:, 0] * tx)).tolist(),
        "R1": rotations[0].tolist(),
        "R2": rotations[index].tolist(),
        "roi_left": None,
        "roi_right": None,
        "image_size": list(size),
        "units": units,
        "frame": f"{anchor}:rectified",
        "right_frame": f"{partner}:rectified",
    }


def _check_rows(record, frames, units, optics, rotations, k, excluded):
    obs = _observations(record, maximum_cameras=3)
    if obs.frames != frames or units != obs.units or obs.sizes != tuple(o[2] for o in optics):
        raise ValidationError("Observation frames, units and image dimensions must match", "check")
    if not set(obs.captures).isdisjoint(excluded):
        raise ValidationError("Check captures must be separate from rig training IDs", "check")
    result = {}
    for index in (1, 2):
        shared = obs.point_visible[0] & obs.point_visible[index]
        if shared.sum() < 6:
            raise ValidationError("Each baseline needs at least six shared check points", "check")
        rows = []
        for view in (0, index):
            intrinsics, distortion, _ = optics[view]
            rectified = cv2.undistortPoints(
                obs.pixels[view][shared].astype(np.float64).reshape(-1, 1, 2),
                intrinsics,
                distortion,
                R=rotations[view],
                P=k,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9),
            )
            rows.append(rectified[:, 0, 1])
        residuals = rows[0] - rows[1]
        if not np.isfinite(residuals).all():
            raise ValidationError("Rectification returned nonfinite check points", "check")
        result[f"1-{index + 1}"] = {
            "matched_points": int(shared.sum()),
            "capture_count": int(shared.any(axis=1).sum()),
            "row_rms_px": float(np.sqrt(np.mean(residuals**2))),
            "row_max_px": float(np.max(np.abs(residuals))),
        }
    return result


def _rectify_image(
    image, model, k, d, rotation, projection, input_size, size, interpolation, border
):
    mx, my = remap_for_camera(model, k, d, rotation, projection, size)
    method, radius = _INTERPOLATION[interpolation]
    rectified = cv2.remap(
        np.asarray(image), mx, my, method, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )
    width, height = input_size
    if radius is None:
        support = cv2.remap(np.full((height, width), 255, np.uint8), mx, my, cv2.INTER_NEAREST)
    else:
        # Conservatively require all taps of the interpolation kernel to be on the sensor.
        support = (
            (mx >= radius - 1)
            & (mx <= width - radius)
            & (my >= radius - 1)
            & (my <= height - radius)
        ).astype(np.uint8) * 255
    return ImageArray(rectified, image_format(image)), support


class RectifyCollinearCameras(Tool3D):
    type_id, title, subtype = "3d.rectify_collinear", "Rectify three collinear cameras", "Stereo"
    description = "Rectify three calibrated pinhole cameras to a shared horizontal stereo grid."
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        *(Port(name) for name in _VIEWS),
        Port("rig", "data", required=False, required_when_connected=True),
        *(
            Port(f"{name}_camera", "data", required=False, required_when_connected=True)
            for name in _VIEWS
        ),
        Port(
            "reference_to_second",
            "data",
            required=False,
            required_when_connected=True,
            visual_type="pose",
        ),
        Port(
            "reference_to_third",
            "data",
            required=False,
            required_when_connected=True,
            visual_type="pose",
        ),
        Port("check", "data", required=False, required_when_connected=True),
    )
    outputs = (
        *(
            Port(name, required=False)
            for name in ("reference_12", "second", "reference_13", "third")
        ),
        *(Port(f"{name}_valid", required=False) for name in _VIEWS),
        Port("rectification_12", "data", required=False),
        Port("rectification_13", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "alpha": 0.0,
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border_value": 0.0,
        "max_center_offset_mm": 0.25,
        "max_row_rms_px": 0.5,
    }
    parameter_ranges = {
        "alpha": NumericRange(-1, 1),
        "width": NumericRange(0, 4096, whole=True),
        "height": NumericRange(0, 4096, whole=True),
        "max_center_offset_mm": NumericRange(0),
        "max_row_rms_px": NumericRange(0),
    }
    parameter_choices = {"interpolation": tuple(_INTERPOLATION)}
    parameter_help = {
        "alpha": "Crop for the reference/second pair. The third camera may need its validity mask.",
        "width": "Rectified width in pixels. Zero uses the reference width.",
        "height": "Rectified height in pixels. Zero uses the reference height.",
        "interpolation": "Sampling kernel for the three remaps.",
        "border_value": "Constant used outside the source rasters.",
        "max_center_offset_mm": "Maximum third center distance from the first-to-second baseline.",
        "max_row_rms_px": "Maximum row mismatch RMS per baseline when check points are connected.",
    }
    input_help = {
        "rig": "Three pinhole cameras from Calibrate multi-camera rig, in image-port order.",
        "reference_to_second": "Reference-to-second camera pose in the declared camera units.",
        "reference_to_third": "Reference-to-third camera pose in the declared camera units.",
        "check": "Independent observations of matched target points on both baselines.",
    }
    output_help = {
        "reference_12": "Reference pixels paired with Second and Rectification 12.",
        "reference_13": "The same reference pixels paired with Third and Rectification 13.",
        "reference_valid": "Trustworthy reference pixels for either baseline's left mask.",
        "second_valid": "Trustworthy second-camera pixels for baseline 12's right mask.",
        "third_valid": "Trustworthy third-camera pixels for baseline 13's right mask.",
        "rectification_12": "Projection, measured signed baseline and Q for reference/second.",
        "rectification_13": "Projection, measured signed baseline and Q for reference/third.",
        "report": "Baseline spacing, off-axis error, optional row residuals and valid fractions.",
    }

    def _fail(self, report, code, message):
        report["status"] = "failed"
        return Result(
            {**dict.fromkeys((port.name for port in self.outputs), NO_VALUE), "report": report},
            passed=False,
            fail_reasons=[code],
            message=message,
        )

    def process(self, inputs):
        cameras, names, units, poses = _rig_inputs(inputs)
        p = self.parameters
        images = [inputs[name] for name in _VIEWS]
        optics = [camera_for_image(cam, image) for cam, image in zip(cameras, images, strict=True)]
        input_size = optics[0][2]
        if any(o[2] != input_size for o in optics):
            raise ValidationError("Calibrated image sizes must match across all three cameras")
        size = (int(p["width"]) or input_size[0], int(p["height"]) or input_size[1])
        if max(size) > 4096:
            raise ValidationError("Set an output canvas of at most 4096 pixels per dimension")

        to_mm = 1000.0 if units == "m" else 1.0
        baseline = np.linalg.norm(poses[0][:3, 3])
        report = {
            "model": "collinear_rectification_report_v1",
            "camera_frames": list(names),
            "image_size": list(size),
            "units": units,
        }
        if baseline * to_mm < _MIN_BASELINE_MM:
            return self._fail(report, "collinear_baseline", "Reference and second cameras coincide")
        k, rotations, translations, vertical = _parallel_rectification(
            cameras, optics, poses, size, float(p["alpha"])
        )
        tx2, tx3 = (float(t[0]) for t in translations)
        offset = float(np.linalg.norm(translations[1][1:])) * to_mm
        report.update(
            signed_translations=[tx2, tx3],
            third_center_offset_mm=offset,
            rotated_vertical_pair=vertical,
        )
        if offset > float(p["max_center_offset_mm"]) + 1e-9:
            return self._fail(
                report,
                "collinear_center_offset",
                f"Third center is {offset:.4g} mm off the baseline",
            )
        if min(abs(tx2), abs(tx3), abs(tx3 - tx2)) * to_mm < _MIN_BASELINE_MM:
            return self._fail(report, "collinear_baseline", "Three distinct centers are required")
        if "check" in inputs:
            rows = _check_rows(
                inputs["check"],
                names,
                units,
                optics,
                rotations,
                k,
                inputs.get("rig", {}).get("training_capture_ids", ()),
            )
            report["check"] = rows
            maximum = max(value["row_rms_px"] for value in rows.values())
            if maximum > float(p["max_row_rms_px"]):
                return self._fail(
                    report, "collinear_row_error", f"Held-out row RMS reaches {maximum:.4g} px"
                )
        pairs = [
            _pair_geometry(k, rotations, tx, index, size, units, names[0], names[index])
            for index, tx in enumerate((tx2, tx3), start=1)
        ]
        views, masks, contexts = [], [], []
        for index, (name, image, cam, (raw_k, d, _), rotation) in enumerate(
            zip(_VIEWS, images, cameras, optics, rotations, strict=True)
        ):
            view, mask = _rectify_image(
                image,
                cam,
                raw_k,
                d,
                rotation,
                k,
                input_size,
                size,
                p["interpolation"],
                float(p["border_value"]),
            )
            views.append(view)
            masks.append(mask)
            context = metadata(image)
            context.pop("pyramid", None)
            context.pop(STEREO_GRID, None)
            context.update(
                spatial=frame(size, f"{self.id}:rectified:{name}"),
                optical_frame=f"{names[index]}:rectified",
            )
            contexts.append(applied_payload(context, cam.get(REVISION)))
        fractions = [float(np.mean(mask != 0)) for mask in masks]
        shared_rows = [
            int(np.count_nonzero(masks[0].any(axis=1) & mask.any(axis=1))) for mask in masks[1:]
        ]
        report.update(valid_fractions=fractions, shared_rows=shared_rows)
        if not all(shared_rows):
            return self._fail(report, "collinear_common_support", "A pair shares no valid rows")

        outputs = {f"{name}_valid": mask for name, mask in zip(_VIEWS, masks, strict=True)}
        payloads = {
            f"{name}_valid": context for name, context in zip(_VIEWS, contexts, strict=True)
        }
        for suffix, name, index, pair in zip(("12", "13"), _VIEWS[1:], (1, 2), pairs, strict=True):
            outputs[f"rectification_{suffix}"] = pair
            stamp = digest(pair)
            for port, camera_index, side in (
                (f"reference_{suffix}", 0, "left"),
                (name, index, "right"),
            ):
                outputs[port] = views[camera_index]
                context = contexts[camera_index]
                payloads[port] = {
                    **context,
                    STEREO_GRID: {"sha256": stamp, "side": side, "spatial": context["spatial"]},
                }
        report["status"] = "ok"
        return Result(
            {**outputs, "report": report},
            payloads=payloads,
            message=f"Two baselines, {offset:.4g} mm third-center offset",
        )
