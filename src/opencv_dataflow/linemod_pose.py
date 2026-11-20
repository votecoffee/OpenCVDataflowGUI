"""Train and localize metric RGB-D views with OpenCV LINEMOD."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .calibration_revisions import is_full_grid
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, pose_record, rigid
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .processing import gray_image, image_value
from .selection import selection_mask
from .stereo3d import metric_depth
from .tickets import ticket_key

_MODEL = "linemod_rgbd_views_v1"
_CLASS = "object"
_STEPS = (5, 8)
_MODALITIES = ("ColorGradient", "DepthNormal")
_MAX_VIEWS = 12
_MAX_SIZE = 1024
_MARGIN = 48
_PATCH = 3
_PATCH_AREA = (2 * _PATCH + 1) ** 2

_SETTINGS = {
    "color_weak_threshold": 10.0,
    "color_strong_threshold": 55.0,
    "color_features": 63,
    "depth_features": 63,
    "normal_max_range_mm": 2000,
    "normal_max_difference_mm": 50,
    "normal_extract_threshold": 2,
}
_SETTING_RANGES = {
    "color_weak_threshold": NumericRange(1, 250),
    "color_strong_threshold": NumericRange(1, 500),
    "color_features": NumericRange(8, 63, whole=True),
    "depth_features": NumericRange(8, 63, whole=True),
    "normal_max_range_mm": NumericRange(100, 10_000, whole=True),
    "normal_max_difference_mm": NumericRange(1, 500, whole=True),
    "normal_extract_threshold": NumericRange(1, 24, whole=True),
}
_INPUTS = (
    Port("image"),
    Port("depth", "depth"),
    Port("camera", "data"),
    Port("mask", required=False, visual_type="region"),
    Port("region", "region", required=False),
)


def _modality_settings(settings):
    if not isinstance(settings, dict) or settings.keys() != _SETTINGS.keys():
        raise ValidationError("Invalid LINEMOD modality settings")
    p = {key: limit.check(settings[key], key, key) for key, limit in _SETTING_RANGES.items()}
    if p["color_weak_threshold"] > p["color_strong_threshold"]:
        raise ValidationError(
            "Strong gradient threshold must be at least Weak gradient threshold",
            "color_weak_threshold",
            "color_strong_threshold",
        )
    return p


def _detector(p):
    if not hasattr(cv2, "linemod"):
        raise RuntimeError("Install an OpenCV contrib build with LINEMOD")
    color = cv2.linemod.ColorGradient.create(
        p["color_weak_threshold"], p["color_features"], p["color_strong_threshold"]
    )
    depth = cv2.linemod.DepthNormal.create(
        p["normal_max_range_mm"],
        p["normal_max_difference_mm"],
        p["depth_features"],
        p["normal_extract_threshold"],
    )
    return cv2.linemod.Detector([color, depth], list(_STEPS))


def _pad(array, fill=0):
    # Each level needs a multiple of its sampling step. A 160-pixel quantum
    # also keeps the decimated row width aligned for the fine-level search.
    h, w = array.shape[:2]
    quantum = 160
    rows, cols = ((n + 2 * _MARGIN + quantum - 1) // quantum * quantum for n in (h, w))
    return cv2.copyMakeBorder(
        array,
        _MARGIN,
        rows - h - _MARGIN,
        _MARGIN,
        cols - w - _MARGIN,
        cv2.BORDER_CONSTANT,
        value=fill,
    )


def _depth_mm(z, maximum):
    mm = np.full(z.shape, 65535, np.uint16)
    valid = np.isfinite(z) & (z >= 0.001) & (z < maximum / 1000)
    mm[valid] = np.clip(np.rint(1000 * z[valid]), 1, maximum - 1).astype(np.uint16)
    return mm


def _captures(block, inputs, aligned):
    image, depth, camera = (inputs[name] for name in ("image", "depth", "camera"))
    k, d, size = camera_for_image(camera, image)
    if min(size) < 48 or max(size) > _MAX_SIZE:
        raise ValidationError(f"LINEMOD requires image sides between 48 and {_MAX_SIZE} pixels")
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValidationError("Depth must be registered to the image and use matching dimensions")
    if camera.get("lens", "pinhole") != "pinhole" or np.any(d):
        raise ValidationError("LINEMOD uses undistorted pinhole camera intrinsics")
    packets = block.input_payloads
    ip, dp = packets["image"], packets["depth"]
    optical = camera.get("frame", "camera")
    for metadata in (ip, dp):
        if not is_full_grid(metadata, size) or metadata.get("optical_frame", optical) != optical:
            raise ValidationError("LINEMOD inputs must use the declared full optical grid")
    if not aligned:
        require_same_frame((image, depth), message="Register depth to the image optical grid")
        image_root = ip["spatial"].get("root_id")
        if image_root is None or dp["spatial"].get("root_id") != image_root:
            raise ValidationError("Register RGB-D grids or declare them aligned in Pixel grid mode")
    names = ("image", "depth", "camera", "mask", "region", "pose")
    tickets = {
        ticket_key(packets[name]["ticket"]) for name in names if "ticket" in packets.get(name, {})
    }
    if len(tickets) > 1:
        raise ValidationError("RGB-D capture inputs must share a ticket")
    color = image_value(image, (np.uint8,), canonical=True)
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    elif color.shape[2] == 4:
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    gray = gray_image(image, (np.uint8,))
    return color, gray, metric_depth(depth, 1.0, "m"), k, size, optical


def _pose(value, optical):
    if (
        not isinstance(value, dict)
        or value.get("model") != "pose_3d_v1"
        or not isinstance(value.get("source_frame"), str)
        or not value["source_frame"]
        or value["source_frame"] == optical
        or value.get("target_frame") != optical
        or value.get("units") not in METERS
    ):
        raise ValidationError("Training pose must map the named metric object into the camera")
    transform = rigid(value).copy()
    transform[:3, 3] *= METERS[value["units"]]
    return value["source_frame"], transform


def _integers(value, length, low, high):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(x) is int and low <= x <= high for x in value)
    )


def _check_view(view, size):
    if (
        not isinstance(view, dict)
        or not isinstance(view.get("id"), str)
        or not 1 <= len(view["id"]) <= 64
    ):
        raise ValidationError("Use a named RGB-D view with a name up to 64 characters")
    rigid(view.get("object_to_camera"))
    box = view.get("bounds")
    if not _integers(box, 4, 0, _MAX_SIZE):
        raise ValidationError("Invalid LINEMOD view bounds")
    x, y, w, h = box
    if min(w, h) < 8 or x + w >= size[0] or y + h >= size[1]:
        raise ValidationError("LINEMOD view bounds must fit inside the training image")
    templates = view.get("templates")
    if not isinstance(templates, list) or len(templates) != len(_STEPS) * len(_MODALITIES):
        raise ValidationError("Each view needs color and depth templates at both scales")
    for i, templ in enumerate(templates):
        level = i // len(_MODALITIES)
        width, height = w >> level, h >> level
        if (
            not isinstance(templ, dict)
            or templ.get("width") != width
            or templ.get("height") != height
            or templ.get("pyramid_level") != level
            or not isinstance(templ.get("features"), list)
            or not 4 <= len(templ["features"]) <= 63
        ):
            raise ValidationError("Invalid bounded LINEMOD pyramid template")
        for feature in templ["features"]:
            if (
                not _integers(feature, 3, 0, _MAX_SIZE)
                or feature[0] > width
                or feature[1] > height
                or feature[2] > 7
            ):
                raise ValidationError("Invalid LINEMOD feature location or orientation")
    anchors = view.get("anchors")
    if not isinstance(anchors, list) or not 8 <= len(anchors) <= 48:
        raise ValidationError("Each RGB-D view needs 8–48 metric anchors")
    for anchor in anchors:
        if not isinstance(anchor, dict) or not _integers(anchor.get("offset"), 2, 0, _MAX_SIZE):
            raise ValidationError("Invalid RGB-D anchor")
        u, v = anchor["offset"]
        if (
            not _PATCH <= x + u < size[0] - _PATCH
            or not _PATCH <= y + v < size[1] - _PATCH
            or u > w
            or v > h
            or not _integers(anchor.get("patch"), _PATCH_AREA, 0, 255)
            or max(anchor["patch"]) == min(anchor["patch"])
            or type(anchor.get("z_m")) not in (float, int)
            or not 0.001 <= anchor["z_m"] < 10
        ):
            raise ValidationError("RGB-D anchor patch and depth must be valid")


def _check_model(model, k, size):
    if (
        not isinstance(model, dict)
        or model.get("model") != _MODEL
        or not isinstance(model.get("object_frame"), str)
        or not model["object_frame"]
        or not isinstance(model.get("views"), list)
        or not 1 <= len(model["views"]) <= _MAX_VIEWS
    ):
        raise ValidationError("Connect a trained LINEMOD model with 1–12 calibrated views")
    expected = np.asarray(model.get("camera_matrix"), np.float64)
    if (
        model.get("image_size") != list(size)
        or expected.shape != (3, 3)
        or not np.allclose(expected, k, rtol=1e-7, atol=1e-7)
    ):
        raise ValidationError(
            "LINEMOD view and search images must share calibrated intrinsics and dimensions"
        )
    settings = _modality_settings(model.get("settings"))
    for view in model["views"]:
        _check_view(view, size)
    if len({view["id"] for view in model["views"]}) != len(model["views"]):
        raise ValidationError("LINEMOD view names must be unique")
    return settings


def _extract_view(detector, color, depth, mask, gray, transform, name):
    identifier, (x, y, w, h) = detector.addTemplate(
        [_pad(color), _pad(depth, 65535)], _CLASS, _pad(mask)
    )
    if identifier < 0:
        return None
    x, y = x - _MARGIN, y - _MARGIN
    native = detector.getTemplates(_CLASS, identifier)
    templates = [
        {
            "width": t.width,
            "height": t.height,
            "pyramid_level": t.pyramid_level,
            "features": [[f.x, f.y, f.label] for f in t.features],
        }
        for t in native
    ]
    # Sample interior corners with enough texture and valid depth for geometry.
    support = mask.copy()
    support[:, : max(0, x)] = 0
    support[: max(0, y), :] = 0
    support[:, x + w + 1 :] = 0
    support[y + h + 1 :, :] = 0
    support = cv2.erode(
        support,
        np.ones((2 * _PATCH + 1,) * 2, np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support[depth == 65535] = 0
    corners = cv2.goodFeaturesToTrack(gray, 48, 0.01, 8, mask=support)
    anchors = []
    if corners is not None:
        for u, v in np.rint(corners.reshape(-1, 2)).astype(int):
            patch = gray[v - _PATCH : v + _PATCH + 1, u - _PATCH : u + _PATCH + 1]
            if patch.std() >= 4:
                anchors.append(
                    {
                        "offset": [int(u) - x, int(v) - y],
                        "patch": patch.ravel().tolist(),
                        "z_m": float(depth[v, u]) / 1000,
                    }
                )
    if len(anchors) < 8:
        return None
    return {
        "id": name,
        "bounds": [x, y, w, h],
        "object_to_camera": transform.tolist(),
        "templates": templates,
        "anchors": anchors,
    }


def _load_templates(settings, views):
    detector = _detector(settings)
    # The Python Template fields are read-only. Materialize validated records
    # through OpenCV's class loader in a process-owned temporary directory.
    native = {
        "class_id": _CLASS,
        "modalities": list(_MODALITIES),
        "pyramid_levels": len(_STEPS),
        "template_pyramids": [
            {
                "template_id": i,
                "templates": [
                    {
                        "width": t["width"],
                        "height": t["height"],
                        "pyramid_level": t["pyramid_level"],
                        "features": t["features"],
                    }
                    for t in view["templates"]
                ],
            }
            for i, view in enumerate(views)
        ],
    }
    with tempfile.TemporaryDirectory(prefix="dataflow-linemod-") as scratch:
        path = Path(scratch)
        (path / f"{_CLASS}.json").write_text(json.dumps(native), encoding="utf-8")
        # Escape percent signs in the root before adding the one class placeholder.
        detector.readClasses([_CLASS], str(path).replace("%", "%%") + "/%s.json")
    return detector


def _backproject(pixels, z, k):
    xy = (pixels - (k[0, 2], k[1, 2])) / (k[0, 0], k[1, 1])
    return np.column_stack((xy * z[:, None], z))


def _evaluate(match, view, gray, depth, mask, k, params):
    x, y = match.x - _MARGIN, match.y - _MARGIN
    u, v, w, h = view["bounds"]
    record = {
        "view": view["id"],
        "similarity_percent": float(match.similarity),
        "bounds": [x, y, w + 1, h + 1],
        "status": "outside_image",
    }
    s = int(params["refine_radius_px"])
    rows, cols = gray.shape
    if not (x >= -s and y >= -s and x + w < cols + s and y + h < rows + s):
        return record, None
    offsets, ref_depth, responses = [], [], []
    for anchor in view["anchors"]:
        dx, dy = anchor["offset"]
        cx, cy = x + dx, y + dy
        radius = s + _PATCH
        if cx < radius or cy < radius or cx + radius >= cols or cy + radius >= rows:
            continue
        window = gray[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        patch = np.asarray(anchor["patch"], np.uint8).reshape(2 * _PATCH + 1, -1)
        responses.append(cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED))
        offsets.append((dx, dy))
        ref_depth.append(anchor["z_m"])
    record["status"] = "insufficient_anchors"
    if len(offsets) < 8:
        return record, None
    response = np.median(responses, axis=0)
    py, px = np.unravel_index(response.argmax(), response.shape)
    x, y = int(x + px - s), int(y + py - s)
    correlation = float(response[py, px])
    record.update(bounds=[x, y, w + 1, h + 1], patch_correlation=correlation)
    if not (x >= 0 and y >= 0 and x + w < cols and y + h < rows):
        record["status"] = "outside_image"
        return record, None
    if correlation < float(params["minimum_patch_correlation"]):
        record["status"] = "appearance_mismatch"
        return record, None
    offsets = np.asarray(offsets)
    observed = offsets + [x, y]
    yy, xx = observed[:, 1], observed[:, 0]
    z = depth[yy, xx]
    valid = (z != 65535) & (mask[yy, xx] != 0)
    reference = _backproject(offsets[valid] + [u, v], np.asarray(ref_depth)[valid], k)
    observed = observed[valid]
    displacement = _backproject(observed, z[valid].astype(float) / 1000, k) - reference
    if not len(displacement):
        return record, None
    center = np.median(displacement, axis=0)
    tolerance = float(params["maximum_anchor_error_mm"]) / 1000
    inliers = np.linalg.norm(displacement - center, axis=1) <= tolerance
    fraction = float(inliers.sum() / len(view["anchors"]))
    record.update(anchor_inliers=int(inliers.sum()), anchor_inlier_fraction=fraction)
    if inliers.sum() < 8 or fraction < float(params["minimum_anchor_fraction"]):
        return record, None
    shift = displacement[inliers].mean(axis=0)
    predicted = reference[inliers] + shift
    if np.any(predicted[:, 2] <= 0):
        record["status"] = "projected_behind_camera"
        return record, None
    pixels = (predicted @ k.T)[:, :2] / predicted[:, 2, None]
    rms = float(np.sqrt(np.mean(np.sum((pixels - observed[inliers]) ** 2, axis=1))))
    record.update(reprojection_rms_px=rms)
    if rms > float(params["maximum_reprojection_rms_px"]):
        record["status"] = "reprojection_limit"
        return record, None
    transform = np.asarray(view["object_to_camera"], dtype=float).copy()
    transform[:3, 3] += shift
    record["status"] = "accepted"
    return record, transform


class _RGBDTemplates(Tool3D):
    subtype = "RGB-D pose templates"
    parameter_choices = {"alignment": ("same_frame", "pixel_grid")}
    parameter_help = {
        "alignment": "Require registered frames, or declare separately loaded grids aligned.",
    }
    input_help = {
        "image": "Undistorted uint8 image on the camera pixel grid.",
        "depth": "Metric camera-Z depth registered to Image.",
        "camera": "Pinhole intrinsics for the full undistorted RGB-D grid.",
        "mask": "Optional uint8 selection on this grid.",
        "region": "Optional region intersected with the mask.",
    }
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def captures(self, inputs):
        aligned = self.parameters["alignment"] == "pixel_grid"
        frames = _captures(self, inputs, aligned)
        return (*frames, selection_mask(inputs, inputs["image"], match_mask_frame=not aligned))


class TrainLINEMODPose(_RGBDTemplates):
    type_id, title = "3d.train_linemod_pose", "Train RGB-D pose template"
    inputs = (
        *_INPUTS,
        Port("pose", "data", visual_type="pose"),
        Port("previous_model", "data", required=False),
    )
    outputs = (Port("model", "data", required=False), Port("report", "data"))
    defaults = {"alignment": "same_frame", "view_name": "reference", **_SETTINGS}
    parameter_ranges = _SETTING_RANGES
    parameter_help = {
        **_RGBDTemplates.parameter_help,
        "view_name": "Unique name for this object orientation. Replaces a view with the same name.",
        "color_weak_threshold": "Discard weaker gradients when quantizing.",
        "color_strong_threshold": "Minimum color gradient magnitude for training features.",
        "color_features": "Fine-level gradient features. The coarse level uses half as many.",
        "depth_features": "Fine-level normal features. The coarse level uses half as many.",
        "normal_max_range_mm": "Ignore samples at or beyond this camera-Z distance.",
        "normal_max_difference_mm": "Maximum depth jump used to estimate a local normal.",
        "normal_extract_threshold": "Minimum distance from a change of quantized normal.",
    }
    input_help = {
        **_RGBDTemplates.input_help,
        "mask": "This and/or Region must isolate the known object for training.",
        "pose": "Verified object-to-camera pose in meters or millimeters for this view.",
        "previous_model": "Optional earlier views of this object on the same calibrated grid.",
    }
    flow_primary = ("model",)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if (
            not isinstance(p["view_name"], str)
            or not p["view_name"].strip()
            or len(p["view_name"]) > 64
        ):
            raise ValidationError("View name must have 1–64 characters", "view_name")
        _modality_settings({key: p[key] for key in _SETTINGS})

    def process(self, inputs):
        color, gray, z, k, size, optical, mask = self.captures(inputs)
        if mask is None:
            raise ValidationError("Supply a mask or region isolating the training object")
        frame, transform = _pose(inputs["pose"], optical)
        p = self.parameters
        settings = _modality_settings({key: p[key] for key in _SETTINGS})
        previous = inputs.get("previous_model")
        if previous is None:
            views = []
        else:
            if _check_model(previous, k, size) != settings or previous["object_frame"] != frame:
                raise ValidationError(
                    "Views in a model must share the object frame and modality settings"
                )
            views = [view for view in previous["views"] if view["id"] != p["view_name"]]
            if len(views) == _MAX_VIEWS:
                raise ValidationError("The LINEMOD model already has 12 views")
        native_depth = _depth_mm(z, settings["normal_max_range_mm"])
        view = _extract_view(
            _detector(settings), color, native_depth, mask, gray, transform, p["view_name"]
        )
        if view is None:
            return Result(
                {"model": NO_VALUE, "report": {"status": "insufficient_training_features"}},
                passed=False,
                fail_reasons=["linemod_insufficient_training_features"],
                message="Training needs gradients, depth normals and interior texture",
            )
        views.append(view)
        model = {
            "model": _MODEL,
            "camera_matrix": k.tolist(),
            "image_size": list(size),
            "object_frame": frame,
            "settings": settings,
            "views": views,
        }
        _check_model(model, k, size)
        report = {"status": "trained", "views": len(views), "anchors": len(view["anchors"])}
        drawings = {}
        if self.generate_graphics:
            x, y, w, h = view["bounds"]
            drawings["Training view"] = GraphicDrawing(
                (graphic_box(x, y, w + 1, h + 1, (75, 210, 240)),)
            )
        return Result(
            {"model": model, "report": report},
            message=f"Trained or retained {len(views)} views",
            drawings=drawings,
        )


class MatchLINEMODPose(_RGBDTemplates):
    type_id, title = "3d.match_linemod_pose", "Find RGB-D template pose"
    inputs = (*_INPUTS, Port("model", "data", required=False))
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("candidates", "data"),
        Port("report", "data"),
    )
    defaults = {
        "alignment": "same_frame",
        "trained_model": {},
        "minimum_similarity": 82.0,
        "max_candidates": 32,
        "refine_radius_px": 10,
        "minimum_patch_correlation": 0.65,
        "maximum_anchor_error_mm": 5.0,
        "minimum_anchor_fraction": 0.65,
        "maximum_reprojection_rms_px": 2.5,
    }
    parameter_ranges = {
        "minimum_similarity": NumericRange(50, 99),
        "max_candidates": NumericRange(1, 64, whole=True),
        "refine_radius_px": NumericRange(0, 24, whole=True),
        "minimum_patch_correlation": NumericRange(0.1, 1),
        "maximum_anchor_error_mm": NumericRange(0.5, 100),
        "minimum_anchor_fraction": NumericRange(0.25, 1),
        "maximum_reprojection_rms_px": NumericRange(0.1, 20),
    }
    parameter_help = {
        **_RGBDTemplates.parameter_help,
        "trained_model": "Reusable training output. A connected model takes precedence.",
        "minimum_similarity": "Minimum native LINEMOD similarity in percent.",
        "max_candidates": "Maximum native proposals to verify with appearance and metric depth.",
        "refine_radius_px": "Local patch search radius around the native grid match.",
        "minimum_patch_correlation": "Minimum median normalized color-patch correlation.",
        "maximum_anchor_error_mm": "Maximum 3D distance from the median anchor displacement.",
        "minimum_anchor_fraction": "Minimum fraction of trained anchors supporting the same shift.",
        "maximum_reprojection_rms_px": "Maximum return projection error at the inlier anchors.",
    }
    input_help = {
        **_RGBDTemplates.input_help,
        "model": "A trained bank from Train RGB-D pose template.",
    }
    flow_primary = ("pose",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._detector_cache = None

    def _cached_detector(self, settings, views):
        template_key = json.dumps([settings, [view["templates"] for view in views]], sort_keys=True)
        if self._detector_cache is None or self._detector_cache[0] != template_key:
            self._detector_cache = (template_key, _load_templates(settings, views))
        return self._detector_cache[1]

    def process(self, inputs):
        p = self.parameters
        color, gray, z, k, size, optical, mask = self.captures(inputs)
        model = inputs.get("model", p["trained_model"])
        settings = _check_model(model, k, size)
        if model["object_frame"] == optical:
            raise ValidationError("Object and search camera frame names must differ")
        views = model["views"]
        detector = self._cached_detector(settings, views)
        native_depth = _depth_mm(z, settings["normal_max_range_mm"])
        if mask is None:
            mask = np.full(gray.shape, 255, np.uint8)
        valid = np.where(native_depth != 65535, mask, 0).astype(np.uint8)
        proposals, _ = detector.match(
            [_pad(color), _pad(native_depth, 65535)],
            float(p["minimum_similarity"]),
            masks=[_pad(mask), _pad(valid)],
        )
        candidates, accepted, objects = [], [], []
        for i, proposal in enumerate(proposals[: int(p["max_candidates"])]):
            candidate, pose = _evaluate(
                proposal, views[proposal.template_id], gray, native_depth, mask, k, p
            )
            candidates.append(candidate)
            if pose is not None:
                rank = (
                    candidate["patch_correlation"],
                    candidate["similarity_percent"],
                    -candidate["reprojection_rms_px"],
                )
                accepted.append((rank, i, pose))
            if self.generate_graphics and candidate["status"] != "outside_image":
                color = (90, 215, 95) if pose is not None else (50, 70, 230)
                objects.append(graphic_box(*candidate["bounds"], color))

        pose = NO_VALUE
        best = max(accepted, default=None, key=lambda item: item[0])
        if best is not None:
            _, index, transform = best
            candidates[index]["selected"] = True
            pose = pose_record(transform, model["object_frame"], optical)
            if self.generate_graphics and transform[2, 3] > 0:
                axes = (
                    np.vstack((np.zeros(3), np.eye(3) * 0.025)) @ transform[:3, :3].T
                    + transform[:3, 3]
                )
                if np.all(axes[:, 2] > 0):
                    projected = (axes @ k.T)[:, :2] / axes[:, 2, None]
                    for endpoint, color in zip(
                        projected[1:], ((35, 50, 240), (50, 220, 55), (240, 90, 45)), strict=True
                    ):
                        objects.append(graphic_object("polyline", (projected[0], endpoint), color))
        found = best is not None
        status = "matched" if found else "unverified" if candidates else "no_proposals"
        report = {
            "status": status,
            "views": len(views),
            "proposals": len(proposals),
            "verified_candidates": len(accepted),
            "selected": None if best is None else candidates[index],
            "rotation_source": "trained_view",
        }
        return Result(
            {"pose": pose, "candidates": candidates, "report": report},
            passed=found,
            score=candidates[index]["similarity_percent"] / 100 if found else 0.0,
            event_data=report,
            drawings={"RGB-D candidates and pose": GraphicDrawing(tuple(objects))}
            if objects
            else {},
            fail_reasons=[] if found else [f"linemod_{status}"],
            message=f"Matched {candidates[index]['view']}"
            if found
            else "No verified RGB-D template pose",
        )
