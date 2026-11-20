"""Decode custom ArUco target layouts from synchronized capture files."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .fiducial_detection import (
    DICTIONARIES,
    MARKER_CHOICES,
    MARKER_OPTIONS,
    MARKER_RANGES,
    detector_parameters,
)
from .geometry3d_common import image_size
from .multicamera_calibration import (
    _MAX_CAMERAS,
    _MAX_CAPTURES,
    _MAX_FILE_BYTES,
    _MAX_POINTS,
    OBSERVATIONS,
    _names,
    _observations,
    _optics,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

MANIFEST = "coded_rig_capture_manifest_v1"


def _load_json(path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ValidationError("Capture manifest exceeds 8 MiB", "path")
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("model") != MANIFEST:
        raise ValidationError(f"Choose a {MANIFEST} JSON record", "path")
    return record


def _layout(target):
    if not isinstance(target, dict) or target.get("model") != "fiducial_target_v1":
        raise ValidationError("Supply a fiducial_target_v1 layout")
    name, units = target.get("dictionary"), target.get("units")
    if name not in DICTIONARIES or units not in ("m", "mm"):
        raise ValidationError("Custom target needs an OpenCV dictionary and metric units")
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    ids, raw = target.get("marker_ids"), target.get("marker_corners_xyz")
    if (
        not isinstance(ids, list)
        or not 2 <= len(ids) <= _MAX_POINTS // 4
        or any(type(i) is not int or not 0 <= i < len(dictionary.bytesList) for i in ids)
        or len(set(ids)) != len(ids)
        or not isinstance(raw, list)
        or len(raw) != len(ids)
    ):
        raise ValidationError(
            "Target requires 2–32 distinct marker IDs and their corner coordinates"
        )
    corners = np.asarray(raw, dtype=np.float64)
    if (
        corners.shape != (len(ids), 4, 3)
        or not np.isfinite(corners).all()
        or np.max(np.abs(corners)) > 1e6
        or np.any(corners[:, :, 2] != 0)
        or len(np.unique(corners.reshape(-1, 3), axis=0)) != corners.size // 3
        or not all(cv2.isContourConvex(quad[:, :2].astype(np.float32)) for quad in corners)
    ):
        raise ValidationError("Target corners must form distinct, convex markers in the XY plane")
    return dictionary, units, ids, corners


def _capture_table(manifest, frames, captures):
    sizes, paths = manifest.get("image_sizes"), manifest.get("images")
    if not isinstance(sizes, list) or len(sizes) != len(frames):
        raise ValidationError("Declare one image size per camera")
    sizes = [image_size(size) for size in sizes]
    if max(w * h for w, h in sizes) > 8_388_608:
        raise ValidationError("Capture images must have at most 8,388,608 pixels")
    if (
        not isinstance(paths, list)
        or len(paths) != len(frames)
        or any(not isinstance(row, list) or len(row) != len(captures) for row in paths)
        or any(
            path is not None and (not isinstance(path, str) or not path.strip())
            for row in paths
            for path in row
        )
    ):
        raise ValidationError(
            "The manifest needs a camera-by-capture table of image paths or nulls"
        )
    return sizes, paths


def _observe(path, detector, size, ids, minimum, preview):
    with path.open("rb") as stream:
        raw = stream.read(24 * 1024 * 1024 + 1)
    if len(raw) > 24 * 1024 * 1024:
        raise ValidationError(f"Capture file exceeds 24 MiB: {path.name}")
    image = cv2.imdecode(
        np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None or image.shape != size[::-1]:
        raise ValidationError(f"Capture image does not match declared resolution: {path.name}")
    corners, decoded, _ = detector.detectMarkers(image)
    detections = (
        []
        if decoded is None
        else [
            (int(tag), xy.reshape(4, 2))
            for tag, xy in zip(decoded.ravel(), corners, strict=True)
            if tag in ids
        ]
    )
    counts = Counter(tag for tag, _ in detections)
    result = {"marker_ids": sorted(counts)}
    duplicates = [tag for tag in sorted(counts) if counts[tag] > 1]
    if duplicates:
        result.update(status="duplicate_markers", duplicate_ids=duplicates)
    elif len(counts) < minimum:
        result["status"] = "below_marker_minimum"
    else:
        by_id = dict(detections)
        # A missing marker occupies four null slots in the canonical corner order.
        pixels = [
            xy.tolist() if xy is not None else None
            for tag in ids
            for xy in by_id.get(tag, (None,) * 4)
        ]
        overlay = None
        if preview:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(
                overlay,
                [xy.reshape(1, 4, 2) for _, xy in detections],
                np.asarray([tag for tag, _ in detections], dtype=np.int32),
            )
            w, h = size
            if w > 640:
                overlay = cv2.resize(
                    overlay, (640, max(1, round(640 * h / w))), interpolation=cv2.INTER_AREA
                )
        return pixels, {**result, "status": "observed"}, overlay
    return None, result, None


class LoadCodedRigObservations(Block):
    type_id, title, category = "input.coded_rig_observations", "Load coded rig captures", "Input"
    block_type, subtype = "Input", "Calibration"
    inputs = (Port("target", "data", required=False),)
    outputs = (
        Port("observations", "data", required=False),
        Port("camera_models", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "path": "",
        "minimum_markers": 2,
        **{k: value for k, value in MARKER_OPTIONS.items() if k != "border_bits"},
    }
    parameter_choices = {"corner_refinement": MARKER_CHOICES["corner_refinement"]}
    parameter_ranges = {
        **{key: limits for key, limits in MARKER_RANGES.items() if key != "border_bits"},
        "minimum_markers": NumericRange(2, _MAX_POINTS // 4, whole=True),
    }
    parameter_help = {
        "path": "JSON camera-by-capture manifest. Relative paths use the project directory.",
        "minimum_markers": "Minimum number of target IDs detected in a view.",
    }
    input_help = {"target": "Optional surveyed layout. Overrides the target in the manifest."}
    project_path_parameters = ("path",)
    flow_primary = ("observations",)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        detector_parameters({**p, "border_bits": 1})

    def process(self, inputs):
        p = self.parameters
        if not p["path"]:
            raise ValidationError("Choose a coded camera capture manifest", "path")
        path = Path(p["path"])
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save the project before opening a relative manifest", "path")
            path = self.project_dir / path
        manifest = _load_json(path)
        frames = _names(manifest.get("camera_frames"), "Camera frames", 3, _MAX_CAMERAS)
        captures = _names(manifest.get("capture_ids"), "Capture IDs", 1, _MAX_CAPTURES)
        sizes, paths = _capture_table(manifest, frames, captures)
        target = inputs.get("target", manifest.get("target"))
        dictionary, units, ids, corners = _layout(target)
        minimum = int(p["minimum_markers"])
        if minimum > len(ids):
            raise ValidationError("Minimum markers exceeds the target size", "minimum_markers")
        settings = detector_parameters({**p, "border_bits": target.get("border_bits", 1)})
        detector = cv2.aruco.ArucoDetector(dictionary, settings)
        images, details, previews = [], [], {}
        for camera, size, camera_paths in zip(frames, sizes, paths, strict=True):
            row, reports = [], []
            for capture, image_path in zip(captures, camera_paths, strict=True):
                if image_path is None:
                    pixels, status, overlay = None, {"status": "missing_image"}, None
                else:
                    file = path.parent / image_path
                    pixels, status, overlay = _observe(
                        file,
                        detector,
                        size,
                        ids,
                        minimum,
                        self.capture_graphics and camera not in previews,
                    )
                if overlay is not None:
                    previews[camera] = overlay
                row.append(pixels)
                reports.append({"capture_id": capture, **status})
            images.append(row)
            details.append({"camera": camera, "views": reports})

        use = [
            i
            for i, column in enumerate(zip(*images, strict=True))
            if any(v is not None for v in column)
        ]
        absent = [
            camera
            for camera, row in zip(frames, images, strict=True)
            if all(v is None for v in row)
        ]
        report = {
            "model": "coded_rig_detection_v1",
            "views": details,
            "dropped_captures": [capture for i, capture in enumerate(captures) if i not in use],
            "cameras_without_observations": absent,
        }
        if absent or not use:
            return Result(
                {"observations": NO_VALUE, "camera_models": NO_VALUE, "report": report},
                graphics=previews,
                passed=False,
                fail_reasons=["coded_rig_incomplete"],
                message="Some cameras have no usable coded target views",
            )
        points = [[row[i] for i in use] for row in images]
        record = {
            "model": OBSERVATIONS,
            "units": units,
            "camera_frames": list(frames),
            "image_sizes": [list(size) for size in sizes],
            "capture_ids": [captures[i] for i in use],
            "object_points": corners.reshape(-1, 3).tolist(),
            "visibility": [[int(view is not None) for view in row] for row in points],
            "image_points": points,
        }
        parsed = _observations(record)
        models = manifest.get("camera_models", NO_VALUE)
        if models is not NO_VALUE:
            if (
                not isinstance(models, list)
                or not models
                or not isinstance(models[0], dict)
                or models[0].get("lens") not in ("pinhole", "fisheye")
            ):
                raise ValidationError("Use ordered calibrated pinhole or fisheye camera records")
            _optics(models, parsed, models[0]["lens"])
        return Result(
            {"observations": record, "camera_models": models, "report": report},
            graphics=previews,
            message=f"Decoded {int(parsed.visible.sum())} views across {len(use)} captures",
        )
