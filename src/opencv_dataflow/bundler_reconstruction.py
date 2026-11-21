"""Read Bundler v0.3 cameras, colored structure and observation provenance."""

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera_record,
    cloud_graphics,
    image_size,
    pose_matrix,
    pose_record,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .pointclouds import PointCloud

_MAX_BYTES = 32 * 1024**2
_MAX_CAMERAS = 2048
_MAX_POINTS = 100_000
_MAX_OBSERVATIONS = 500_000
_MAX_NUMBER = 1e12
_BUNDLE_AXES = np.diag([1.0, -1.0, -1.0])
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INT = re.compile(r"[+-]?[0-9]+")
_TOKEN = re.compile(r"\S+")
_ZERO = np.zeros(3, np.float64)


class _BundleTokens:
    def __init__(self, body):
        self.stream = (match.group() for match in _TOKEN.finditer(body))

    def _next(self, where, pattern):
        token = next(self.stream, None)
        if token is None:
            raise ValidationError(f"Bundler file ended while reading {where}", "path")
        if len(token) > 80 or pattern.fullmatch(token) is None:
            raise ValidationError(f"Invalid numeric token while reading {where}", "path")
        return token

    def integer(self, where, minimum, maximum):
        token = self._next(where, _INT)
        value = int(token)
        if not minimum <= value <= maximum:
            raise ValidationError(f"{where} must be in [{minimum}, {maximum}]", "path")
        return value

    def vector(self, count, where):
        values = np.array([float(self._next(where, _FLOAT)) for _ in range(count)], np.float64)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > _MAX_NUMBER:
            raise ValidationError(f"Nonfinite or out-of-range {where}", "path")
        return values


@dataclass(frozen=True)
class _Camera:
    focal: float
    radial: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def center(self):
        return -self.rotation.T @ self.translation

    def intrinsics(self, size=None):
        cx, cy = (0, 0) if size is None else (size[0] / 2, size[1] / 2)
        return np.array([[self.focal, 0, cx], [0, self.focal, cy], [0, 0, 1]])

    @property
    def distortion(self):
        return np.array([*self.radial, 0.0, 0.0, 0.0], np.float64)


@dataclass(frozen=True)
class _Bundle:
    cameras: list
    points: np.ndarray
    colors: np.ndarray
    observations: list
    sha256: str


def _read_bundle(path):
    try:
        with path.open("rb") as source:
            raw = source.read(_MAX_BYTES + 1)
        if not 0 < len(raw) <= _MAX_BYTES:
            raise ValidationError("Bundler input must be nonempty and at most 32 MiB", "path")
        header, _, body = raw.decode("utf-8-sig").partition("\n")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Could not read Bundler file: {exc}", "path") from exc
    if header.strip() != "# Bundle file v0.3":
        raise ValidationError(
            "Unsupported reconstruction format. Choose a Bundler v0.3 file.", "path"
        )

    data = _BundleTokens(body)
    n_cameras = data.integer("Camera count", 1, _MAX_CAMERAS)
    n_points = data.integer("Point count", 0, _MAX_POINTS)
    cameras = []
    for index in range(n_cameras):
        focal, *radial = data.vector(3, f"camera {index} intrinsics")
        rotation = data.vector(9, f"camera {index} rotation").reshape(3, 3)
        translation = data.vector(3, f"camera {index} translation")
        if focal < 0:
            raise ValidationError(f"Camera {index} has a negative focal length", "path")
        if focal == 0:
            # Bundler keeps an input camera slot even when it is unregistered.
            cameras.append(None)
            continue
        rotation, translation = _BUNDLE_AXES @ rotation, _BUNDLE_AXES @ translation
        try:
            pose_matrix(rotation, translation)
        except ValueError as exc:
            raise ValidationError(f"Camera {index} requires a proper rotation", "path") from exc
        cameras.append(_Camera(float(focal), np.asarray(radial), rotation, translation))

    points = np.empty((n_points, 3), np.float64)
    colors = np.empty((n_points, 3), np.uint8)
    observations = [[] for _ in cameras]
    seen_keys = [set() for _ in cameras]
    total = 0
    for index in range(n_points):
        points[index] = data.vector(3, f"point {index} XYZ")
        colors[index] = [data.integer(f"Point {index} RGB sample", 0, 255) for _ in range(3)]
        count = data.integer(f"Point {index} observation count", 0, n_cameras)
        total += count
        if total > _MAX_OBSERVATIONS:
            raise ValidationError("Bundler input exceeds 500,000 observations", "path")
        seen_cameras = set()
        for _ in range(count):
            camera = data.integer(f"Point {index} camera index", 0, n_cameras - 1)
            key = data.integer(f"Point {index} feature index", 0, 2**31 - 1)
            x, y = data.vector(2, f"point {index} observed pixel")
            if cameras[camera] is None:
                raise ValidationError(
                    f"Point {index} refers to unregistered camera {camera}", "path"
                )
            if camera in seen_cameras or key in seen_keys[camera]:
                raise ValidationError(f"Duplicate Bundler observation at point {index}", "path")
            seen_cameras.add(camera)
            seen_keys[camera].add(key)
            observations[camera].append((index, key, x, y))
    if next(data.stream, None) is not None:
        raise ValidationError("Unexpected trailing data after the declared Bundler points", "path")
    return _Bundle(cameras, points, colors, observations, sha256(raw).hexdigest())


def _camera_sizes(value, count=None):
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_CAMERAS:
        raise ValidationError(
            "Use [width, height] or an ordered list of camera sizes", "image_sizes"
        )
    if not value:
        return [] if count is None else [None] * count
    try:
        if len(value) == 2 and all(type(number) is int for number in value):
            size = image_size(list(value))
            return [size] if count is None else [size] * count
        sizes = [None if item is None else image_size(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Camera sizes need positive whole widths and heights", "image_sizes"
        ) from exc
    if count is not None and len(sizes) != count:
        raise ValidationError("Provide one size or null per Bundler camera slot", "image_sizes")
    return sizes


def _scale(bundle, p):
    if p["scale_mode"] == "unitless":
        return 1.0, "unitless", {"mode": "unitless", "meters_per_bundle_unit": None}
    first, last = p["baseline_camera_indices"]
    try:
        a, b = bundle.cameras[first], bundle.cameras[last]
    except IndexError as exc:
        raise ValidationError(
            "Baseline camera indices are outside the bundle", "baseline_camera_indices"
        ) from exc
    if a is None or b is None:
        raise ValidationError("Choose two registered baseline cameras", "baseline_camera_indices")
    raw_distance = float(np.linalg.norm(a.center - b.center))
    if raw_distance <= 0:
        raise ValidationError(
            "The selected cameras have coincident centers", "baseline_camera_indices"
        )
    measured = float(p["baseline_length_m"])
    scale = measured / raw_distance
    if not np.isfinite(scale) or not 1e-12 <= scale <= 1e12:
        raise ValidationError(
            "Measured baseline implies an unsupported numeric scale", "baseline_length_m"
        )
    return (
        scale,
        "m",
        {
            "mode": "known_baseline",
            "camera_indices": [first, last],
            "bundle_baseline": raw_distance,
            "measured_baseline_m": measured,
            "meters_per_bundle_unit": scale,
        },
    )


def _audit(camera, rows, points, size):
    values = np.asarray(rows, np.float64).reshape(-1, 4)
    ids = values[:, 0].astype(np.intp)
    observed = values[:, 2:] * [1, -1]
    xyz = points[ids] @ camera.rotation.T + camera.translation
    front = xyz[:, 2] > 0
    errors = np.full(len(rows), np.nan, np.float64)
    if front.any():
        projected = cv2.projectPoints(
            xyz[front], _ZERO, _ZERO, camera.intrinsics(), camera.distortion
        )[0].reshape(-1, 2)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = projected - observed[front]
            errors[front] = np.hypot(delta[:, 0], delta[:, 1])
    scored = np.isfinite(errors) & (errors <= _MAX_NUMBER)
    errors[~scored] = np.nan
    centered = values[:, 2:]
    pixels = None if size is None else observed + np.asarray(size) / 2
    outside = (
        None
        if size is None
        else int(np.count_nonzero(np.any((pixels < 0) | (pixels >= size), axis=1)))
    )
    metrics = {
        "observations": len(rows),
        "scored_observations": int(scored.sum()),
        "nonpositive_depth_observations": int(np.count_nonzero(~front)),
        "invalid_projection_observations": int(np.count_nonzero(front & ~scored)),
        "outside_image_observations": outside,
        "rms_px": float(np.sqrt(np.mean(errors[scored] ** 2))) if scored.any() else None,
        "maximum_error_px": float(errors[scored].max()) if scored.any() else None,
    }
    return metrics, centered, pixels, errors


class ImportBundlerReconstruction(Tool3D):
    type_id, title, category = (
        "input.bundler_reconstruction",
        "Import Bundler reconstruction",
        "Input",
    )
    block_type, subtype = "Input", "Reconstruction"
    inputs = (Port("image_sizes", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("cloud", "cloud"),
        Port("cameras", "data"),
        Port("camera", "data", required=False),
        Port("world_to_camera", "data", required=False, visual_type="pose"),
        Port("camera_to_world", "data", required=False, visual_type="pose"),
        Port("observations", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "path": "",
        "world_frame": "bundler_world",
        "image_sizes": [],
        "camera_index": 0,
        "scale_mode": "unitless",
        "baseline_camera_indices": [0, 1],
        "baseline_length_m": 1.0,
        "maximum_reprojection_error_px": 0.0,
    }
    project_path_parameters = ("path",)
    parameter_choices = {"scale_mode": ("unitless", "known_baseline")}
    parameter_ranges = {
        "camera_index": NumericRange(0, _MAX_CAMERAS - 1, whole=True),
        "baseline_length_m": NumericRange(1e-9, 100000),
        "maximum_reprojection_error_px": NumericRange(0, 10000),
    }
    parameter_help = {
        "path": "Bundler v0.3 bundle file, using a project-relative or absolute path.",
        "world_frame": "Name for the original reconstruction's coordinate axes.",
        "image_sizes": "Common [width, height], or one size or null per original camera slot.",
        "camera_index": "Zero-based slot for direct camera, pose and observation outputs.",
        "scale_mode": "Preserve unitless scale or recover meters from a measured camera baseline.",
        "baseline_camera_indices": "Two registered camera indices for the known scale baseline.",
        "baseline_length_m": "Measured optical-center separation in meters.",
        "maximum_reprojection_error_px": "Imported reprojection limit. Zero reports errors only.",
    }
    input_help = {"image_sizes": "Optional runtime override for the Image Sizes setting."}
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Choose a Bundler file path", "path")
        name = p["world_frame"]
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64:
            raise ValidationError("Use a world frame name of 1 to 64 characters", "world_frame")
        _camera_sizes(p["image_sizes"])
        pair = p["baseline_camera_indices"]
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(index) is not int or not 0 <= index < _MAX_CAMERAS for index in pair)
            or pair[0] == pair[1]
        ):
            raise ValidationError(
                "Use two different nonnegative camera indices", "baseline_camera_indices"
            )

    def process(self, inputs):
        p = self.parameters
        if not p["path"].strip():
            raise ValidationError("Choose a Bundler v0.3 file", "path")
        path = Path(p["path"])
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError(
                    "Save or open the project before using a relative Bundler path", "path"
                )
            path = self.project_dir / path
        bundle = _read_bundle(path)
        sizes = _camera_sizes(inputs.get("image_sizes", p["image_sizes"]), len(bundle.cameras))
        selected = int(p["camera_index"])
        if selected >= len(bundle.cameras):
            raise ValidationError("Selected camera index is outside the bundle", "camera_index")
        scale, units, scale_status = _scale(bundle, p)
        world = p["world_frame"].strip()
        xyz = bundle.points * scale
        if not np.isfinite(xyz).all() or np.any(np.abs(xyz) > _MAX_NUMBER):
            raise ValidationError("Scaled points exceed the supported coordinate range")
        cloud = PointCloud(
            xyz,
            units,
            world,
            colors=bundle.colors,
            metadata={"bundler": {"sha256": bundle.sha256, "scale": scale_status}},
            attributes={"source_index": np.arange(len(xyz), dtype=np.int32)},
        )

        outputs = {
            "cloud": cloud,
            "cameras": [],
            **dict.fromkeys(
                ("camera", "world_to_camera", "camera_to_world", "observations"), NO_VALUE
            ),
        }
        count = scored = behind = nonfinite = 0
        maximum = squared = 0.0
        registered = 0
        selected_frame = None
        for index, (camera, size, rows) in enumerate(
            zip(bundle.cameras, sizes, bundle.observations, strict=True)
        ):
            record = {
                "index": index,
                "registered": camera is not None,
                "image_size": None if size is None else list(size),
            }
            outputs["cameras"].append(record)
            if camera is None:
                continue
            registered += 1
            name = f"{world}:camera:{index}"
            offset, center = camera.translation * scale, camera.center * scale
            if np.any(np.abs(offset) > _MAX_NUMBER) or np.any(np.abs(center) > _MAX_NUMBER):
                raise ValidationError(
                    f"Scaled camera {index} exceeds the supported coordinate range"
                )
            transform = pose_matrix(camera.rotation, offset)
            inverse = pose_matrix(camera.rotation.T, center)
            model = (
                camera_record(camera.intrinsics(size), camera.distortion, size, units, name)
                if size is not None
                else None
            )
            world_to_camera = pose_record(transform, world, name, units)
            camera_to_world = pose_record(inverse, name, world, units)
            audit, centered, pixels, errors = _audit(camera, rows, bundle.points, size)
            record.update(
                focal_px=camera.focal,
                radial_distortion=camera.radial.tolist(),
                camera=model,
                world_to_camera=world_to_camera,
                camera_to_world=camera_to_world,
                **audit,
            )
            count += audit["observations"]
            scored += audit["scored_observations"]
            behind += audit["nonpositive_depth_observations"]
            nonfinite += audit["invalid_projection_observations"]
            finite = errors[np.isfinite(errors)]
            if len(finite):
                maximum = max(maximum, float(finite.max()))
                squared += float(np.dot(finite, finite))
            if index == selected:
                outputs.update(
                    camera=model if model is not None else NO_VALUE,
                    world_to_camera=world_to_camera,
                    camera_to_world=camera_to_world,
                    observations={
                        "model": "bundler_observations_v1",
                        "camera_index": index,
                        "point_indices": [row[0] for row in rows],
                        "feature_indices": [row[1] for row in rows],
                        "bundler_centered_xy": centered.tolist(),
                        "image_points": None if pixels is None else pixels.tolist(),
                        "reprojection_errors_px": [
                            float(err) if np.isfinite(err) else None for err in errors
                        ],
                    },
                )
                if size is not None:
                    selected_frame = frame(size, f"{self.id}:bundler:{bundle.sha256[:12]}:{index}")

        limit = float(p["maximum_reprojection_error_px"])
        report = {
            "model": "bundler_import_v1",
            "file": path.name,
            "sha256": bundle.sha256,
            "frame": world,
            "units": units,
            "scale": scale_status,
            "camera_slots": len(bundle.cameras),
            "registered_cameras": registered,
            "camera_index": selected,
            "points": len(cloud.points),
            "observations": count,
            "scored_observations": scored,
            "rms_px": (squared / scored) ** 0.5 if scored else None,
            "maximum_error_px": maximum if scored else None,
            "nonpositive_depth_observations": behind,
            "invalid_projection_observations": nonfinite,
        }
        fail = []
        if not registered or not len(xyz):
            fail.append("bundler_no_registered_structure")
        if limit and (maximum > limit or behind or nonfinite):
            fail.append("bundler_reprojection_limit")
        return Result(
            {**outputs, "report": report},
            payloads=(
                {"camera": {"spatial": selected_frame}, "observations": {"spatial": selected_frame}}
                if selected_frame
                else {}
            ),
            graphics=cloud_graphics(self, cloud) if len(cloud.points) else {},
            passed=not fail,
            fail_reasons=fail,
            message=f"Imported {len(cloud.points)} points and {registered} cameras ({units})",
        )
