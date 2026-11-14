"""Finite physical tool/bin settings and explicit rectified first-return camera contracts."""

from dataclasses import asdict, dataclass, field, replace
from numbers import Real

import numpy as np

MAX_PIXELS = 2_000_000


def number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def label(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 64 or value != value.strip()
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Coordinate frames need 1..64 plain characters")
    return value


def vector(value):
    data = np.array(value, float)
    if data.shape != (3,) or not np.isfinite(data).all() or np.any(abs(data) > 1e9):
        raise ValueError("Use three finite coordinates within +/-1e9 units")
    return tuple(data.tolist())


def rigid(value):
    data = np.array(value, float, copy=True)
    if (data.shape != (4, 4) or not np.isfinite(data).all() or np.any(abs(data[:3, 3]) > 1e9)
            or not np.allclose(data[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(data[:3, :3].T @ data[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(data[:3, :3]), 1., atol=1e-5, rtol=0)):
        raise ValueError("Bin-to-camera transform must be a proper rigid 4x4 matrix")
    u, _, vt = np.linalg.svd(data[:3, :3])
    data[:3, :3] = u @ vt
    data.flags.writeable = False
    return data


@dataclass(frozen=True)
class Camera:
    matrix: tuple
    image_size: tuple
    frame: str = "camera"

    def __post_init__(self):
        k = np.asarray(self.matrix, float)
        size = tuple(self.image_size)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or np.any(abs(k) > 1e7) or
                not 1e-6 <= k[0, 0] <= 1e7 or not 1e-6 <= k[1, 1] <= 1e7 or
                not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0) or
                abs(k[0, 1]) > 1e-12 or abs(k[1, 0]) > 1e-12):
            raise ValueError("Use rectified pinhole intrinsics with positive focal lengths")
        if (len(size) != 2 or any(type(v) is not int or not 2 <= v <= 16384 for v in size)
                or size[0] * size[1] > MAX_PIXELS):
            raise ValueError("Camera image size must have 2..16384 sides and <=2 million pixels")
        label(self.frame)
        object.__setattr__(self, "matrix", tuple(tuple(float(v) for v in row) for row in k))
        object.__setattr__(self, "image_size", size)

    @classmethod
    def from_record(cls, value):
        if not isinstance(value, dict) or value.get("model") != "camera_model_v1":
            raise ValueError("Connect an explicit camera_model_v1 record")
        if value.get("lens", "pinhole") != "pinhole" or np.any(value.get("distortion", [])):
            raise ValueError("Rectify distorted camera images before bin picking")
        return cls(value.get("camera_matrix", value.get("matrix")), value["image_size"],
                   value.get("frame", "camera"))

    def record(self, units):
        return {"model": "camera_model_v1", "camera_matrix": self.matrix,
                "image_size": self.image_size, "frame": self.frame, "units": units,
                "lens": "pinhole", "distortion": []}


@dataclass(frozen=True)
class SuctionSettings:
    cup_radius: float = 5.
    body_radius: float = 6.
    body_length: float = 25.
    pregrasp_distance: float = 30.
    tcp_offset: float = 0.
    surface_tolerance: float = 1.
    clearance: float = 1.
    min_pick_height: float = 5.
    max_tilt_degrees: float = 25.
    min_contact_fraction: float = .95
    overlap_threshold: float = .2
    max_candidates: int = 256
    max_picks: int = 10
    workers: int = 1

    def __post_init__(self):
        for name in ("cup_radius", "body_radius", "body_length", "pregrasp_distance",
                     "surface_tolerance", "clearance", "min_pick_height"):
            object.__setattr__(self, name, number(getattr(self, name), name, 1e-8, 1e6))
        object.__setattr__(self, "tcp_offset", number(self.tcp_offset, "TCP offset", -1e6, 1e6))
        if self.body_radius < self.cup_radius:
            raise ValueError("The conservative body radius must cover the cup radius")
        if self.surface_tolerance >= min(self.cup_radius, self.body_length / 2,
                                          self.pregrasp_distance / 2):
            raise ValueError("Contact tolerance must be below cup radius and half approach/body")
        for name, low, high in (("max_tilt_degrees", 1., 80.), ("min_contact_fraction", .5, 1.),
                                ("overlap_threshold", 0., 1.)):
            object.__setattr__(self, name, number(getattr(self, name), name, low, high))
        for name, high in (("max_candidates", 2048), ("max_picks", 100), ("workers", 8)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be an integer from 1 to {high}")

    @classmethod
    def for_units(cls, units):
        if units not in ("mm", "m"):
            raise ValueError("Declare mm or m units")
        settings = cls()
        names = ("cup_radius", "body_radius", "body_length", "pregrasp_distance", "tcp_offset",
                 "surface_tolerance", "clearance", "min_pick_height")
        return (settings if units == "mm" else
                replace(settings, **{name: getattr(settings, name) * .001 for name in names}))


def depth_array(value, camera):
    raw = np.asarray(value)
    if raw.shape != camera.image_size[::-1] or raw.dtype not in (np.uint16, np.float32, np.float64):
        raise ValueError("Depth must be a numeric plane matching the retained camera image size")
    data = np.array(raw, np.float64, order="C", copy=True)
    data.flags.writeable = False
    return data


@dataclass(frozen=True)
class Recipe:
    camera: Camera
    reference_depth: np.ndarray = field(repr=False, compare=False)
    bin_to_camera: np.ndarray = field(repr=False, compare=False)
    minimum: tuple
    maximum: tuple
    units: str = "mm"
    frame: str = "bin"
    depth_scale: float = 1.
    settings: SuctionSettings = field(default_factory=SuctionSettings)
    reference_rgb: np.ndarray | None = field(default=None, repr=False, compare=False)
    reference_instances: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.units not in ("mm", "m") or not isinstance(self.camera, Camera):
            raise ValueError("Recipe requires mm/m units and a Camera")
        label(self.frame)
        minimum, maximum = vector(self.minimum), vector(self.maximum)
        if np.any(np.asarray(minimum) >= maximum):
            raise ValueError("Bin maximum must exceed minimum on each axis; +Z is up")
        if not isinstance(self.settings, SuctionSettings):
            raise ValueError("Recipe requires SuctionSettings")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "depth_scale", number(self.depth_scale, "Depth scale", 1e-12, 1e6))
        object.__setattr__(self, "reference_depth", depth_array(self.reference_depth, self.camera))
        object.__setattr__(self, "bin_to_camera", rigid(self.bin_to_camera))
        if self.reference_rgb is not None:
            raw = np.asarray(self.reference_rgb)
            if raw.shape != (*self.camera.image_size[::-1], 3) or raw.dtype != np.uint8:
                raise ValueError("Reference appearance requires matching uint8 RGB pixels")
            image = np.array(raw, copy=True, order="C")
            image.flags.writeable = False
            object.__setattr__(self, "reference_rgb", image)
        if self.reference_instances is not None:
            raw = np.asarray(self.reference_instances)
            if (raw.shape != self.camera.image_size[::-1] or raw.dtype.kind not in "ui"
                    or np.any(raw < 0) or np.any(raw > 2**31 - 1)):
                raise ValueError("Reference instances require matching nonnegative integer labels")
            labels = np.array(raw, np.int32, copy=True, order="C")
            labels.flags.writeable = False
            object.__setattr__(self, "reference_instances", labels)

    def configuration(self):
        return {"camera": asdict(self.camera), "bin_to_camera": self.bin_to_camera.tolist(),
                "minimum": self.minimum, "maximum": self.maximum, "units": self.units,
                "frame": self.frame, "depth_scale": self.depth_scale,
                "settings": asdict(self.settings)}


def floor_frame(points):
    data = np.asarray(points, float)
    if data.ndim != 2 or data.shape[1] != 3 or not 3 <= len(data) <= 1024:
        raise ValueError("Pick 3..1024 spread-out points on the bin floor")
    if not np.isfinite(data).all():
        raise ValueError("Floor picks must be finite")
    origin = data.mean(0)
    _, singular, vectors = np.linalg.svd(data - origin, full_matrices=False)
    if singular[1] <= max(singular[0] * 1e-6, 1e-9):
        raise ValueError("Floor picks must span a surface, not a line")
    normal = vectors[-1]
    if normal @ -origin < 0:
        normal = -normal
    x = np.array([1., 0., 0.]) - normal * normal[0]
    if np.linalg.norm(x) < .1:
        x = np.cross([0, 1, 0], normal)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(normal, x), normal])
    pose[:3, 3] = origin
    return rigid(pose), float(np.sqrt(np.mean(((data - origin) @ normal)**2)))
