"""Typed point clouds: finite XYZ, explicit units/frames, optional color/normals/pixels."""

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    units: str = "m"
    frame: str = "camera"
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    pixels: np.ndarray | None = None
    source_size: tuple | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, np.float32)
        if self.points.ndim != 2:
            raise ValueError("Point clouds require an Nx3 XYZ array")
        n = len(self.points)
        if self.points.shape != (n, 3) or n > 2_000_000 or not np.isfinite(self.points).all():
            raise ValueError("Point clouds require 0..2,000,000 finite XYZ points")
        if (
            self.units not in ("m", "mm", "unitless")
            or not isinstance(self.frame, str)
            or not self.frame
        ):
            raise ValueError("Clouds require m/mm/unitless units and a named coordinate frame")
        for name in ("colors", "normals", "pixels"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value)
            width = 2 if name == "pixels" else 3
            if value.shape != (n, width) or not np.isfinite(value).all():
                raise ValueError(f"Cloud {name} do not match point count")
            if name in ("colors", "pixels") and np.any(value != np.rint(value)):
                raise ValueError(f"Cloud {name} must be integers")
            if name == "colors" and (np.any(value < 0) or np.any(value > 255)):
                raise ValueError("Cloud colors must be RGB bytes")
            if name == "pixels":
                if (
                    self.source_size is None
                    or len(self.source_size) != 2
                    or any(type(v) is not int or v < 1 for v in self.source_size)
                    or np.any(value < 0)
                    or np.any(value >= np.asarray(self.source_size))
                ):
                    raise ValueError("Cloud pixels must lie inside the declared source image")
            if (
                name == "normals"
                and n
                and not np.allclose(np.linalg.norm(value, axis=1), 1, atol=0.01)
            ):
                raise ValueError("Cloud normals must be unit vectors")
            setattr(
                self,
                name,
                np.ascontiguousarray(
                    value,
                    np.uint8 if name == "colors" else np.int32 if name == "pixels" else np.float32,
                ),
            )
        from .payloads import validate_payload

        validate_payload(self.metadata)

    @property
    def nbytes(self):
        return sum(
            v.nbytes for v in (self.points, self.colors, self.normals, self.pixels) if v is not None
        )

    def subset(self, selected):
        return PointCloud(
            self.points[selected].copy(),
            self.units,
            self.frame,
            **{
                name: getattr(self, name)[selected].copy()
                if getattr(self, name) is not None
                else None
                for name in ("colors", "normals", "pixels")
            },
            source_size=self.source_size,
            metadata=copy.deepcopy(self.metadata),
        )

    def describe(self):
        return {
            "model": "point_cloud_v1",
            "count": len(self.points),
            "units": self.units,
            "frame": self.frame,
            "colors": self.colors is not None,
            "normals": self.normals is not None,
            "source_size": self.source_size,
            "pixel_correspondence": self.pixels is not None,
            "bytes": self.nbytes,
            "minimum": self.points.min(axis=0).tolist() if len(self.points) else None,
            "maximum": self.points.max(axis=0).tolist() if len(self.points) else None,
        }


def render_cloud(cloud, width=640, height=480, yaw=25.0, pitch=-20.0, limit=20000, point_size=2):
    """Orthographic preview with far-to-near drawing; no GPU/native viz dependency."""
    image = np.full((height, width, 3), (32, 25, 18), np.uint8)
    if not len(cloud.points):
        return image
    selected = np.linspace(0, len(cloud.points) - 1, min(len(cloud.points), limit), dtype=int)
    points = cloud.points[selected].astype(float)
    points -= (points.min(axis=0) + points.max(axis=0)) / 2
    a, b = np.deg2rad([yaw, pitch])
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rotation = (
        np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]) @ rotation
    )
    points = points @ rotation.T
    span = np.ptp(points[:, :2], axis=0)
    scale = min((width - 30) / max(span[0], 1e-6), (height - 30) / max(span[1], 1e-6))
    pixels = np.rint(points[:, :2] * [scale, -scale] + [width / 2, height / 2]).astype(int)
    colors = (
        cloud.colors[selected, ::-1]
        if cloud.colors is not None
        else np.tile([100, 215, 245], (len(points), 1))
    )
    for i in np.argsort(-points[:, 2]):
        cv2.circle(image, tuple(pixels[i]), point_size, tuple(map(int, colors[i])), -1)
    return image
