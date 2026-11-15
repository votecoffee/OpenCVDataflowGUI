"""Retained expected planar-top samples, learned once and held fixed during picking."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class TopTemplate:
    def __init__(self, points_xy_m, *, tolerance_m=0.0022, source=None):
        points = np.asarray(points_xy_m, float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 8 <= len(points) <= 20000
            or not np.isfinite(points).all()
            or np.max(np.abs(points)) > 1
            or not np.isfinite(tolerance_m)
            or not 1e-5 <= tolerance_m <= 0.01
        ):
            raise ValueError("Expected bounded finite physical top-template samples/tolerance")
        self.points = points.copy()
        self.points.flags.writeable = False
        self.tolerance_m = float(tolerance_m)
        self.source = source or {}

    def record(self):
        return dict(
            schema="expected_planar_top_v1",
            points_xy_m=self.points.tolist(),
            tolerance_m=self.tolerance_m,
            source=self.source,
            meaning="expected surface coverage; not object identity or grasp probability",
        )

    def save(self, path):
        path = Path(path)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(self.record(), stream, indent=2, allow_nan=False)
            stream.write("\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, sha256):
        path = Path(path)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Top-template size or SHA256 differs")
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Top-template size or SHA256 differs")
        value = json.loads(data)
        if value.get("schema") != "expected_planar_top_v1":
            raise ValueError("Unknown top-template schema")
        return cls(value["points_xy_m"], tolerance_m=value["tolerance_m"], source=value["source"])

    def coverage(self, surface_xy_m, center_xy_m, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        expected = self.points @ np.array([[c, s], [-s, c]]) + center_xy_m
        distance, _ = cKDTree(surface_xy_m).query(expected, k=1)
        matched = distance <= self.tolerance_m
        return float(np.mean(matched)), expected, matched


def train_mesh_top(vertices_m, faces, *, pitch_m=0.0015, top_band_m=0.002, source=None):
    """Clip mesh faces to the upper band and rasterize expected XY surface samples."""
    import cv2

    vertices = np.asarray(vertices_m, float)
    triangles = vertices[np.asarray(faces, int)]
    low, high = vertices.min(0), vertices.max(0)
    shape = np.ceil((high[:2] - low[:2]) / pitch_m).astype(int) + 3
    if np.prod(shape) > 1_000_000:
        raise ValueError("Top-template training raster is too large")
    origin = low[:2] - pitch_m
    mask = np.zeros(tuple(shape[::-1]), np.uint8)
    z = high[2] - top_band_m
    for triangle in triangles:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if a[2] >= z:
                polygon.append(a)
            if (a[2] >= z) != (b[2] >= z):
                polygon.append(a + (b - a) * ((z - a[2]) / (b[2] - a[2])))
        if len(polygon) >= 3:
            xy = np.asarray(polygon)[:, :2]
            pixels = np.rint((xy - origin) / pitch_m).astype(np.int32)
            cv2.fillConvexPoly(mask, pixels, 1)
    y, x = np.nonzero(mask)
    points = np.c_[x, y] * pitch_m + origin - (low[:2] + high[:2]) / 2
    return TopTemplate(
        points,
        tolerance_m=pitch_m * np.sqrt(2) + 0.0001,
        source={**(source or {}), "pitch_m": pitch_m, "top_band_m": top_band_m},
    ), mask
