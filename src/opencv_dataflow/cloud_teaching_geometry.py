"""Full-source orthographic picking and editable boxes for retained cloud teaching."""

import hashlib
from dataclasses import dataclass

import numpy as np

from .pointclouds import PointCloud


def vector(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Expected three finite coordinates")
    return value


@dataclass(frozen=True)
class CloudBox:
    name: str
    minimum: tuple
    maximum: tuple

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A region needs a name")
        lo, hi = vector(self.minimum), vector(self.maximum)
        if np.any(lo >= hi):
            raise ValueError("Region maximum must exceed minimum on every axis")
        object.__setattr__(self, "minimum", tuple(lo))
        object.__setattr__(self, "maximum", tuple(hi))

    @property
    def corners(self):
        return np.array([[x, y, z] for x in (self.minimum[0], self.maximum[0])
                         for y in (self.minimum[1], self.maximum[1])
                         for z in (self.minimum[2], self.maximum[2])])

    @property
    def faces(self):
        center = (np.array(self.minimum) + self.maximum) / 2
        faces = np.tile(center, (6, 1))
        for axis in range(3):
            faces[2 * axis, axis] = self.minimum[axis]
            faces[2 * axis + 1, axis] = self.maximum[axis]
        return faces


class CloudTeachingSource:
    """Private immutable sensor rows, never mixed with generated overlay samples."""

    def __init__(self, cloud):
        if not isinstance(cloud, PointCloud):
            raise ValueError("Teaching requires a native PointCloud")
        self.cloud = cloud.subset(slice(None))
        for name in ("points", "colors", "normals", "pixels"):
            value = getattr(self.cloud, name)
            if value is not None:
                value.flags.writeable = False
        xyz = self.cloud.points
        self.center = (xyz.min(0).astype(float) + xyz.max(0)) / 2 if len(xyz) else np.zeros(3)
        self.radius = max(float(np.max(np.linalg.norm(xyz - self.center, axis=1))), 1e-9) \
            if len(xyz) else 1.
        digest = hashlib.sha256()
        digest.update((cloud.units + "\0" + cloud.frame + "\0").encode("utf-8"))
        digest.update(xyz.tobytes())
        self.identity = digest.hexdigest()


class CloudProjection:
    """Nearest visible row per pixel; picking uses all source rows, not display decimation."""

    def __init__(self, source, width, height, *, yaw=25, pitch=-20, zoom=1, pan=(0, 0)):
        if not isinstance(source, CloudTeachingSource):
            raise ValueError("Expected a retained teaching source")
        if (type(width) is not int or type(height) is not int
                or not 1 <= width <= 4096 or not 1 <= height <= 4096
                or not np.isfinite([yaw, pitch, zoom, *pan]).all() or not .1 <= zoom <= 100):
            raise ValueError("Invalid teaching viewport")
        self.source, self.width, self.height = source, int(width), int(height)
        a, b = np.deg2rad([yaw, pitch])
        self.rotation = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)],
                                  [0, np.sin(b), np.cos(b)]]) @ np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        self.scale = max(1., min(width, height) - 40) / (2 * source.radius) * zoom
        self.offset = np.array([width / 2, height / 2]) + pan
        pixels, depth = self.project(source.cloud.points)
        self.pixels, self.depth = pixels, depth
        inside = np.all((pixels >= 0) & (pixels < [width, height]), axis=1)
        self.rows = np.flatnonzero(inside)
        pixel = np.rint(pixels[self.rows]).astype(int)
        valid = np.all(pixel < [width, height], axis=1)
        rows, pixel = self.rows[valid], pixel[valid]
        flat = pixel[:, 1] * width + pixel[:, 0]
        z = np.full(width * height, np.inf)
        np.minimum.at(z, flat, depth[rows])
        front = depth[rows] == z[flat]
        owner = np.full(width * height, len(source.cloud.points), dtype=np.int32)
        np.minimum.at(owner, flat[front], rows[front])
        owner[owner == len(source.cloud.points)] = -1
        self.owner = owner.reshape(height, width)

    def project(self, xyz):
        rotated = (np.asarray(xyz) - self.source.center) @ self.rotation.T
        return rotated[..., :2] * [self.scale, -self.scale] + self.offset, rotated[..., 2]

    def pick(self, x, y, radius=6):
        if not np.isfinite([x, y, radius]).all() or not 0 <= radius <= 50:
            raise ValueError("Invalid pick position or radius")
        x0, y0 = max(0, int(x - radius)), max(0, int(y - radius))
        x1, y1 = min(self.width, int(x + radius) + 1), min(self.height, int(y + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = np.nonzero(self.owner[y0:y1, x0:x1] >= 0)
        rows = self.owner[yy + y0, xx + x0]
        distance = (xx + x0 - x)**2 + (yy + y0 - y)**2
        keep = distance <= radius**2
        if not keep.any():
            return None
        rows, distance = rows[keep], distance[keep]
        row = int(rows[np.lexsort((rows, self.depth[rows], distance))[0]])
        return {"row": row, "xyz": self.source.cloud.points[row].astype(float).tolist(),
                "source_identity": self.source.identity,
                "units": self.source.cloud.units, "frame": self.source.cloud.frame}

    def box_from_rectangle(self, first, last, name):
        low, high = np.minimum(first, last), np.maximum(first, last)
        lo = np.maximum(0, np.floor(low).astype(int))
        hi = np.minimum([self.width - 1, self.height - 1], np.ceil(high).astype(int))
        if np.any(lo > hi):
            raise ValueError("The drawn patch lies outside the cloud viewport")
        rows = self.owner[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        rows = np.unique(rows[rows >= 0])
        if len(rows) < 3:
            raise ValueError("Draw a patch containing at least three visible source points")
        points = self.source.cloud.points[rows].astype(float)
        pad = max(float(np.linalg.norm(np.ptp(points, axis=0))) * .02,
                  self.source.radius * .001)
        return CloudBox(name, tuple(points.min(0) - pad), tuple(points.max(0) + pad))

    def drag_face(self, box, face, delta):
        if type(face) is not int or not 0 <= face < 6:
            raise ValueError("Choose a box face")
        axis = face // 2
        direction = self.rotation[:2, axis] * [self.scale, -self.scale]
        norm = float(direction @ direction)
        if norm < (self.scale * .05)**2:
            raise ValueError("Rotate the view to drag this depth-aligned face")
        offset = float(np.asarray(delta) @ direction / norm)
        bounds = [list(box.minimum), list(box.maximum)]
        bounds[face % 2][axis] += offset
        # Crossing an opposite face is invalid, never silently inverted.
        return CloudBox(box.name, tuple(bounds[0]), tuple(bounds[1]))

    def image(self, colors=None):
        image = np.full((self.height, self.width, 3), (32, 25, 18), np.uint8)
        visible = self.owner >= 0
        rows = self.owner[visible]
        colors = self.source.cloud.colors if colors is None else colors
        image[visible] = (colors[rows, ::-1] if colors is not None else [150, 200, 220])
        return image
