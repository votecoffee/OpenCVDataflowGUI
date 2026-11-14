"""First-return projection: invalid/outside/occluded rays remain unknown."""

import numpy as np
from scipy.spatial import cKDTree

from .specs import depth_array, rigid


class Scene:
    def __init__(self, recipe, depth, *, bin_to_camera=None, instances=None):
        self.recipe = recipe
        self.depth = depth_array(depth, recipe.camera) * recipe.depth_scale
        self.pose = rigid(recipe.bin_to_camera if bin_to_camera is None else bin_to_camera)
        valid = np.isfinite(self.depth) & (self.depth > 0) & (self.depth <= 1e9)
        y, x = np.nonzero(valid)
        k = np.asarray(recipe.camera.matrix)
        z = self.depth[y, x]
        camera = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
        self.points = np.ascontiguousarray((camera - self.pose[:3, 3]) @ self.pose[:3, :3])
        self.points.flags.writeable = False
        self.pixels = np.c_[x, y].astype(np.int32)
        self.pixels.flags.writeable = False
        self.tree = cKDTree(self.points, copy_data=False) if len(self.points) else None
        self.instances = None
        if instances is not None:
            raw = np.asarray(instances)
            if (raw.shape != self.depth.shape or raw.dtype.kind not in "ui" or
                    np.any(raw < 0) or np.any(raw > 2**31 - 1)):
                raise ValueError("Instance labels need a matching unsigned/integer image")
            self.instances = np.array(raw, np.int32, copy=True)
            self.instances.flags.writeable = False
        self.depth.flags.writeable = False

    def project(self, points):
        data = np.asarray(points) @ self.pose[:3, :3].T + self.pose[:3, 3]
        k = np.asarray(self.recipe.camera.matrix)
        z = data[:, 2]
        uv = np.empty((len(data), 2))
        uv[:] = np.nan
        valid = z > 1e-12
        uv[valid, 0] = data[valid, 0] * k[0, 0] / z[valid] + k[0, 2]
        uv[valid, 1] = data[valid, 1] * k[1, 1] / z[valid] + k[1, 2]
        return uv, z

    def sample(self, points):
        """Use all four neighboring depth rays; holes never interpolate to observations."""
        uv, z = self.project(points)
        w, h = self.recipe.camera.image_size
        inside = (np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
        xy = np.zeros((len(uv), 2), int)
        xy[inside] = np.floor(uv[inside]).astype(int)
        x, y = xy.T
        samples = self.depth[y[:, None] + [0, 0, 1, 1], x[:, None] + [0, 1, 0, 1]]
        observed = (inside & np.isfinite(samples).all(1) & (samples > 0).all(1)
                    & (samples <= 1e9).all(1))
        nearest = samples.min(1)
        farthest = samples.max(1)
        return observed, nearest, farthest, uv, z

    def contact(self, points, center, normal, instance):
        observed, _, _, uv, _ = self.sample(points)
        xy = np.zeros((len(uv), 2), int)
        xy[observed] = np.floor(uv[observed]).astype(int)
        x = xy[:, 0, None] + [0, 1, 0, 1]
        y = xy[:, 1, None] + [0, 0, 1, 1]
        z = self.depth[y, x]
        k = np.asarray(self.recipe.camera.matrix)
        data = np.stack([(x - k[0, 2]) * z / k[0, 0],
                         (y - k[1, 2]) * z / k[1, 1], z], axis=-1)
        local = (data - self.pose[:3, 3]) @ self.pose[:3, :3]
        gap = np.max(abs((local - center) @ normal), axis=1)
        same = (np.ones(len(points), bool) if self.instances is None else
                np.all(self.instances[y, x] == instance, axis=1))
        return observed, gap, same
