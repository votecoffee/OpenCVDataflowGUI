"""Bounded local PCA shared by cloud region growing and normal quality measurement."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalSurfaceFits:
    normals: np.ndarray
    curvature: np.ndarray
    normal_quality: np.ndarray
    neighbor_counts: np.ndarray


def fit_local_surfaces(points, tree, radius, neighbors, cancelled, *, minimum_neighbors=3):
    """Estimate unsigned normals, variance ratios and neighborhood support."""
    normals = np.zeros_like(points)
    curvature = np.full(len(points), np.nan)
    quality = np.full(len(points), np.nan)
    neighbor_counts = np.zeros(len(points), np.int16)
    k = min(neighbors, len(points))
    if not k:
        return LocalSurfaceFits(normals, curvature, quality, neighbor_counts)

    batch = 65_536 // k
    for start in range(0, len(points), batch):
        if cancelled():
            return None
        stop = start + batch
        centers = points[start:stop]
        distances, indices = tree.query(
            centers, k=k, distance_upper_bound=np.nextafter(radius, np.inf)
        )
        if k == 1:
            distances, indices = distances[:, None], indices[:, None]
        valid = distances <= radius
        counts = np.count_nonzero(valid, axis=1)
        neighbor_counts[start:stop] = counts
        rows = np.flatnonzero(counts >= minimum_neighbors)
        if not len(rows):
            continue
        valid, indices = valid[rows], indices[rows]
        offsets = points[np.where(valid, indices, rows[:, None] + start)] - centers[rows, None]
        # Center local offsets rather than absolute coordinates to retain small features
        # far from the origin. Missing neighbors contribute zero to covariance.
        offsets -= offsets.sum(axis=1, keepdims=True) / counts[rows, None, None]
        offsets[~valid] = 0
        values, vectors = np.linalg.eigh(offsets.transpose(0, 2, 1) @ offsets)
        values = np.maximum(values, 0)
        gap = values[:, 1] - values[:, 0]
        largest = values[:, 2]
        quality[start + rows] = np.divide(gap, largest, out=np.zeros_like(gap), where=largest > 0)
        # Coincident, collinear and isotropic neighborhoods have no unique normal.
        usable = gap > largest * 1e-8
        output_rows = start + rows[usable]
        normals[output_rows] = vectors[usable, :, 0]
        curvature[output_rows] = values[usable, 0] / values[usable].sum(axis=1)
    return LocalSurfaceFits(normals, curvature, quality, neighbor_counts)
