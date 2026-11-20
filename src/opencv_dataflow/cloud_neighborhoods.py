"""Bounded spatial graph and preview helpers for point-cloud segmentation."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .pointclouds import PointCloud, render_cloud


def count_radius_pairs(tree, radius):
    # The self-query includes N self-pairs and both orders of every distinct pair.
    return (int(tree.count_neighbors(tree, radius)) - tree.n) // 2


def pair_components(pairs, size):
    edges = coo_matrix((np.ones(len(pairs), np.int8), pairs.T), shape=(size, size))
    return connected_components(edges, directed=False)


def labeled_cloud_preview(cloud, labels, count):
    palette = np.random.default_rng(83).integers(65, 256, size=(count, 3), dtype=np.uint8)
    display = PointCloud(cloud.points, cloud.units, cloud.frame, palette[labels - 1])
    return render_cloud(display)
