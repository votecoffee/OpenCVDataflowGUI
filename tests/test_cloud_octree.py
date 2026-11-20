"""Verify octree queries against exhaustive metric searches and original cloud rows."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_octree import QueryCloudOctree, SparseOctree
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.pointclouds import PointCloud

PROJECT = Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_octree_query"


def _brute(points, ids, query, limit, radius=None):
    dist_sq = np.sum((points - query) ** 2, axis=1)
    eligible = np.arange(len(ids))
    if radius is not None:
        eligible = eligible[dist_sq <= radius**2]
    order = np.lexsort((ids[eligible], dist_sq[eligible]))
    best = eligible[order[:limit]]
    return ids[best], np.sqrt(dist_sq[best]), len(eligible) > limit if radius is not None else False


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_imported_cloud_exercises_duplicates_ties_bounds_and_units(profile):
    graph = Graph.load(PROJECT / "cloud_octree_query.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        cloud = graph.blocks["source"].last_result.outputs["cloud"]
        points = cloud.points.astype(np.float64) * 0.001
        keep = np.flatnonzero(
            ((points >= [-0.25, -0.25, 1]) & (points <= [0.25, 0.25, 1.5])).all(axis=1)
        )
        for name in ("knn", "radius"):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            report = result.outputs["report"]
            radi = 0.125 if name == "radius" else None
            assert report["indexed_points"] == 164 and report["excluded_points"] == 1
            assert report["index_bytes"] == len(keep) * 36
            for row in report["queries"]:
                ids, distance, truncated = _brute(
                    points[keep], keep, row["xyz_m"], block.parameters["max_neighbors"], radi
                )
                np.testing.assert_array_equal(row["indices"], ids)
                np.testing.assert_allclose(row["distances_m"], distance, rtol=1e-13, atol=1e-15)
                assert row["truncated"] == truncated
            selected = report["neighbor_cloud_source_indices"]
            neighbors = result.outputs["neighbors"]
            assert (neighbors.units, neighbors.frame) == ("mm", "scan")
            assert result.outputs["count"] == len(selected)
            np.testing.assert_array_equal(neighbors.points, cloud.points[selected])
            np.testing.assert_array_equal(neighbors.colors, cloud.colors[selected])
        knn = graph.blocks["knn"].last_result.outputs["report"]
        radius = graph.blocks["radius"].last_result.outputs["report"]
        assert knn["queries"][0]["indices"][:2] == [40, 162]
        assert radius["queries"][0]["truncated"]
        assert radius["queries"][1]["indices"] == [163]
        assert radius["queries"][1]["distances_m"] == [0.125]
        assert radius["levels"] == 6 and radius["cell_side_m"] == 0.5 / 64
    finally:
        graph.close()


@pytest.mark.parametrize("depth", [0, 6, 20])
def test_random_clusters_and_outside_queries_match_independent_search(depth):
    rng = np.random.default_rng(307)
    points = rng.uniform(0, 1, size=(1500, 3))
    points[:800] *= 0.1
    points[:5] = 0
    points[-1] = 1
    ids = rng.permutation(len(points)).astype(np.int32)
    index = SparseOctree(points, ids, np.array([0.0, 0.0, 0.0]), 1.0, depth)
    queries = np.vstack((rng.uniform(-0.25, 1.25, size=(25, 3)), (0, 0, 0), (1, 1, 1)))
    for query in queries:
        for radius in (None, 0.05, 0.2):
            pairs, truncated = index.query(query, 13, radius)
            expected, dists, more = _brute(points, ids, query, 13, radius)
            np.testing.assert_array_equal([i for i, d in pairs], expected)
            np.testing.assert_allclose([d for i, d in pairs], dists, atol=1e-14, rtol=1e-14)
            assert truncated == more


def test_large_coincident_leaf_preserves_ties_and_linear_storage():
    count = 16000
    points = np.zeros((count, 3), np.float32)
    points[:, 2] = 1
    cloud = PointCloud(points, "m", "fixture")
    block = QueryCloudOctree(
        parameters={"max_depth": 20, "max_neighbors": 6, "search": "radius", "radius_m": 0.01}
    )
    result = block.execute(
        {"cloud": cloud, "queries": {"frame": "fixture", "units": "mm", "points": [[0, 0, 1000]]}}
    )
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["index_bytes"] == count * 36
    assert report["queries"][0]["indices"] == list(range(6))
    assert report["queries"][0]["distances_m"] == [0] * 6 and report["queries"][0]["truncated"]
    assert report["levels"] == 20


def test_explicit_exclusion_and_no_hits_are_inspection_failure():
    cloud = PointCloud([[0.5, 0.5, 0.5], [2, 2, 2]], "m", "fixture")
    block = QueryCloudOctree(
        parameters={"bounds": "explicit", "search": "radius", "radius_m": 0.01}
    )
    result = block.execute({"cloud": cloud})
    assert result is not None and not result.passed
    assert result.outputs["neighbors"].points.shape == (0, 3)
    assert result.outputs["report"]["indexed_points"] == 1
    assert result.fail_reasons[0]["code"] == "octree_no_neighbors"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("budget", "point budget"),
        ("depth", "more than 20"),
        ("frame", "different coordinate frame"),
        ("units", "units of m or mm"),
        ("requests", "128 finite"),
    ],
)
def test_query_validation_and_allocation_limits(mode, message):
    cloud = PointCloud([[0, 0, 0], [1, 1, 1]], "m", "fixture")
    queries, parameters = [[0, 0, 0]], {}
    if mode == "budget":
        parameters["max_input_points"] = 1
    if mode == "depth":
        parameters.update(subdivision="resolution", resolution_m=1e-9)
    if mode == "frame":
        queries = ValuePacket(queries, {"optical_frame": "other"})
    if mode == "units":
        cloud = PointCloud(cloud.points, "unitless", cloud.frame)
    if mode == "requests":
        queries *= 129
    block = QueryCloudOctree(parameters=parameters)
    assert block.execute({"cloud": cloud, "queries": queries}) is None
    assert message in block.error
