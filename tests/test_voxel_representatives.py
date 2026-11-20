"""Qualify exact measured voxel representatives and the legacy voxel modes."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_tools import VoxelCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/voxel_representatives/voxel_representatives.json"
)


def _oracle(points, sizes):
    groups = defaultdict(list)
    for row, p in enumerate(points):
        groups[tuple(np.floor(p / sizes).astype(np.int64))].append(row)
    indices, centroids = [], []
    for key in sorted(groups):
        rows = groups[key]
        coords = points[rows].astype(np.float64)
        center = coords.mean(axis=0)
        squared = np.sum((coords - center) ** 2, axis=1)
        indices.append(rows[int(squared.argmin())])
        centroids.append(center)
    return indices, np.array(centroids)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_retains_rows_with_anisotropic_voxels(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["source"].last_result.outputs["cloud"]
        measured = graph.blocks["representative"].last_result
        synthetic = graph.blocks["centroid"].last_result
        assert measured and measured.passed and synthetic and synthetic.passed
        expected, centers = _oracle(source.points, np.array([10.0, 20.0, 30.0]))
        assert expected == [0, 10, 3, 9, 5, 7]
        assert measured.outputs["indices"] == expected
        assert measured.outputs["count"] == synthetic.outputs["count"] == 6
        sample = measured.outputs["cloud"]
        assert (sample.frame, sample.units) == ("voxel_fixture", "mm")
        for attr in ("points", "normals", "colors"):
            np.testing.assert_array_equal(getattr(sample, attr), getattr(source, attr)[expected])
        assert synthetic.outputs["indices"] is NO_VALUE
        np.testing.assert_allclose(synthetic.outputs["cloud"].points, centers)
        assert synthetic.outputs["cloud"].normals is None
    finally:
        graph.close()


def test_pixel_provenance_and_equal_distance_ties_keep_the_first_measured_row():
    # The centroid lies at 7/3. Rows 0 and 1 are coincident and win together.
    raw = [[2, 0, 0], [2, 0, 0], [3, 0, 0]]
    ticket = TicketClock().issue("cloud")
    coords = frame([3, 1], "camera")
    source = PointCloud(
        raw,
        "m",
        "fixture",
        colors=[[20, 0, 0], [30, 0, 0], [40, 0, 0]],
        normals=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pixels=[[0, 0], [1, 0], [2, 0]],
        source_size=(3, 1),
        metadata={"spatial": coords},
    )
    block = VoxelCloud(parameters={"voxel_size": 10, "mode": "representative"})
    result = block.execute({"cloud": ValuePacket(source, {"ticket": ticket})})
    assert result and result.passed, block.error
    output = result.outputs["cloud"]
    assert result.outputs["indices"] == [0]
    assert result.payloads["cloud"]["ticket"] == ticket
    assert result.payloads["cloud"]["spatial"] == coords
    assert output.source_size == source.source_size
    for attr in ("points", "colors", "normals", "pixels"):
        np.testing.assert_array_equal(getattr(output, attr), getattr(source, attr)[:1])

    block.parameters["mode"] = "centroid"
    result = block.execute({"cloud": source})
    mean = result.outputs["cloud"]
    np.testing.assert_allclose(mean.points, [[7 / 3, 0, 0]])
    np.testing.assert_array_equal(mean.colors, [[30, 0, 0]])
    assert mean.normals is None and mean.pixels is None and mean.source_size is None
    assert result.outputs["indices"] is NO_VALUE


def test_legacy_saved_block_uses_scalar_and_preserves_first_row_order(tmp_path):
    graph, registry = Graph(), builtin_registry()
    block = graph.add(
        registry.create("3d.cloud_voxel", block_id="voxel", parameters={"voxel_size": 1})
    )
    block.exposed_outputs = ["cloud", "count"]
    saved = graph.to_dict()
    graph.close()
    record = saved["blocks"][0]
    record["parameters"] = {"voxel_size": 1, "mode": "first"}
    opened = Graph.from_dict(saved, registry, project_dir=tmp_path)
    try:
        cloud = PointCloud([[1.1, 2, 0], [-0.01, 0, 0], [1.9, 2, 0], [2, 2, 0]])
        result = opened.blocks["voxel"].execute({"cloud": cloud})
        assert result and result.passed
        assert result.outputs["count"] == 3
        np.testing.assert_array_equal(result.outputs["cloud"].points, cloud.points[[1, 0, 3]])
        assert result.outputs["indices"] is NO_VALUE
    finally:
        opened.close()


@pytest.mark.parametrize("setting", [0, -1, np.inf, np.nan])
def test_invalid_per_axis_spacing_fails_with_the_named_field(setting):
    cloud = PointCloud([[0, 0, 0]])
    block = VoxelCloud(parameters={"spacing": "per_axis", "voxel_size_y": setting})
    assert block.execute({"cloud": cloud}) is None
    assert "Voxel size y" in block.error


def test_empty_and_sparse_extreme_coordinates():
    block = VoxelCloud(parameters={"mode": "representative", "voxel_size": 1.0})
    empty = block.execute({"cloud": PointCloud(np.empty((0, 3)))})
    assert empty and empty.passed and empty.outputs["indices"] == []
    spread = PointCloud([[-1e12, 0, 0], [0, 0, 0], [1e12, 0, 0]])
    result = block.execute({"cloud": spread})
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["cloud"].points, spread.points)
    block.parameters["voxel_size"] = 1e-12
    assert block.execute({"cloud": spread}) is None
    assert "Voxel coordinates exceed" in block.error
