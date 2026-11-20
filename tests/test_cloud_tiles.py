"""Qualify radius halos and exclusive, ordered PLY export."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud, write_cloud
from opencv_dataflow.cloud_tiles import TiledCloudRadius
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/tiled_cloud_radius"


def _settings(source, output, tmp_path, **parameters):
    return {
        "path": str(source),
        "output_path": str(output),
        "scratch_directory": str(tmp_path),
        "tile_size_m": 0.25,
        "radius_m": 0.125,
        "minimum_neighbors": 1,
        **parameters,
    }


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_project_crosses_seams_preserves_order_and_keeps_attributes(profile, tmp_path):
    graph = Graph.load(EXAMPLE / "tiled_cloud_radius.json", builtin_registry())
    destination = tmp_path / "kept.ply"
    try:
        block = graph.blocks["filter"]
        block.parameters.update(output_path=str(destination), scratch_directory=str(tmp_path))
        graph.set_run_profile(profile)
        graph.run()
        result = block.last_result
        assert result is not None and result.passed, block.error
        assert Path(result.outputs["path"]) == destination
        original = read_cloud(EXAMPLE / "assets/scan.ply")
        metric = original.points.astype(np.float64) * 0.001
        distances = np.linalg.norm(metric[:, None] - metric[None, :], axis=-1)
        counts = (distances <= 0.125).sum(axis=1) - 1
        indices = np.flatnonzero(counts)
        exported = read_cloud(destination)
        np.testing.assert_array_equal(exported.points, original.points[indices])
        np.testing.assert_array_equal(exported.colors, original.colors[indices])
        np.testing.assert_array_equal(exported.attributes["source_index"], [31, 9, 89, 12, 14, 22])
        report = result.outputs["report"]
        assert exported.units == report["cloud_units"] == "mm"
        assert exported.frame == report["frame"] == "inspection"
        assert report["neighbor_count_sum"] == counts.sum()
        assert report["minimum_observed_neighbors"] == counts.min()
        assert report["maximum_observed_neighbors"] == counts.max()
        assert report["mean_observed_neighbors"] == counts.mean()
        assert report["kept_points"] == 6 and report["removed_points"] == 2
        assert report["peak_core_points"] < report["peak_window_points"] < len(counts)
        assert not list(tmp_path.glob("dataflow-cloud-*"))
    finally:
        graph.close()


def test_new_indices_ignore_invalid_xyz_and_preserve_unknown_scalar_bytes(tmp_path):
    source, output = tmp_path / "scan.ply", tmp_path / "export.ply"
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("temperature", "<u2")])
    rows = np.array([(0.0, 0, 0, 1000), (np.nan, 0, 0, 20), (0.125, 0, 0, 3000)], dtype)
    source.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property ushort temperature\nend_header\n" + rows.tobytes()
    )
    block = TiledCloudRadius(parameters=_settings(source, output, tmp_path))
    result = block.execute({})
    assert result is not None and result.passed, block.error
    assert result.outputs["report"]["invalid_xyz_rows"] == 1
    body, packed = output.read_bytes().split(b"end_header\n")
    assert b"property uint16 temperature\n" in body
    retained = np.frombuffer(packed, dtype=np.dtype(dtype.descr + [("source_index", "<i4")]))
    np.testing.assert_array_equal(retained["temperature"], [1000, 3000])
    np.testing.assert_array_equal(retained["source_index"], [0, 2])
    before = output.read_bytes()
    assert block.execute({}) is None and "new PLY" in block.error
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("dataflow-cloud-*"))


@pytest.mark.parametrize(
    "settings, status",
    [
        ({"max_window_points": 2}, "window_budget"),
        ({"max_tiles": 1}, "tile_budget"),
        ({"minimum_neighbors": 4}, "no_inliers"),
    ],
)
def test_stopped_scan_cannot_publish_a_partial_result(settings, status, tmp_path):
    source, output = tmp_path / "scan.ply", tmp_path / "out.ply"
    write_cloud(source, PointCloud([[0, 0, 0], [0.125, 0, 0], [-0.125, 0, 0]]), binary=True)
    block = TiledCloudRadius(parameters=_settings(source, output, tmp_path, **settings))
    result = block.execute({})
    assert result is not None and not result.passed, block.error
    assert result.outputs["path"] is NO_VALUE and result.outputs["report"]["status"] == status
    assert not output.exists() and not list(tmp_path.glob("dataflow-cloud-*"))


def test_bad_layout_and_cancellation_leave_output_absent(tmp_path):
    source, output = tmp_path / "scan.ply", tmp_path / "out.ply"
    write_cloud(source, PointCloud([[0, 0, 0], [0.125, 0, 0]]), binary=True)
    full = source.read_bytes()
    source.write_bytes(full[:-1])
    block = TiledCloudRadius(parameters=_settings(source, output, tmp_path))
    assert block.execute({}) is None and "declared vertex rows" in block.error
    source.write_bytes(full)
    result = block.execute({}, cancelled=lambda: True)
    assert result is not None and result.outputs["report"]["status"] == "cancelled"
    assert not output.exists() and not list(tmp_path.glob("dataflow-cloud-*"))


def test_disk_budget_cleans_partial_index_without_export(tmp_path):
    source, output = tmp_path / "scan.ply", tmp_path / "out.ply"
    xyz = np.arange(60_000 * 3, dtype=np.float32).reshape(-1, 3) / 8192
    write_cloud(source, PointCloud(xyz), binary=True)
    block = TiledCloudRadius(
        parameters=_settings(source, output, tmp_path, max_index_mib=1),
    )
    result = block.execute({})
    assert result is not None and not result.passed, block.error
    assert result.outputs["path"] is NO_VALUE
    assert result.outputs["report"]["status"] == "index_budget"
    assert not output.exists() and not list(tmp_path.glob("dataflow-cloud-*"))
