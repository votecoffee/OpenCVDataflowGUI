"""Verify typed inspection columns across file, graph and row-selection paths."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud, write_cloud
from opencv_dataflow.cloud_merge import MergeCloudScans
from opencv_dataflow.cloud_sampling import SampleCloud
from opencv_dataflow.cloud_tools import FilterCloud, TransformCloud, VoxelCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix, pose_record
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_inspection_attributes"


def _cloud():
    return PointCloud(
        [[0, 0, 400], [2, 0, 400], [30, 0, 400]],
        "mm",
        "fixture",
        attributes={
            "intensity": [112.12345678901234, np.nan, 34.625],
            "confidence": [np.nan, 0.94, 0.4],
            "source_index": [3, 8, 2**31 - 1],
            "source_scan": [0, 0, 1],
        },
    )


def _assert_columns(cloud, expected):
    assert set(cloud.attributes) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(cloud.attributes[name], values)


@pytest.mark.parametrize(
    "attributes",
    [
        {"intensity": [1]},
        {"confidence": [0.5, np.inf]},
        {"source_index": [1, 2.5]},
        {"source_index": [1, 2**31]},
        {"source_scan": [0, -2]},
        {"intensity": [1, 2**53 + 1]},
        {"unknown": [1, 2]},
    ],
)
def test_rejects_malformed_attribute_columns(attributes):
    with pytest.raises(ValueError, match="attribute"):
        PointCloud([[0, 0, 1], [1, 0, 1]], attributes=attributes)


@pytest.mark.parametrize("format", ["ply", "pcd"])
@pytest.mark.parametrize("binary", [False, True])
def test_file_round_trip_retains_typed_columns_and_explicit_source_ids(tmp_path, format, binary):
    cloud = _cloud()
    path = tmp_path / f"inspection.{format}"
    write_cloud(path, cloud, binary)
    result = read_cloud(path, track_source_rows=True)
    assert (result.frame, result.units) == (cloud.frame, cloud.units)
    _assert_columns(result, cloud.attributes)
    assert result.attributes["intensity"].dtype == np.float64
    assert result.attributes["source_index"].dtype == np.int32
    assert result.describe()["attributes"]["confidence"] == {"dtype": "float64", "known": 2}
    assert result.nbytes == 3 * (12 + 8 + 8 + 4 + 4)
    with pytest.raises(FileExistsError):
        write_cloud(path, cloud, binary)


def test_organized_export_recovers_holes_and_original_row_positions(tmp_path):
    cloud = PointCloud(
        [[0, 0, 1], [0, 1, 1]],
        "m",
        "depth",
        pixels=[[0, 0], [0, 1]],
        source_size=(2, 2),
        attributes={"intensity": [17, 19]},
    )
    path = tmp_path / "organized.pcd"
    write_cloud(path, cloud, binary=True, pcd_organization="source_pixels")
    result = read_cloud(path, track_source_rows=True)
    np.testing.assert_array_equal(result.pixels, cloud.pixels)
    _assert_columns(result, {"intensity": [17, 19], "source_index": [0, 2]})


def test_pcd_vector_named_intensity_remains_readable_as_geometry(tmp_path):
    path = tmp_path / "vector.pcd"
    path.write_bytes(
        b"FIELDS intensity x y z\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 2 1 1 1\n"
        b"POINTS 1\nDATA ascii\n6 7 0 1 2\n"
    )
    cloud = read_cloud(path)
    assert not cloud.attributes
    np.testing.assert_array_equal(cloud.points, [[0, 1, 2]])


def test_lossy_obj_requires_explicit_attribute_omission(tmp_path):
    cloud = _cloud()
    path = tmp_path / "geometry.obj"
    with pytest.raises(ValueError, match="inspection attributes"):
        write_cloud(path, cloud)
    assert not path.exists()
    write_cloud(path, cloud, write_inspection_attributes=False)
    assert not read_cloud(path).attributes


def test_filter_sampling_and_transform_keep_row_alignment():
    cloud = _cloud()
    exclude = FilterCloud(parameters={"attribute": "confidence", "attribute_minimum": 0.8})
    selected = exclude.execute({"cloud": cloud}).outputs["cloud"]
    _assert_columns(selected, {name: values[[1]] for name, values in cloud.attributes.items()})
    keep = FilterCloud(
        parameters={
            "attribute": "confidence",
            "attribute_minimum": 0.8,
            "missing_attribute": "keep",
        }
    )
    selected = keep.execute({"cloud": cloud}).outputs["cloud"]
    _assert_columns(selected, {name: values[:2] for name, values in cloud.attributes.items()})
    sampling = SampleCloud(parameters={"max_points": 2, "seed": 32}).execute({"cloud": cloud})
    rows = sampling.outputs["indices"]
    sampled = sampling.outputs["cloud"]
    _assert_columns(sampled, {name: values[rows] for name, values in cloud.attributes.items()})
    pose = pose_record(pose_matrix(np.eye(3), [10, -2, 1]), "fixture", "station", "mm")
    transformed = TransformCloud().execute({"cloud": sampled, "transform": pose}).outputs["cloud"]
    _assert_columns(transformed, sampled.attributes)
    assert transformed.frame == "station"
    assert not np.shares_memory(
        transformed.attributes["intensity"], sampled.attributes["intensity"]
    )


def test_voxel_centroid_averages_known_measurements_and_omits_row_ids():
    cloud = _cloud()
    result = (
        VoxelCloud(parameters={"voxel_size": 10, "mode": "centroid"})
        .execute({"cloud": cloud})
        .outputs["cloud"]
    )
    _assert_columns(result, {"intensity": [112.12345678901234, 34.625], "confidence": [0.94, 0.4]})


def test_merge_disambiguates_source_rows_across_nested_inputs():
    a = _cloud()
    b = PointCloud([[90, 0, 400]], "mm", "fixture", attributes={"source_index": [3]})
    tool = MergeCloudScans(parameters={"output_units": "mm"})
    first = tool.execute({"cloud_1": a, "cloud_2": b})
    cloud = first.outputs["cloud"]
    np.testing.assert_array_equal(cloud.attributes["source_scan"], [0, 0, 1, 2])
    np.testing.assert_array_equal(cloud.attributes["source_index"], [3, 8, 2**31 - 1, 3])
    assert np.isnan(cloud.attributes["intensity"][-1])
    after = tool.execute({"cloud_1": b, "cloud_2": cloud}).outputs
    np.testing.assert_array_equal(after["cloud"].attributes["source_scan"], [0, 1, 1, 2, 3])
    np.testing.assert_array_equal(
        after["cloud"].attributes["source_index"], [3, 3, 8, 2**31 - 1, 3]
    )
    assert after["report"]["scans"][1]["source_scan_range"] == [1, 4]
    strict = MergeCloudScans(parameters={"attribute_policy": "require_matching"})
    assert strict.execute({"cloud_1": a, "cloud_2": b}) is None
    assert "inspection fields" in strict.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_graph_preserves_row_ids_after_overlap_filter_and_voxel_sample(profile, tmp_path):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / "cloud_inspection_attributes.json", registry)
    try:
        export = graph.add(
            registry.create(
                "output.cloud",
                block_id="export",
                parameters={"path": str(tmp_path / "inspection.pcd"), "pcd_encoding": "binary"},
            )
        )
        graph.connect("sample", "cloud", export.id, "cloud")
        graph.set_run_profile(profile)
        graph.run()
        assert graph.blocks["merge"].last_result.outputs["report"]["suppressed_points"] == 1
        attributes = {
            "intensity": [41, 55, np.nan],
            "confidence": [0.95, 0.75, 0.77],
            "source_index": [2, 5, 8],
            "source_scan": [0, 0, 1],
        }
        _assert_columns(graph.blocks["sample"].last_result.outputs["cloud"], attributes)
        _assert_columns(read_cloud(tmp_path / "inspection.pcd"), attributes)
    finally:
        graph.close()
