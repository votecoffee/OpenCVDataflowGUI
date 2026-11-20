"""Verify OBJ, native OpenCV interchange and existing point-cloud block behavior."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud, write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.toolbox import block_entry

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_obj_interchange/cloud_obj_interchange.json"
)


def _assert_attributes(actual, expected, *, atol=1e-7, rtol=0):
    np.testing.assert_allclose(actual.points, expected.points, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual.normals, expected.normals, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(actual.colors, expected.colors)
    assert (actual.units, actual.frame) == (expected.units, expected.frame)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_obj_and_ply_branches_match_and_publish_tickets(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        obj = graph.blocks["source_obj"].last_result
        ply = graph.blocks["source_ply"].last_result
        assert obj and obj.passed and ply and ply.passed
        assert obj.outputs["count"] == ply.outputs["count"] == 12
        assert obj.outputs["ticket"] and ply.outputs["ticket"]
        _assert_attributes(obj.outputs["cloud"], ply.outputs["cloud"])
        assert obj.outputs["cloud"].units == "mm"
        assert obj.outputs["cloud"].frame == "folded_coupon"
        a = graph.blocks["measure_obj"].last_result.outputs["measurements"]
        b = graph.blocks["measure_ply"].last_result.outputs["measurements"]
        assert a["centroid"] == b["centroid"] and a["extent"] == b["extent"]
    finally:
        graph.close()


def test_existing_save_block_publishes_obj_on_unicode_paths_and_preserves_destination(tmp_path):
    cloud = PointCloud(
        [[-1.2345678, 2.45, 9], [3.0, -4.2345678, 11]],
        "mm",
        "fixture \u2603",
        colors=[[0, 1, 255], [255, 12, 0]],
        normals=[[0, 0, 1], [0, -1, 0]],
    )
    destination = tmp_path / "\u2603.OBJ"
    registry = builtin_registry()
    sink = registry.create("output.cloud", parameters={"path": destination.name})
    sink.project_dir = tmp_path
    result = sink.execute({"cloud": cloud})
    assert result and result.passed, sink.error
    first = destination.read_bytes()
    assert b"v " in first and b"vn " in first
    origin = registry.create("input.cloud", parameters={"path": destination.name})
    origin.project_dir = tmp_path
    loaded = origin.execute({})
    assert loaded and loaded.passed, origin.error
    _assert_attributes(loaded.outputs["cloud"], cloud, atol=0)
    assert sink.execute({"cloud": cloud}) is None
    assert destination.read_bytes() == first
    assert list(tmp_path.iterdir()) == [destination]
    assert "OBJ" in block_entry(origin).title and "OBJ" in block_entry(sink).title


def test_obj_without_optional_attributes_supports_comments_and_unit_weight(tmp_path):
    path = tmp_path / "external.obj"
    path.write_bytes(
        b"\xef\xbb\xbf# measurement\r\nmtllib unused.mtl\r\n"
        b" v 1 2 3 1 # point one\r\n\tv 4 5 6\r\np 1 2\r\n"
    )
    cloud = read_cloud(path, units="mm", frame="scanner")
    np.testing.assert_array_equal(cloud.points, [[1, 2, 3], [4, 5, 6]])
    assert cloud.normals is None and cloud.colors is None
    assert (cloud.units, cloud.frame) == ("mm", "scanner")


def test_obj_writer_roundtrips_float32_extremes(tmp_path):
    points = np.array(
        [[-np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1)), 0]],
        dtype=np.float32,
    )
    path = tmp_path / "extreme.obj"
    write_cloud(path, PointCloud(points))
    np.testing.assert_array_equal(read_cloud(path).points, points)


@pytest.mark.parametrize("suffix,binary", [(".obj", False), (".ply", False), (".ply", True)])
def test_opencv_can_read_app_output_and_app_can_read_native_output(tmp_path, suffix, binary):
    cloud = PointCloud(
        [[1.2345678, -2, 3], [4, 5, -6.9876543], [-7, 8, 9]],
        "mm",
        "native_fixture",
        colors=[[255, 0, 1], [0, 128, 255], [9, 17, 200]],
        normals=[[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
    )
    app_path = tmp_path / f"app{suffix}"
    write_cloud(app_path, cloud, binary=binary)
    v, n, rgb = cv2.loadPointCloud(str(app_path))
    np.testing.assert_array_equal(v.reshape(-1, 3), cloud.points)
    np.testing.assert_array_equal(n.reshape(-1, 3), cloud.normals)
    np.testing.assert_array_equal(np.rint(rgb.reshape(-1, 3) * 255), cloud.colors)

    native_path = tmp_path / f"native{suffix}"
    cv2.savePointCloud(
        str(native_path),
        cloud.points[:, None, :],
        cloud.normals[:, None, :],
        cloud.colors.astype(np.float32)[:, None, :] / 255,
    )
    actual = read_cloud(native_path, units=cloud.units, frame=cloud.frame)
    _assert_attributes(actual, cloud, atol=1e-6, rtol=5e-6)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"v 1 2 3\nv 4 oops 6\n", "line 2: Invalid OBJ number"),
        (b"v 1_2 3 4\n", "Invalid OBJ number"),
        (b"v nan 0 1\n", "must be finite"),
        (b"v 1e100 0 1\n", "float32 range"),
        (b"v 0 0 1 0.5\n", "weight must be one"),
        (b"v 1 2 3 0 1\n", "vertex requires"),
        (b"v 1 2 3 255 0 0\n", "RGB values must be normalized"),
        (b"v 1 2 3 1 0 0\nv 4 5 6\n", "every vertex or omitted"),
        (b"v 1 2 3\nvn 0 0 0\n", "nonzero length"),
        (b"v 1 2 3\nv 4 5 6\nvn 0 0 1\n", "one ordered normal per vertex"),
        (b"v 1 2 3\nf 1//1 2//1 3//1\n", "Unsupported OBJ point-cloud directive 'f'"),
        (b"# dataflow_metadata []\nv 1 2 3\n", "Invalid OBJ dataflow metadata"),
        (b"# dataflow_metadata {}\n# dataflow_metadata {}\n", "Duplicate OBJ"),
        (b"#" + b" " * 8192 + b"\n", "exceeds 8192 bytes"),
    ],
)
def test_obj_rejects_malformed_or_unaligned_records(tmp_path, content, error):
    path = tmp_path / "malformed.obj"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        read_cloud(path)


def test_obj_point_and_normal_limits_precede_conversion(tmp_path):
    path = tmp_path / "two.obj"
    path.write_text("v 0 0 0\nv 1 1 1\n", encoding="ascii")
    with pytest.raises(ValueError, match="vertex count exceeds"):
        read_cloud(path, max_points=1)
    path.write_text("v 0 0 0\nvn 0 0 1\nvn 0 1 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="normal count exceeds"):
        read_cloud(path, max_points=1)


def test_failed_obj_write_removes_staging_file_and_withholds_destination(tmp_path, monkeypatch):
    path = tmp_path / "new.obj"

    def fail(stream, rows, fmt):
        stream.write(b"v 0")
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savetxt", fail)
    with pytest.raises(OSError, match="injected"):
        write_cloud(path, PointCloud([[0, 0, 0]]))
    assert not list(tmp_path.iterdir())
