"""Exercise organized PCD rows, invalid samples and interoperability layouts."""

import struct
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import LoadCloud, SaveCloud, read_cloud, write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/organized_pcd"


@pytest.fixture
def organized_binary(tmp_path):
    path = tmp_path / "padded.pcd"
    header = (
        b"VERSION 0.7\n"
        b"FIELDS z _ feature x _ y normal_x normal_y normal_z rgba\n"
        b"SIZE 4 1 2 4 1 4 4 4 4 4\n"
        b"TYPE F U I F U F F F F F\n"
        b"COUNT 1 1 2 1 1 1 1 1 1 1\n"
        b"WIDTH 3\nHEIGHT 2\nVIEWPOINT 0.25 -0.5 3 0 1 0 0\nPOINTS 6\nDATA binary\n"
    )
    rows = []
    for i in range(6):
        z = {1: float("nan"), 5: float("inf")}.get(i, 9.5)
        normal = (0, 0, 0) if i == 3 else (0, -3, 0)
        # Alpha 0xff decorates a float NaN bit pattern. RGB must be read bitwise.
        color = 0xFFC21234 + i
        rows.append(struct.pack("<fB2hfBf3fI", z, 67, -56, 86, i, 43, -i, *normal, color))
    path.write_bytes(header + b"".join(rows))
    return path


def test_binary_offsets_counts_padding_nonfinite_and_packed_color(organized_binary):
    cloud = read_cloud(organized_binary, units="mm", frame="camera")
    np.testing.assert_array_equal(cloud.pixels, [[0, 0], [2, 0], [1, 1]])
    np.testing.assert_array_equal(cloud.points, [[0, 0, 9.5], [2, -2, 9.5], [4, -4, 9.5]])
    np.testing.assert_array_equal(cloud.colors, [[194, 18, 52], [194, 18, 54], [194, 18, 56]])
    np.testing.assert_array_equal(cloud.normals, [[0, -1, 0]] * 3)
    assert cloud.source_size == (3, 2)
    assert cloud.metadata["pcd_viewpoint"] == {
        "frame": "camera",
        "units": "mm",
        "values": [0.25, -0.5, 3, 0, 1, 0, 0],
    }
    with pytest.raises(ValueError, match="including holes"):
        read_cloud(organized_binary, max_points=5)


@pytest.mark.parametrize("encoding", ["ascii", "binary"])
def test_save_block_restores_grid_after_subset_and_reordering(tmp_path, organized_binary, encoding):
    loaded = LoadCloud(parameters={"path": str(organized_binary), "units": "mm"}).execute({})
    cloud = loaded.outputs["cloud"].subset([2, 0])
    out = tmp_path / "réseau.pcd"
    tool = SaveCloud(
        parameters={
            "path": str(out),
            "pcd_encoding": encoding,
            "pcd_organization": "source_pixels",
        }
    )
    result = tool.execute({"cloud": cloud})
    assert result.passed
    data = out.read_bytes()
    header, body = data.split(f"DATA {encoding}\n".encode(), 1)
    assert b"WIDTH 3\nHEIGHT 2\n" in header and b"POINTS 6\n" in header
    if encoding == "binary":
        rows = list(struct.iter_unpack("<6fI", body))
        assert len(rows) == 6
        assert np.isnan(np.array(rows)[[1, 2, 3, 5], :6]).all()
        assert rows[4][-1] == 0xC21238
    else:
        assert body.splitlines()[4].endswith(b"12718648")
        assert body.splitlines()[2] == b"nan nan nan nan nan nan 0"
    reopened = read_cloud(out)
    assert reopened.source_size == (3, 2)
    assert reopened.units == "mm" and reopened.metadata == cloud.metadata
    for attr in ("points", "normals", "pixels", "colors"):
        np.testing.assert_array_equal(getattr(reopened, attr), getattr(cloud, attr)[::-1])
    assert tool.execute({"cloud": cloud}) is None
    assert out.read_bytes() == data


@pytest.mark.parametrize(
    "old,new,error",
    [
        (b"HEIGHT 2", b"HEIGHT 3", "POINTS does not match"),
        (b"SIZE 4 1 2", b"SIZE 4 1", "must describe the same fields"),
        (b"TYPE F U I", b"TYPE F U X", "Unsupported PCD type"),
        (b"DATA binary", b"DATA binary_compressed", "uncompressed"),
        (b"VIEWPOINT 0.25 -0.5 3 0 1 0 0", b"VIEWPOINT 0 0 0 0 0 0 0", "VIEWPOINT"),
    ],
)
def test_reject_malformed_binary_headers(organized_binary, old, new, error):
    path = organized_binary.with_name("invalid.pcd")
    path.write_bytes(organized_binary.read_bytes().replace(old, new, 1))
    with pytest.raises(ValueError, match=error):
        read_cloud(path)


@pytest.mark.parametrize("extra", [False, True])
def test_reject_binary_body_length_mismatch(organized_binary, extra):
    data = organized_binary.read_bytes()
    path = organized_binary.with_name("wrong-length.pcd")
    path.write_bytes(data + b"\x00" if extra else data[:-1])
    with pytest.raises(ValueError, match="body length"):
        read_cloud(path)


def test_legacy_ascii_and_typed_colors_validate_rows(tmp_path):
    path = tmp_path / "legacy.pcd"
    path.write_bytes(b"FIELDS x y z\nPOINTS 1\nDATA ascii\n1 2 3\n")
    cloud = read_cloud(path)
    assert cloud.source_size is None and cloud.pixels is None
    np.testing.assert_array_equal(cloud.points, [[1, 2, 3]])
    header = b"FIELDS x y z rgb\nTYPE F F F U\nPOINTS 1\nDATA ascii\n"
    path.write_bytes(header + b"1 2 3 4294967296\n")
    with pytest.raises(ValueError, match="value in rgb"):
        read_cloud(path)
    path.write_bytes(header + b"1 2 3 123\n2 3 4 321\n")
    with pytest.raises(ValueError, match="beyond its declared"):
        read_cloud(path)


def test_source_grid_export_requires_bounded_unique_pixels(tmp_path):
    path = tmp_path / "organized.pcd"
    plain = PointCloud([[0, 0, 0]])
    duplicate = PointCloud([[0, 0, 0], [1, 1, 1]], pixels=[[0, 1]] * 2, source_size=(1, 2))
    large = PointCloud([[0, 0, 0]], pixels=[[0, 0]], source_size=(4096, 2048))
    for cloud, match in (
        (plain, "source pixel grid"),
        (duplicate, "same source pixel"),
        (large, "2 million"),
    ):
        with pytest.raises(ValueError, match=match):
            write_cloud(path, cloud, True, pcd_organization="source_pixels")
        assert not path.exists()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_preserves_grid_and_geometry(profile):
    graph = Graph.load(EXAMPLE / "organized_pcd.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        binary, ascii_cloud = (
            graph.blocks[key].last_result.outputs["cloud"] for key in ("binary", "ascii")
        )
        assert binary.source_size == ascii_cloud.source_size == (64, 44)
        assert 2000 < len(binary.points) < 64 * 44
        assert binary.frame == "reference_scan" and binary.units == "mm"
        for attr in ("points", "normals", "pixels", "colors"):
            np.testing.assert_array_equal(getattr(binary, attr), getattr(ascii_cloud, attr))
        for key in ("view_binary", "view_ascii"):
            assert graph.blocks[key].last_result.passed
    finally:
        graph.close()
