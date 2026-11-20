"""Qualify adjacent-grid topology, optical coordinates and bounded failures."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.mesh_io import read_mesh, write_mesh
from opencv_dataflow.organized_mesh import MeshOrganizedDepth
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/organized_depth_mesh"


def _camera(depth):
    h, w = depth.shape
    return camera_record([[100, 0, 2.5], [0, 120, 1.5], [0, 0, 1]], [], (w, h), frame="optic")


def _pixels(mesh, size):
    w, h = size
    return np.floor(mesh.texcoords * [w, -h] + [0, h]).astype(int)


def _run(depth, **parameters):
    return MeshOrganizedDepth(parameters=parameters).execute(
        {"depth": depth, "camera": _camera(depth)}
    )


def test_metric_plane_has_two_triangles_per_cell_camera_facing_normals_and_uv(tmp_path):
    depth = DepthArray(np.full((5, 7), 800), "mm")
    result = _run(depth, output_units="mm")
    assert result.passed
    mesh, report = (result.outputs[key] for key in ("mesh", "report"))
    assert (mesh.frame, mesh.units) == ("optic", "mm")
    assert report["full_cells"] == 24 and report["face_count"] == 48
    assert (
        report["source_grid"] == [7, 5]
        and report["camera_matrix"] == _camera(depth)["camera_matrix"]
    )
    pixels = _pixels(mesh, (7, 5))
    x, y = pixels.T
    assert list(map(tuple, pixels)) == [(x, y) for y, x in product(range(5), range(7))]
    np.testing.assert_allclose(
        mesh.vertices, np.c_[(x - 2.5) * 8, (y - 1.5) * 800 / 120, x * 0 + 800], atol=1e-4
    )
    np.testing.assert_array_equal(mesh.normals, np.tile([0, 0, -1], (35, 1)))
    xyz = mesh.vertices[mesh.faces]
    assert (np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])[:, 2] < 0).all()
    path = tmp_path / "grid.ply"
    write_mesh(path, mesh)
    restored = read_mesh(path)
    assert (restored.units, restored.frame) == ("mm", "optic")
    np.testing.assert_array_equal(restored.faces, mesh.faces)
    np.testing.assert_array_equal(_pixels(restored, (7, 5)), pixels)


def test_depth_gate_keeps_three_sample_cells_without_crossing_invalid_or_step():
    depth = np.full((4, 6), 800, np.uint16)
    depth[:, 3:] = 1100
    depth[1, 1] = 65535
    result = _run(depth)
    assert result.passed
    mesh, report = (result.outputs[k] for k in ("mesh", "report"))
    assert (report["face_count"], report["empty_cells"], report["partial_cells"]) == (20, 3, 4)
    triangle_pixels = _pixels(mesh, (6, 4))[mesh.faces]
    assert np.all(np.ptp(triangle_pixels, axis=1) <= 1)
    assert not np.any((triangle_pixels == [1, 1]).all(axis=2))
    assert np.max(np.ptp(mesh.vertices[mesh.faces, 2], axis=1)) < 1e-6
    loose = _run(depth, max_depth_jump_m=0.35)
    assert loose.outputs["report"]["face_count"] == report["face_count"] + 6
    edge = _run(depth, max_depth_jump_m=0.35, max_edge_length_m=0.03)
    np.testing.assert_array_equal(edge.outputs["mesh"].faces, mesh.faces)


def test_mask_and_missing_rows_withhold_mesh_and_report_the_reason():
    depth = DepthArray(np.ones((3, 5)))
    camera = _camera(depth)
    mask = np.zeros(depth.shape, np.uint8)
    mask[0] = 255
    result = MeshOrganizedDepth().execute({"depth": depth, "camera": camera, "mask": mask})
    assert not result.passed and result.outputs["mesh"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "organized_mesh_no_surface"
    assert result.outputs["report"]["empty_cells"] == 8
    budget = _run(depth, max_faces=3)
    assert budget.outputs["mesh"] is NO_VALUE
    assert budget.fail_reasons[0]["code"] == "organized_mesh_face_limit"
    later_batch = _run(DepthArray(np.ones((100, 100))), max_faces=18_000)
    assert later_batch.outputs["mesh"] is NO_VALUE
    assert later_batch.fail_reasons[0]["code"] == "organized_mesh_face_limit"
    bad_frame = DepthArray(depth)
    bad_frame.metadata["optical_frame"] = "other"
    tool = MeshOrganizedDepth()
    assert tool.execute({"depth": bad_frame, "camera": camera}) is None
    assert "optical frame" in tool.error


def test_organized_cloud_uses_rotated_viewpoint_rgb_and_reordered_pixels():
    pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    x, y = pixels.T
    z = np.where(x == 2, 1000, 800)
    root = np.sqrt(0.5)
    cloud = PointCloud(
        np.c_[20 + z, 30 + 11 * y, 40 - 9 * x][::-1],
        "mm",
        "rig",
        colors=np.c_[15 * x, 180 * y, x + y][::-1],
        pixels=pixels[::-1],
        source_size=(3, 2),
        metadata={
            "pcd_viewpoint": {
                "frame": "rig",
                "units": "mm",
                "values": [20, 30, 40, root, 0, root, 0],
            }
        },
    )
    result = MeshOrganizedDepth().execute({"cloud": cloud})
    assert result.passed
    mesh, report = (result.outputs[k] for k in ("mesh", "report"))
    assert report["viewpoint"] == "pcd_viewpoint" and report["face_count"] == 2
    assert report["viewpoint_origin_m"] == [0.02, 0.03, 0.04]
    indices = [0, 2, 1, 3]
    np.testing.assert_array_equal(_pixels(mesh, (3, 2)), pixels[indices])
    np.testing.assert_array_equal(mesh.colors, cloud.colors[::-1][indices])
    np.testing.assert_allclose(mesh.normals, np.tile([-1, 0, 0], (4, 1)))
    duplicate = cloud.subset([0, 0])
    tool = MeshOrganizedDepth()
    assert tool.execute({"cloud": duplicate}) is None
    assert "multiple points" in tool.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_keeps_oval_and_changes_only_depth_step_faces(profile):
    graph = Graph.load(EXAMPLE / "organized_depth_mesh.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        strict, loose = (graph.blocks[k].last_result for k in ("strict", "loose"))
        assert strict.passed and loose.passed
        mesh = strict.outputs["mesh"]
        assert mesh.frame == "step_camera"
        assert len(loose.outputs["mesh"].faces) - len(mesh.faces) == 54
        pixels = _pixels(mesh, (42, 28))
        assert not np.any((pixels == [10, 14]).all(axis=1))
        assert np.unique(mesh.vertices[:, 2]).size == 2
        assert np.max(np.ptp(mesh.vertices[mesh.faces, 2], axis=1)) < 1e-6
        assert strict.payloads["mesh"]["organized_mesh"]["source_grid"] == [42, 28]
        assert strict.payloads["mesh"]["ticket"] == loose.payloads["mesh"]["ticket"]
    finally:
        graph.close()
