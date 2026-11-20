"""Metric posed TSDF extraction, observed topology and exclusive export."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.control_flow import byte_size
from opencv_dataflow.core import Graph, Port, validate_value
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, describe_value
from opencv_dataflow.port_types import compatible

pytest.importorskip("skimage.measure")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/skimage tools/tsdf_mesh"
K = [[96, 0, 47.5], [0, 96, 35.5], [0, 0, 1]]
SIZE = (96, 72)
TYPE = "3d.tsdf_mesh"


def test_saved_pose_mesh_matches_world_plane_and_keeps_unobserved_gap():
    graph = Graph.load(EXAMPLE / "tsdf_mesh.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["surface"]
        result = block.last_result
        assert result and result.passed, block.error
        mesh = result.outputs["mesh"]
        assert mesh.units == "m" and mesh.frame == "world"
        assert len(mesh.faces) > 2000
        surface = mesh.vertices
        error = surface[:, 2] - (1.02 + 0.14 * surface[:, 0] - 0.09 * surface[:, 1])
        assert np.max(np.abs(error)) < 0.003
        pose = np.asarray(graph.blocks["pose"].last_result.outputs["transform"]["source_to_target"])
        camera_xyz = (surface - pose[:3, 3]) @ pose[:3, :3]
        u = camera_xyz[:, 0] / camera_xyz[:, 2] * 100 + 47.5
        assert np.any(u < 41) and np.any(u > 55)
        # All faces lie on one side of the gap or the other.
        us = u[mesh.faces]
        assert np.all((us.max(axis=1) < 44.5) | (us.min(axis=1) > 50.5))
        tri = surface[mesh.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert np.all(normals[:, 2] < 0)
        assert result.graphics["Mesh wireframe"].shape == (400, 560, 3)
        summary = describe_value(mesh)
        assert summary["face_count"] == len(mesh.faces) and "vertices" not in summary
        assert summary["bytes"] == byte_size(mesh) == mesh.vertices.nbytes + mesh.faces.nbytes
        assert compatible("mesh", "mesh") and not compatible("mesh", "cloud")
    finally:
        graph.close()


def test_multi_pose_fusion_recovers_support_from_both_views_and_respects_observation_count():
    registry = builtin_registry()
    block = registry.create(TYPE)
    a = np.full(SIZE[::-1], 1000, np.uint16)
    b = np.full(SIZE[::-1], 850, np.uint16)
    ta, tb = np.eye(4), np.eye(4)
    ta[0, 3], tb[0, 3], tb[2, 3] = -150, 150, 150
    inputs = {
        "frames": FlowGroup((ValuePacket(a, {}), ValuePacket(b, {})), ("left", "right")),
        "camera": camera_record(K, [], SIZE),
        "poses": [pose_record(t, "camera", "station", "mm") for t in (ta, tb)],
    }
    block.set_graphics_mode("off")
    result = block.execute(inputs)
    assert result and result.passed and not result.graphics, block.error
    mesh = result.outputs["mesh"]
    assert mesh.frame == "station"
    np.testing.assert_allclose(mesh.vertices[:, 2], 1, atol=5e-7)
    assert mesh.vertices[:, 0].min() < -0.55 and mesh.vertices[:, 0].max() > 0.54
    block.parameters["minimum_observations"] = 2
    result = block.execute(inputs)
    assert result and result.passed, block.error
    shared = result.outputs["mesh"]
    assert np.ptp(shared.vertices[:, 0]) < 0.65 * np.ptp(mesh.vertices[:, 0])
    assert len(shared.faces) < len(mesh.faces)


def test_no_measured_crossing_and_resource_limits_withhold_mesh():
    block = builtin_registry().create(TYPE)
    inputs = {
        "depth": np.zeros(SIZE[::-1], np.uint16),
        "camera": camera_record(K, [], SIZE),
        "poses": pose_record(np.eye(4), "camera", "world"),
    }
    result = block.execute(inputs)
    assert result and not result.passed and result.outputs["mesh"] is NO_VALUE
    assert result.outputs["report"]["observed_voxels"] == 0
    inputs["depth"].fill(1000)
    block.parameters["max_faces"] = 1
    assert block.execute(inputs) is None
    assert "exceeds maximum faces" in block.error
    block.parameters["resolution"] = [256, 256, 256]
    assert block.execute(inputs) is None
    assert "sample limit" in block.error


def test_binary_export_preserves_metadata_topology_and_existing_file(tmp_path):
    mesh = TriangleMesh(
        [[0, 0, 1], [0, 2, 1], [3, 0, 1], [3, 2, 1]], [[0, 1, 2], [2, 1, 3]], "mm", "table"
    )
    validate_value(Port("mesh", "mesh"), mesh)
    with pytest.raises(ValueError, match="distinct vertex"):
        TriangleMesh(mesh.vertices, [[0, 0, 2]])
    with pytest.raises(ValueError, match="distinct vertex"):
        TriangleMesh(mesh.vertices, [[0, 1, 4294967296]])
    target = tmp_path / "control.ply"
    block = builtin_registry().create("output.mesh_ply", parameters={"path": str(target)})
    block.parameters["max_faces"] = 1
    assert block.execute({"mesh": mesh}) is None and not target.exists()
    block.parameters["max_faces"] = 2
    result = block.execute({"mesh": mesh})
    assert result and result.passed, block.error
    before = target.read_bytes()
    header, payload = before.split(b"end_header\n", 1)
    info = next(row for row in header.splitlines() if row.startswith(b"comment "))
    assert json.loads(info.split(b" ", 2)[2]) == {"units": "mm", "frame": "table"}
    rows = np.frombuffer(payload, dtype="<f4", count=12).reshape(-1, 3)
    faces = np.frombuffer(payload[rows.nbytes :], dtype=[("count", "u1"), ("face", "<i4", 3)])
    np.testing.assert_array_equal(rows, mesh.vertices)
    np.testing.assert_array_equal(faces["face"], mesh.faces)
    assert np.all(faces["count"] == 3)
    vertices, indices, *_ = cv2.loadMesh(str(target))
    np.testing.assert_array_equal(vertices.reshape(-1, 3), mesh.vertices)
    np.testing.assert_array_equal(np.asarray(indices).reshape(-1, 3), mesh.faces)
    assert block.execute({"mesh": mesh}) is None
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
