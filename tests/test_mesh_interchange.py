"""Verify triangle topology, per-vertex attributes and native interchange."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import read_mesh, write_mesh
from opencv_dataflow.meshes import TriangleMesh

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/mesh_interchange/mesh_interchange.json"
)
PLY_HEADER = (
    b"ply\nformat ascii 1.0\nelement vertex 3\n"
    b"property float x\nproperty float y\nproperty float z\n"
    b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
)
OBJ_VERTICES = b"v 0 0 0\nv 2 0 0\nv 0 2 0\n"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_mesh_graph_has_identical_topology_and_attributes(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        results = {kind: graph.blocks[f"load_{kind}"].last_result for kind in ("obj", "ply")}
        assert all(value and value.passed for value in results.values())
        for result in results.values():
            assert result.outputs["ticket"]
            assert ("Mesh wireframe" in result.graphics) == (profile == "debug")
        assert results["obj"].outputs["report"] == results["ply"].outputs["report"]
        first, second = (results[kind].outputs["mesh"] for kind in ("obj", "ply"))
        for name in ("vertices", "faces", "normals", "colors", "texcoords"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        assert first.faces.tolist() == [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]]
        assert (first.units, first.frame) == ("mm", "folded_panel")
        assert first.nbytes == sum(
            getattr(first, name).nbytes
            for name in ("vertices", "faces", "normals", "colors", "texcoords")
        )
    finally:
        graph.close()


@pytest.mark.parametrize("suffix", [".obj", ".ply"])
def test_legacy_save_block_writes_both_formats_and_preserves_existing_file(tmp_path, suffix):
    mesh = TriangleMesh(
        [[0, 0, 1], [1.2345678, 0, 1], [0, 1, 1]],
        [[0, 1, 2]],
        "mm",
        "bench \u2603",
        normals=[[0, 0, 1]] * 3,
        colors=[[0, 128, 255]] * 3,
        texcoords=[[0, 0], [1, 0], [0, 1]],
    )
    path = tmp_path / f"surface-\u2603{suffix.upper()}"
    registry = builtin_registry()
    save = registry.create("output.mesh_ply", parameters={"path": str(path)})
    assert save.execute({"mesh": mesh}).passed
    original = path.read_bytes()
    load = registry.create("input.mesh", parameters={"path": str(path)})
    result = load.execute({})
    assert result and result.passed, load.error
    restored = result.outputs["mesh"]
    for field in ("vertices", "faces", "normals", "colors", "texcoords"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(mesh, field))
    assert (restored.units, restored.frame) == (mesh.units, mesh.frame)
    assert save.execute({"mesh": mesh}) is None
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("suffix", [".obj", ".ply"])
def test_opencv_can_read_app_meshes_and_app_reads_native_meshes(tmp_path, suffix):
    mesh = TriangleMesh(
        [[0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 2]],
        [[0, 1, 2], [2, 1, 3]],
        normals=[[0, 0, 1]] * 4,
        colors=[[200, 0, 79], [0, 128, 255], [0, 100, 40], [1, 35, 238]],
        texcoords=[[0, 0, 0], [1, 0, 0.25], [0, 1, 0.5], [1, 1, 0.75]],
    )
    path = tmp_path / f"app{suffix}"
    write_mesh(path, mesh)
    v, f, n, c, uvw = cv2.loadMesh(str(path))
    for actual, expected in (
        (v, mesh.vertices),
        (f, mesh.faces),
        (n, mesh.normals),
        (np.rint(c * 255), mesh.colors),
        (uvw, mesh.texcoords),
    ):
        np.testing.assert_array_equal(np.asarray(actual).reshape(expected.shape), expected)

    path = tmp_path / f"native{suffix}"
    cv2.saveMesh(
        str(path),
        mesh.vertices[:, None, :],
        mesh.faces,
        mesh.normals[:, None, :],
        mesh.colors[:, None, :].astype(np.float32) / 255,
        mesh.texcoords[:, None, :],
    )
    restored = read_mesh(path, units="mm", frame="native")
    assert (restored.units, restored.frame) == ("mm", "native")
    for field in ("vertices", "faces", "normals", "colors", "texcoords"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(mesh, field))


def test_obj_splits_normal_and_texture_seams_without_changing_triangle_winding(tmp_path):
    path = tmp_path / "seams.obj"
    path.write_bytes(
        OBJ_VERTICES + b"v 2 2 1\n"
        b"vt 0 0\nvt 1 0\nvt 0 1\nvt .25 .75\n"
        b"vn 0 0 1\nvn 0 1 0\n"
        b"f 1/1/1 2/2/1 3/3/1\nf -4/-1/-1 -2/-2/-2 -1/-3/-1\n"
    )
    mesh = read_mesh(path)
    assert len(mesh.vertices) == 5
    np.testing.assert_array_equal(mesh.faces, [[0, 2, 3], [1, 3, 4]])
    np.testing.assert_array_equal(mesh.vertices[:2], [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(mesh.normals[:2], [[0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.texcoords[:2], [[0, 0], [0.25, 0.75]])
    with pytest.raises(ValueError, match="after splitting"):
        read_mesh(path, max_vertices=4)


@pytest.mark.parametrize(
    "byteorder, encoding", [("<", "binary_little_endian"), (">", "binary_big_endian")]
)
def test_read_binary_ply_face_indices_with_declared_endianness(tmp_path, byteorder, encoding):
    path = tmp_path / "binary.ply"
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=byteorder + "f4")
    face = np.array([(3, [2, 1, 0])], dtype=[("count", "u1"), ("index", byteorder + "i4", 3)])
    path.write_bytes(
        PLY_HEADER.replace(b"ascii", encoding.encode()) + points.tobytes() + face.tobytes()
    )
    mesh = read_mesh(path)
    np.testing.assert_array_equal(mesh.vertices, points)
    np.testing.assert_array_equal(mesh.faces, [[2, 1, 0]])


@pytest.mark.parametrize(
    ("suffix", "data", "message"),
    [
        (".obj", OBJ_VERTICES + b"f 1 2 0\n", "face index"),
        (".obj", OBJ_VERTICES + b"f 1 2 4\n", "face index"),
        (".obj", OBJ_VERTICES + b"f -4 2 3\n", "face index"),
        (".obj", OBJ_VERTICES + b"f 1 2 3 1\n", "must be triangles"),
        (".obj", OBJ_VERTICES + b"vn 0 0 1\nf 1//1 2 3//1\n", "attributes must be complete"),
        (".obj", OBJ_VERTICES + b"vn 0 0 1\nf 1//1 1//1 3//1\n", "distinct vertex"),
        (".obj", OBJ_VERTICES + b"vt 0 0\nvt 0 0 0\nf 1 2 3\n", "consistent width"),
        (".ply", PLY_HEADER + b"0 0 0\n2 0 0\n0 2 0\n4 0 1 2 0\n", "exactly three"),
        (".ply", PLY_HEADER + b"0 0 0\n2 0 0\n0 2 0\n3 0 1 4294967296\n", "outside"),
        (".ply", PLY_HEADER + b"0 0 0\n2 0 0\n0 2 0\n3 0 1 2\n1 2 3\n", "Unexpected data"),
        (
            ".ply",
            PLY_HEADER.replace(b"property float z", b"property float unknown"),
            "vertex property",
        ),
    ],
)
def test_malformed_topology_and_unsupported_layouts_are_rejected(tmp_path, suffix, data, message):
    path = tmp_path / f"bad{suffix}"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        read_mesh(path)
