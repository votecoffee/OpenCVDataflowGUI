"""Check mesh topology and volume against fixtures with known geometry."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import read_mesh
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/mesh_validity"


@pytest.fixture
def box():
    return read_mesh(EXAMPLE / "assets/closed_box.ply")


def inspect(mesh, **parameters):
    block = builtin_registry().create("3d.mesh_inspect", parameters=parameters)
    result = block.execute({"mesh": mesh})
    assert result is not None, block.error
    return result, result.outputs["report"]


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_measures_closed_box_and_reports_two_damaged_controls(profile):
    graph = Graph.load(EXAMPLE / "mesh_validity.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        closed, open_box, flap = (
            graph.blocks[key].last_result for key in ("closed_box", "open_box", "flap_box")
        )
        assert closed.passed and not open_box.passed and not flap.passed
        measure = closed.outputs["report"]
        assert measure["length_units"] == "mm" and measure["frame"] == "fixture"
        assert measure["area"] == pytest.approx(11_200)
        assert measure["volume"] == pytest.approx(64_000)
        assert measure["winding"] == "outward"
        hole = open_box.outputs["report"]
        assert (hole["boundary_edges"], hole["boundary_loops"]) == (4, 1)
        assert open_box.outputs["area"] == pytest.approx(8000)
        defect = flap.outputs["report"]
        assert defect["nonmanifold_edges"] == 1
        assert defect["nonmanifold_edge_examples"] == [[0, 1]]
        for result in (open_box, flap):
            assert result.outputs["volume"] is NO_VALUE
            assert ("Topology" in result.graphics) == (profile == "debug")
        if profile == "debug":
            review = flap.graphics["Topology"]
            assert np.any(np.all(review == (73, 75, 250), axis=2))
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("face", "defect"), [([0, 1, 99], "Out-of-range"), ([0, 0, 1], "Repeated")]
)
def test_bad_face_indices_are_rejected_with_face_diagnostics(box, face, defect):
    box.faces[0] = face
    block = builtin_registry().create("3d.mesh_inspect")
    assert block.execute({"mesh": box}) is None
    assert f"{defect} index faces (1): [0]" in block.error


@pytest.mark.local_examples
def test_winding_conflict_differs_from_global_reversal(box):
    flipped = box.faces.copy()
    flipped[0] = flipped[0, ::-1]
    result, report = inspect(TriangleMesh(box.vertices, flipped))
    assert not result.passed and report["inconsistent_winding_edges"] == 3
    result, report = inspect(TriangleMesh(box.vertices, box.faces[:, ::-1]), require_volume=True)
    assert result.passed and report["winding"] == "inward"
    assert result.outputs["volume"] == pytest.approx(64_000)


@pytest.mark.local_examples
def test_duplicate_face_and_triangle_area_floor(box):
    inputs = TriangleMesh(box.vertices, np.vstack((box.faces, box.faces[0, ::-1])))
    result, report = inspect(inputs, minimum_triangle_area=450)
    assert not result.passed
    assert report["duplicate_faces"] == 1
    assert report["degenerate_faces"] == 4
    assert report["nonmanifold_edges"] == 3


@pytest.mark.local_examples
def test_two_closed_shells_with_one_shared_vertex_have_disconnected_fans(box):
    vertices = np.vstack((box.vertices, -box.vertices[1:]))
    second = np.where(box.faces, box.faces + len(box.vertices) - 1, 0)
    result, report = inspect(TriangleMesh(vertices, np.vstack((box.faces, second))))
    assert not result.passed
    assert report["face_components"] == 2
    assert report["split_vertex_examples"] == [0]
    assert report["boundary_edges"] == 0


@pytest.mark.local_examples
def test_volume_withheld_for_multiple_shells_concavity_and_budget(box):
    separate = TriangleMesh(
        np.vstack((box.vertices, box.vertices + [100, 0, 0])),
        np.vstack((box.faces, box.faces + len(box.vertices))),
    )
    concave = TriangleMesh(
        np.vstack((box.vertices, [40, 20, 10])),
        np.vstack((box.faces[:2], box.faces[4:], [[4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]])),
    )
    for mesh, options, status in (
        (separate, {}, "multiple_shells"),
        (concave, {}, "not_convex"),
        (box, {"max_support_tests": 95}, "support_budget"),
    ):
        result, report = inspect(mesh, require_volume=True, **options)
        assert not result.passed and report["closed"]
        assert report["volume_status"] == status
        assert report["area"] > 0 and result.outputs["volume"] is NO_VALUE
