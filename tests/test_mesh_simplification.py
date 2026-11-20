"""Spot-check bounded decimation, indexed boundaries and source attributes."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_distance import TriangleSurface
from opencv_dataflow.mesh_io import read_mesh
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/mesh_simplification"


@pytest.fixture
def panel():
    return read_mesh(EXAMPLE / "assets/crowned_panel.ply")


def simplify(mesh, **params):
    block = builtin_registry().create("3d.mesh_simplify", parameters=params)
    result = block.execute({"mesh": mesh})
    assert result is not None, block.error
    return result, result.outputs["report"]


def boundary_edges(mesh):
    counts = Counter(
        tuple(sorted((tuple(mesh.vertices[a]), tuple(mesh.vertices[b]))))
        for face in mesh.faces
        for a, b in zip(face, np.roll(face, 1), strict=True)
    )
    return {edge for edge, count in counts.items() if count == 1}


def check_attributes(source, reduced):
    rows = {tuple(value): i for i, value in enumerate(source.vertices)}
    keep = [rows[tuple(value)] for value in reduced.vertices]
    assert (reduced.units, reduced.frame) == (source.units, source.frame)
    for name in ("normals", "colors", "texcoords"):
        np.testing.assert_array_equal(getattr(source, name)[keep], getattr(reduced, name))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_reaches_budget_and_preserves_boundaries_and_samples(profile, panel):
    graph = Graph.load(EXAMPLE / "mesh_simplification.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        review, strict = (graph.blocks[key].last_result for key in ("review", "strict"))
        assert review is not None and review.passed
        assert strict is not None and not strict.passed
        report = review.outputs["report"]
        reduced = review.outputs["mesh"]
        assert (report["input_faces"], report["output_faces"], report["boundary_edges"]) == (
            544,
            220,
            80,
        )
        assert 0 < report["surface_deviation_bound"] <= 35
        assert len(boundary_edges(reduced)) == 80
        assert boundary_edges(panel) == boundary_edges(reduced)
        check_attributes(panel, reduced)
        assert ("Simplified mesh" in review.graphics) == (profile == "debug")
        control = strict.outputs["report"]
        assert control["status"] == "constraints" and control["output_faces"] == 544
        assert control["surface_deviation_bound"] == 0
    finally:
        graph.close()


@pytest.mark.local_examples
def test_two_way_bound_contains_independent_exact_surface_distances(panel):
    result, report = simplify(
        panel, target_faces=220, deviation_mode="Mesh units", max_deviation=35
    )
    assert result.passed
    reduced = result.outputs["mesh"]
    maximum = report["surface_deviation_bound"]
    for source, target in ((panel, reduced), (reduced, panel)):
        triangles = source.vertices.astype(np.float64)[source.faces]
        weights = np.array([[0.6, 0.3, 0.1], [0, 0.5, 0.5], [0.1, 0.8, 0.1]])
        probes = (weights @ triangles).reshape(-1, 3)
        surface = TriangleSurface(target.vertices.astype(np.float64), target.faces, 45)
        matches, _ = surface.query(
            probes,
            maximum + 1e-7,
            2_000_000,
            lambda: False,
        )
        assert matches is not None and (matches.face >= 0).all()
        assert matches.distance.max() <= maximum + 1e-7


@pytest.mark.local_examples
def test_work_budget_returns_a_qualified_partial_mesh(panel):
    result, report = simplify(
        panel, target_faces=80, deviation_mode="Mesh units", max_deviation=35, max_edge_checks=1
    )
    assert not result.passed
    assert report["status"] == "edge_check_limit" and report["edge_checks"] == 1
    assert report["output_faces"] >= 542 and report["surface_deviation_bound"] <= 35
    assert boundary_edges(result.outputs["mesh"]) == boundary_edges(panel)


def test_closed_octahedron_reduces_to_tetrahedron_without_disappearing():
    points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    top = [(4, index, (index + 1) % 4) for index in range(4)]
    bottom = [(5, (index + 1) % 4, index) for index in range(4)]
    mesh = TriangleMesh(points, top + bottom)
    result, report = simplify(
        mesh, target_faces=3, deviation_mode="Mesh units", max_deviation=3, max_normal_change_deg=89
    )
    output = result.outputs["mesh"]
    assert not result.passed and report["status"] == "constraints"
    assert len(output.faces) == 4 and len(output.vertices) == 4
    assert not boundary_edges(output)


@pytest.mark.local_examples
def test_nonmanifold_and_degenerate_input_fail_without_emitting_mesh(panel):
    duplicated = TriangleMesh(panel.vertices, np.vstack((panel.faces, panel.faces[5])))
    vertices = panel.vertices.copy()
    vertices[0] = vertices[1]
    collapsed = TriangleMesh(vertices, panel.faces)
    for mesh, defect in ((duplicated, "duplicate_faces"), (collapsed, "degenerate_faces")):
        result, report = simplify(mesh)
        assert not result.passed and report["status"] == "invalid_topology"
        assert report[defect] and result.outputs["mesh"] is NO_VALUE


@pytest.mark.local_examples
def test_input_face_ceiling_rejects_coarse_configuration(panel):
    block = builtin_registry().create("3d.mesh_simplify", parameters={"max_input_faces": 100})
    assert block.execute({"mesh": panel}) is None
    assert "input face limit" in block.error
