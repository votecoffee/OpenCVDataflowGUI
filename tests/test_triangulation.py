"""Verify indexed Delaunay geometry and incomplete subdivision handling."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame


def test_delaunay_keeps_first_duplicate_index_coordinates_and_payload():
    block = builtin_registry().create("inspect.delaunay_triangulation")
    local = np.array([[10, 10], [90, 10], [15, 90], [92, 85], [48, 43]], dtype=float)
    points = local + [9_999_900, -9_999_900]
    spatial = frame([110, 110], "camera")
    result = block.execute(
        {"points": ValuePacket(points[[0, 1, 1, 2, 3, 4]].tolist(), {"spatial": spatial})}
    )
    assert result is not None, block.error
    outputs = result.outputs
    assert outputs["points"] == points.tolist()
    assert outputs["source_indices"] == [0, 1, 3, 4, 5]
    assert outputs["triangles"] == [[0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]
    assert outputs["triangle_count"] == 4
    assert outputs["edges"] == [[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    assert outputs["edge_lengths"] == pytest.approx(
        [np.linalg.norm(local[a] - local[b]) for a, b in outputs["edges"]]
    )
    assert all(result.payloads[name]["spatial"] == spatial for name in outputs)
    drawing = result.drawings["Delaunay edges"]
    assert drawing.source == "points"
    assert [obj.points for obj in drawing.objects] == [
        (tuple(points[a]), tuple(points[b])) for a, b in outputs["edges"]
    ]


def test_delaunay_includes_shallow_face_at_the_hull():
    block = builtin_registry().create("inspect.delaunay_triangulation")
    result = block.execute({"points": [[0, 0], [1, 1e-6], [2, 0], [2, 2], [0, 2]]})
    assert result is not None, block.error
    assert result.outputs["triangle_count"] == 4
    assert [0, 1, 2] in result.outputs["triangles"]


@pytest.mark.parametrize(
    ("points", "error"),
    [
        ([[0, 0], [0, 0], [2, 2]], "three distinct sites"),
        ([[0, 0], [1, 1], [2, 2]], "non-collinear"),
        ([[0, 0], [1, 1], [2, None]], "finite"),
        ([[0, 0], [0, 1e-10], [500, 0], [0, 500]], "coordinate precision"),
        ([[0, 0], [1, 1e-9], [2, 0]], "complete triangulation"),
    ],
)
def test_delaunay_rejects_invalid_or_unresolved_geometry(points, error):
    block = builtin_registry().create("inspect.delaunay_triangulation")
    assert block.execute({"points": points}) is None
    assert error in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_delaunay_example_connects_detected_centers_and_overlays_edges(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/geometry/delaunay_triangulation/delaunay_triangulation.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    contours = graph.blocks["contours"].last_result
    mesh = graph.blocks["triangulation"].last_result
    assert mesh.outputs["triangle_count"] == 4
    centers = [
        [item["metrics"]["center_x"], item["metrics"]["center_y"]]
        for item in contours.outputs["collection"]["items"]
    ]
    assert mesh.outputs["points"] == centers
    assert set(map(tuple, centers)) == {(10, 10), (90, 10), (15, 90), (92, 85), (48, 43)}
    assert mesh.outputs["source_indices"] == list(range(5))
    assert mesh.payloads["points"]["spatial"] == contours.payloads["collection"]["spatial"]
    assert len(mesh.outputs["edges"]) == len(mesh.outputs["result_graphics"].layers[0].objects) == 8
    image = graph.blocks["overlay"].last_result.outputs["image"]
    assert image.shape == (110, 110, 3) and np.any(image[..., 0] != image[..., 1])
