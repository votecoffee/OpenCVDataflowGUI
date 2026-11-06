"""Exercise weighted minimum spanning trees and their saved inspection graph."""

from math import dist, fsum
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


@pytest.mark.parametrize("algorithm", ("prim", "kruskal"))
def test_tree_uses_lower_parallel_weight_and_ignores_self_loops(algorithm):
    block = builtin_registry().create(
        "inspect.minimum_spanning_tree",
        parameters={"algorithm": algorithm, "root": 2, "node_count": 3},
    )
    result = block.execute(
        {"edges": [[0, 1], [1, 2], [0, 2], [0, 0], [1, 0]], "edge_weights": [4, 3, 10, -20, -8]}
    )
    assert result is not None, block.error
    assert result.outputs == {
        "connected": True,
        "edges": [[0, 1], [1, 2]],
        "edge_weights": [-8.0, 3.0],
        "total_weight": -5.0,
    }


def test_weighted_tree_uses_wired_count_and_costs_even_with_points_for_overlay():
    block = builtin_registry().create("inspect.minimum_spanning_tree", parameters={"node_count": 1})
    spatial = frame([100, 100], "camera")
    points = [[5, 5], [10, 5], [90, 5]]
    result = block.execute(
        {
            "node_count": 3,
            "edges": [[0, 1], [0, 2], [1, 2]],
            "edge_weights": [8, 1, 2],
            "points": ValuePacket(points, {"spatial": spatial}),
        }
    )
    assert result is not None, block.error
    assert result.outputs["edges"] == [[0, 2], [1, 2]]
    assert result.outputs["total_weight"] == 3
    assert result.payloads["edges"]["spatial"] == spatial
    assert [line.points for line in result.drawings["Spanning tree"].objects] == [
        ((5, 5), (90, 5)),
        ((10, 5), (90, 5)),
    ]


def test_disconnected_graph_emits_no_partial_tree_and_single_node_has_empty_tree():
    block = builtin_registry().create("inspect.minimum_spanning_tree")
    result = block.execute({"node_count": 3, "edges": [[0, 1]], "edge_weights": [1]})
    assert result is not None and not result.passed
    assert result.outputs["connected"] is False
    assert all(result.outputs[key] is NO_VALUE for key in ("edges", "edge_weights", "total_weight"))
    assert result.fail_reasons[0]["code"] == "spanning_tree_disconnected"
    assert not result.drawings
    result = block.execute({"node_count": 1, "edges": []})
    assert result is not None and result.passed
    assert result.outputs == {"connected": True, "edges": [], "edge_weights": [], "total_weight": 0}


@pytest.mark.parametrize(
    ("inputs", "parameters", "message"),
    [
        ({"edges": [[0, 3]], "edge_weights": [5]}, {}, "whole node indices"),
        ({"edges": [[0, 0.5]], "edge_weights": [5]}, {}, "whole node indices"),
        ({"edges": [[0, 1]], "edge_weights": []}, {}, "one finite weight"),
        ({"edges": [[0, 1]]}, {}, "define edge costs"),
        ({"edges": [], "points": [[0, 0]]}, {}, "match the number of points"),
        ({"edges": []}, {"algorithm": "prim", "root": 3}, "Prim root"),
        (
            {
                "edges": ValuePacket([[0, 1]], {"spatial": frame([100, 100], "a")}),
                "points": ValuePacket(
                    [[0, 0], [1, 1], [2, 2]], {"spatial": frame([100, 100], "b")}
                ),
            },
            {},
            "same coordinate frame",
        ),
    ],
)
def test_tree_rejects_invalid_graphs(inputs, parameters, message):
    block = builtin_registry().create(
        "inspect.minimum_spanning_tree", parameters={"node_count": 3, **parameters}
    )
    assert block.execute(inputs) is None
    assert message in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_spanning_tree_example_connects_delaunay_sites_and_draws_edges(profile):
    project = (
        Path(__file__).parents[1] / "examples/minimum_spanning_tree/minimum_spanning_tree.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    points = graph.blocks["triangulation"].last_result.outputs["points"]
    tree = graph.blocks["tree"].last_result
    center = points.index([48, 43])
    assert len(tree.outputs["edges"]) == 4
    assert all(center in pair for pair in tree.outputs["edges"])
    assert tree.outputs["total_weight"] == pytest.approx(
        fsum(dist(point, points[center]) for point in points)
    )
    assert len(tree.outputs["result_graphics"].layers[0].objects) == 4
    image = graph.blocks["overlay"].last_result.outputs["image"]
    assert image.shape == (110, 110, 3) and np.any(image[..., 0] != image[..., 1])
