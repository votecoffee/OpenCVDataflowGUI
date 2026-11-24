"""Verify ranked Hausdorff distances against independent pairwise geometry."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.toolbox import block_entry

EXAMPLE = (
    Path(__file__).parents[1] / "examples/opencv tools/hausdorff_distance/hausdorff_distance.json"
)


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("metric", ["L1", "L2"])
@pytest.mark.parametrize("rank", [0.0, 0.6, 1.0])
def test_matches_brute_force_and_ignores_point_order(registry, metric, rank):
    # Different point counts require independent order-statistic indices.
    reference = np.array([[0, 0], [6, 0], [6, 9], [0, 9]], dtype=float)
    sample = np.array([[1, 2], [6, 0], [7, 11], [0, 9], [100, 97]], dtype=float)
    norm = 1 if metric == "L1" else 2
    pairwise = np.linalg.norm(reference[:, None] - sample, ord=norm, axis=2)
    forward = sorted(pairwise.min(axis=1))[int(rank * (len(reference) - 1))]
    reverse = sorted(pairwise.min(axis=0))[int(rank * (len(sample) - 1))]
    block = registry.create(
        "inspect.hausdorff_distance", parameters={"metric": metric, "rank_proportion": rank}
    )
    for query, target in (
        (reference, sample),
        (np.vstack((reference[::-1], reference[-1])), sample[[4, 1, 2, 0, 3]]),
    ):
        result = block.execute({"reference": query.tolist(), "sample": target.tolist()})
        assert result and result.passed, block.error
        assert result.outputs == pytest.approx(
            {
                "distance": max(forward, reverse),
                "reference_to_sample": forward,
                "sample_to_reference": reverse,
            }
        )


def test_invalid_inputs_and_large_offset_precision(registry):
    block = registry.create("inspect.hausdorff_distance")
    result = block.execute(
        {
            "reference": [[999_999.125, -999_999]],
            "sample": [[999_999.25, -999_999]],
        }
    )
    assert result and result.outputs["distance"] == 0.125
    assert block.execute({"reference": [[0, 0]], "sample": []}) is None
    assert "ordered finite" in block.error
    assert block.execute({"reference": [[0, 0]], "sample": [["inf", 0]]}) is None
    assert "ordered finite" in block.error


def test_saved_example_and_inspection_limit(registry):
    graph = Graph.load(EXAMPLE, registry)
    try:
        graph.run()
        expected = {"shift_l2": 5, "shift_l1": 7, "tab_full": 10, "tab_partial": 0}
        for name, distance in expected.items():
            block = graph.blocks[name]
            assert block.last_result and block.last_result.passed, block.error
            assert block.last_result.outputs["distance"] == distance
            assert block_entry(block).path[:2] == ("OpenCV", "Shape Distance and Matching (shape)")
            block.parameters["check_distance"] = True
        direction = graph.blocks["tab_full"].last_result.outputs
        assert direction["reference_to_sample"] == pytest.approx(np.sqrt(10))
        assert direction["sample_to_reference"] == 10
        graph.run()
        for name, distance in expected.items():
            result = graph.blocks[name].last_result
            assert result.passed == (distance <= 5)
            assert result.outputs["distance"] == distance
        (reason,) = graph.blocks["tab_full"].last_result.fail_reasons
        assert reason["code"] == "contour_distance_out_of_tolerance"
        assert "10 exceeds 5" in reason["message"]
    finally:
        graph.close()
