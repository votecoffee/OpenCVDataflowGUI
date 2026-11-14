"""Saved S1-118 examples show accepted area, fractional reuse and frame rejection."""

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/mask_resampling"


@pytest.mark.parametrize("case", ["coverage_thresholds", "fractional_chain", "incompatible_target"])
def test_saved_project_results_and_provenance(case):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.run()
        assert "MIT" in graph.blocks["scope"].description
        if case == "coverage_thresholds":
            results = [graph.blocks[name].last_result for name in ("full", "half", "any")]
            counts = [r.outputs["info"]["accepted_pixels"] for r in results]
            assert counts[0] < counts[1] < counts[2]
            for name, result in zip(("full", "half", "any"), results, strict=True):
                assert result.outputs["mask"].shape == (96, 128)
                assert result.outputs["info"]["target_frame_verified"]
                assert (
                    graph.blocks[f"measure_{name}"].last_result.outputs["area"]
                    == (result.outputs["info"]["accepted_pixels"])
                )
            assert all(b.state in ("pass", "inactive") for b in graph.processing_blocks())
        elif case == "fractional_chain":
            chained = graph.blocks["fractional"].last_result.outputs
            direct = graph.blocks["direct"].last_result.outputs
            thresholded = graph.blocks["thresholded_first"].last_result.outputs
            np.testing.assert_array_equal(chained["coverage"], direct["coverage"])
            np.testing.assert_array_equal(chained["mask"], direct["mask"])
            assert thresholded["info"]["mean_coverage"] < chained["info"]["mean_coverage"]
            assert not np.array_equal(thresholded["mask"], chained["mask"])
            assert all(b.state in ("pass", "inactive") for b in graph.processing_blocks())
        else:
            block = graph.blocks["rejected"]
            assert block.error_kind == "invalid" and "coordinate frame" in block.error
            assert not block.current_emissions
    finally:
        graph.close()


def test_projects_and_png_regenerate_with_identical_bytes():
    paths = sorted([*BASE.glob("*.json"), BASE / "mask.png"])
    assert len(paths) == 4
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
