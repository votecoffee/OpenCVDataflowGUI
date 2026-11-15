"""Public saved projects, independent expected geometry and guarded native limitations."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/run_contours"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_geometry_and_expected_judgments(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["runs"]
        if case == "run_budget_guard":
            assert block.state == "error" and block.error_kind == "invalid"
            assert not block.current_emissions
            return
        result = block.last_result
        assert result is not None, block.error
        assert result.passed == (case != "count_failure")
        assert result.outputs["data"]["hierarchy_semantics"] == "flat_list"
        if case == "nested_boundaries":
            assert sorted(result.outputs["data"]["areas"]) == TRUTH["nested_run_areas"]
            assert (
                sorted(graph.blocks["dense"].last_result.outputs["data"]["areas"])
                == TRUTH["nested_dense_areas"]
            )
            assert (
                sorted(
                    n["depth"]
                    for n in graph.blocks["tree"].last_result.outputs["hierarchy"]["nodes"]
                )
                == TRUTH["nested_tree_depths"]
            )
        elif case == "selection_offset":
            item = result.outputs["collection"]["items"][0]
            m = item["metrics"]
            assert [m[k] for k in ("x", "y", "width", "height")] == TRUTH["clip_shift_bounds"]
            assert m["area"] == TRUTH["clip_shift_area"]
        else:
            assert sorted(result.outputs["data"]["areas"]) == sorted(TRUTH["sparse_polygon_areas"])
            assert result.outputs["data"]["horizontal_runs"] == TRUTH["sparse_runs"]
            if case == "sparse_compare":
                assert graph.blocks["verify"].last_result.outputs["value"] == 0
        assert "MIT" in graph.blocks["scope"].description
        graph.save(tmp_path / "saved.xml")
        reopened = Graph.load(tmp_path / "saved.xml", builtin_registry())
        try:
            reopened.run()
            assert reopened.blocks["runs"].last_result.outputs["data"] == result.outputs["data"]
        finally:
            reopened.close()
    finally:
        graph.close()


def test_original_masks_and_projects_regenerate_identically():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*.png")])
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
