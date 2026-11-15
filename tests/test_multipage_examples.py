"""Public TIFF memory projects, normal scheduling and reproducible original assets."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multipage_memory"
CASES = (
    "typed_roundtrip",
    "raw_roundtrip",
    "selected_pages",
    "pixel_budget_guard",
    "predictor_guard",
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_projects_normal_scheduling_pixels_and_page_tickets(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        for _ in range(3):
            graph.run()
        assert graph.blocks["source"].exhausted
        if "guard" in case:
            failed = graph.blocks["decode" if case == "pixel_budget_guard" else "encode"]
            assert failed.state == "error" and failed.error_kind == "invalid"
            assert not failed.current_emissions
            return
        assert all(b.state == "pass" for b in graph.processing_blocks())
        assert graph.blocks["verify"].last_result.outputs["value"] == 0
        result = graph.blocks["decode"].last_result
        pages = result.outputs["pages"].items
        indexes = [0, 2] if case == "selected_pages" else [0, 1, 2]
        assert result.outputs["info"]["selected_indexes"] == indexes
        assert len(graph.blocks["pages"].current_emissions["value"]) == len(indexes)
        assert [p.payload["multipage"]["page_index"] for p in pages] == indexes
        if case != "raw_roundtrip":
            assert [p.payload["ticket"]["value"] for p in pages] == [
                f"mixed-pages.tiff#page-{i:06d}" for i in indexes
            ]
        else:
            assert not result.outputs["info"]["restored_page_context"]
        assert "MIT" in graph.blocks["scope"].description
        # Save As keeps the ordinary file source accessible without copying private assets.
        path = tmp_path / "saved.json"
        graph.save(path)
        reopened = Graph.load(path, builtin_registry())
        try:
            for _ in range(3):
                reopened.run()
            assert reopened.blocks["verify"].last_result.outputs["value"] == 0
        finally:
            reopened.close()
    finally:
        graph.close()


def test_original_fixtures_truth_and_deterministic_regeneration():
    truth = json.loads((BASE / "truth.json").read_bytes())
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    scope = runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert (
        hashlib.sha256((BASE / "assets/mixed-pages.tiff").read_bytes()).hexdigest()
        == truth["asset_sha256"]
    )
    for page, info in zip(scope["reference_pages"](), truth["pages"], strict=True):
        assert hashlib.sha256(page.tobytes()).hexdigest() == info["samples_sha256"]
        assert page.dtype == np.dtype(info["dtype"])
