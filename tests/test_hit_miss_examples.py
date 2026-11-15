"""Original public binary patterns, known anchors and deliberate pass/fail/error outcomes."""

import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/hit_or_miss"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", TRUTH["cases"])
def test_saved_example_points_masks_judgments_and_save_as(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        if case == "invalid_binary":
            block = graph.blocks["isolated"]
            assert block.state == "error" and block.error_kind == "invalid"
            assert not block.current_emissions
            return
        for block in graph.processing_blocks():
            if block.type_id != "cv.hit_or_miss":
                assert block.state == "pass", (block.id, block.error)
                continue
            expected = (
                TRUTH["points"][block.id]
                if block.id in TRUTH["points"]
                else [[0, 0], *TRUTH["points"]["isolated"]]
            )
            if case == "selected_centers":
                expected = [[36, 12]]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["points"] == expected
            assert result.outputs["count"] == len(expected)
            mask = np.zeros((80, 112), np.uint8)
            for x, y in expected:
                mask[y, x] = 255
            np.testing.assert_array_equal(result.outputs["image"], mask)
            assert result.passed == (case != "count_failure")
            assert bool(result.fail_reasons) == (case == "count_failure")
            if case == "count_failure":
                assert result.fail_reasons[0]["code"] == "hit_or_miss_count_out_of_range"
                assert "expected 4..4" in result.fail_reasons[0]["message"]
            assert result.payloads["image"]["spatial"] == block.input_payloads["image"]["spatial"]
            if profile == "debug":
                assert len(result.drawings["binary matches"].objects) == len(expected)
            else:
                assert not result.drawings
        assert "MIT" in graph.blocks["scope"].description
        graph.save(tmp_path / "saved.xml")
        reopened = Graph.load(tmp_path / "saved.xml", builtin_registry())
        try:
            reopened.run()
            assert reopened.blocks["isolated"].last_result.outputs["count"] == (
                1 if case == "selected_centers" else 3
            )
        finally:
            reopened.close()
    finally:
        graph.close()


def test_original_asset_truth_and_deterministic_regeneration():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*.png")])
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    scope = runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    source = cv2.imread(str(BASE / "assets/patterns.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(source, scope["fixture"]())
