"""Saved examples preserve controlled lighting and independent defect truth."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/differential_exposure"


@pytest.mark.parametrize(
    "case", ["local_lighting", "local_defect", "global_color", "insufficient_fit"]
)
def test_saved_example_outcome_and_measurements(case):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.run()
        block = graph.blocks["inspect"]
        assert not block.error
        result = block.last_result
        assert result is not None
        assert result.passed == (case in ("local_lighting", "global_color"))
        assert "MIT" in graph.blocks["scope"].description
        info = result.outputs["info"]
        truth = json.loads((BASE / "truth.json").read_text())
        if case == "local_lighting":
            assert info["raw"]["rmse"] == pytest.approx(truth["raw_good_rmse"])
            assert info["adjusted"]["rmse"] < 1 and info["valid_fraction"] == 1
        elif case == "local_defect":
            expected = np.zeros((128, 192), np.uint8)
            x, y, w, h = truth["defect_xywh"]
            expected[y : y + h, x : x + w] = 255
            np.testing.assert_array_equal(result.outputs["defect_mask"], expected)
            assert info["defect_pixels"] == truth["expected_defect_pixels"]
            assert info["fit_inlier_pixels"] == 128 * 192 - 20
            assert result.drawings
        elif case == "global_color":
            assert result.outputs["adjusted"].dtype == np.uint16
            assert len(info["channel_statistics"]) == 3
            for c in range(3):
                np.testing.assert_allclose(
                    result.outputs["gain"][..., c], truth["color_reference_gain_bgr"][c]
                )
                np.testing.assert_allclose(
                    result.outputs["offset"][..., c],
                    truth["color_reference_offset_bgr"][c],
                    atol=1e-9,
                )
                assert info["channel_statistics"][c]["adjusted"]["maximum"] < 1e-8
        else:
            assert info["valid_fraction"] < 0.1
            assert "exposure_insufficient_support" in [r["code"] for r in result.fail_reasons]
    finally:
        graph.close()


def test_fixture_and_project_regeneration_is_byte_identical():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*.png")])
    assert len(paths) == 10
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
