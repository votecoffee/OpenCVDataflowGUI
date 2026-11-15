"""Real-photo replay has explicit software truth and a frozen reference."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/camera_health"
TRUTH = json.loads((BASE / "truth.json").read_text())


@pytest.mark.parametrize("case,condition", TRUTH["cases"].items())
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_real_photo_examples_and_known_changes(case, condition, profile):
    reference_bytes = (BASE / "reference.npz").read_bytes()
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["monitor"]
        assert not block.error
        result = block.last_result
        assert result is not None
        expected = condition["expected"]
        assert result.outputs["state"] == expected
        assert result.passed == (expected != "alarm")
        assert result.outputs["warning"] == (expected == "warning")
        assert result.outputs["alarm"] == (expected == "alarm")
        assert bool(result.fail_reasons) == (expected == "alarm")
        info = result.outputs["report"]
        assert info["capture_count"] == 3 and info["reference_is_frozen"]
        assert info["reference_sha256"] == TRUTH["reference_sha256"]
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        assert "CC BY 4.0" in graph.blocks["scope"].description
        assert "VisA" in graph.blocks["scope"].description
        assert (BASE / "reference.npz").read_bytes() == reference_bytes
        if not condition.get("exclude_all"):
            assert info["foreground_pixels"] > 1000
            assert info["position"]["displacement_px"] == pytest.approx(
                condition.get("shift_px", 0), abs=0.15
            )
            assert all(
                r["coverage"] >= block.parameters["minimum_region_coverage"]
                for r in info["checks"]
                if "coverage" in r
            )
        if "gain" in condition:
            lights = [r for r in info["checks"] if r["check"] == "lighting"]
            assert len(lights) == 2 and all(r["state"] == expected for r in lights)
            for row in lights:
                np.testing.assert_allclose(
                    row["channel_change_percent"], (condition["gain"] - 1) * 100, atol=0.15
                )
            assert all(r["state"] == "healthy" for r in info["checks"] if r["check"] == "focus")
        if "blur_sigma" in condition:
            focus = [r for r in info["checks"] if r["check"] == "focus"]
            assert len(focus) == 2 and all(r["state"] == expected for r in focus)
            assert (
                all(20 <= r["value"] < 40 for r in focus)
                if expected == "warning"
                else all(r["value"] >= 40 for r in focus)
            )
        if condition.get("mode") == "fiducials":
            assert all(row["found"] for row in info["position"]["fiducials"])
        if condition.get("exclude_all"):
            assert info["position"]["state"] == "unavailable"
            assert all(row["state"] == "alarm" for row in info["checks"])
            assert not result.outputs["background_mask"].any()
    finally:
        graph.close()


def test_rebuild_is_identical_and_all_asset_hashes_match():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("*.npz"), *BASE.glob("assets/*.png")])
    assert len(paths) == 25
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
