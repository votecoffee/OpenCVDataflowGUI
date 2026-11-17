"""Authored radiance/motion truths through native HDR, alignment and saved graphs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/hdr_motion"
pytestmark = pytest.mark.local_examples


def fixture_module():
    spec = importlib.util.spec_from_file_location("hdr_motion_fixture", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "case", ["static", "moving", "strict", "ignore_clipping", "missing", "aligned"]
)
def test_saved_hdr_merge_uses_only_independently_supported_radiance(profile, case):
    expected = json.loads((BASE / "truth.json").read_bytes())["cases"][case]
    _, radiance, valid, rejected, _ = fixture_module().controls(case)
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["hdr"]
        result = block.last_result
        assert block.state == ("pass" if expected["expected_pass"] else "fail"), block.error
        assert not any(b.state == "error" for b in graph.processing_blocks())
        assert result is not None
        report = result.outputs["report"]
        assert report["measured_pixels"] == expected["measured_pixels"]
        assert report["inconsistent_pixels"] == expected["inconsistent_pixels"]
        assert report["display_filled_pixels"] == expected["fallback_pixels"]
        assert report["coverage"] == expected["coverage"]
        assert (
            hashlib.sha256(result.outputs["valid_mask"]).hexdigest()
            == expected["valid_mask_sha256"]
        )
        assert (
            hashlib.sha256(result.outputs["motion_mask"]).hexdigest()
            == expected["motion_mask_sha256"]
        )
        np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, valid)
        np.testing.assert_array_equal(result.outputs["motion_mask"] != 0, rejected)
        np.testing.assert_array_equal(result.outputs["image"][valid], radiance[valid])
        assert not result.outputs["image"][~valid].any()
        np.testing.assert_array_equal(result.outputs["fallback_mask"] != 0, ~valid)
        assert not result.outputs["contributors"][~valid].any()
        statistics = graph.blocks["statistics"].last_result.outputs
        assert statistics["selected_count"] == expected["measured_pixels"]
        assert statistics["value"] == pytest.approx(expected["selected_mean_radiance"], abs=1e-6)
        if case == "aligned":
            info = graph.blocks["align"].last_result.outputs["info"]
            assert [[p["dx"], p["dy"]] for p in info["shifts"]] == expected["shifts_to_reference"]
        if case in ("moving", "strict"):
            conventional = graph.blocks["conventional"].last_result.outputs["image"]
            reference = graph.blocks["exposure_1"].last_result.outputs["image"].astype(float) / 64
            assert np.mean(abs(conventional[54:64, 20:32] - reference[54:64, 20:32])) > 0.1
        assert (
            result.payloads["image"]["spatial"]["root_id"]
            == graph.blocks["exposure_1"].last_result.payloads["image"]["spatial"]["root_id"]
        )
        assert result.payloads["display"]["hdr_support"]["review_only"] is True
        assert graph.blocks["filmstrip"].last_result is not None
        assert bool(result.drawings) is (profile == "debug")
        if profile == "debug":
            assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
    finally:
        graph.close()


def test_all_original_hdr_assets_and_projects_regenerate_exactly_without_changing_sources(tmp_path):
    module = fixture_module()
    paths = (
        ["truth.json"]
        + [f"{case}.json" for case in module.CASES]
        + [f"assets/{case}_{i}.png" for case in module.CASES for i in range(3)]
    )
    before = {p: (BASE / p).read_bytes() for p in paths}
    module.HERE = tmp_path
    module.build()
    assert {p: (tmp_path / p).read_bytes() for p in paths} == before
    assert {p: (BASE / p).read_bytes() for p in paths} == before
    truth = json.loads(before["truth.json"])
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256(before["assets/" + name]).hexdigest() == digest
