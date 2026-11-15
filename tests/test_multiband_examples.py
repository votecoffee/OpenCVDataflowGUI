"""Saved real-photo workflows and deterministic public derivative assets."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multiband_mosaic"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case,description", TRUTH["cases"].items())
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_real_photo_mosaic_outcomes_depth_and_reports(case, description, profile):
    mode, depth, backend = description
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        for _ in range(1 if mode == "pair" else 3):
            graph.run()
        tools = [
            b
            for b in graph.blocks.values()
            if b.type_id
            in ("cv.stitch_image_pair", "cv.stitch_scan_sequence", "cv.render_mosaic_tile")
        ]
        for block in tools:
            if case == "native_depth_guard":
                assert block.error_kind == "invalid" and "Float64" in block.error
                assert not block.current_emissions
                continue
            assert not block.error and block.last_result.passed
            out = block.last_result.outputs
            image = out["image"]
            assert image.dtype == (np.uint16 if depth == 16 else np.uint8)
            assert image.ndim == 3 and image.shape[2] == 3
            assert int(image.max()) > (20000 if depth == 16 else 80)
            assert out["coverage"].shape == out["overlap"].shape == image.shape[:2]
            assert np.all(out["overlap"] <= out["coverage"])
            assert not image[out["coverage"] == 0].any()
            report = out["tile"] if mode == "tile" else out["layout"]
            info = report["multiband"]
            assert info["backend"] == backend and info["bands"] == 3
            assert 0 < info["workspace_upper_bound_bytes"] <= 256 * 1024**2
            assert info["source_only_pixels_restored"]
            if mode == "pair":
                assert report["image_size"] == TRUTH["nominal_pair_size"]
            if case == "scan_exposure_gain":
                for view, gain in zip(report["views"], TRUTH["view_gains"], strict=True):
                    assert view["exposure"]["gain"] == pytest.approx(1 / gain, abs=0.01)
        assert "CC BY 4.0" in graph.blocks["scope"].description
        assert "VisA" in graph.blocks["scope"].description
    finally:
        graph.close()


def test_regenerated_pngs_projects_and_truth_are_byte_identical():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*.png")])
    assert len(paths) == 14
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
