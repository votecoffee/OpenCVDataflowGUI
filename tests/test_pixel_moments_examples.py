"""Recorded stationary noise, paired/channel covariance and explicit negative cases."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/temporal_pixel_statistics"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_temporal_replays_match_independent_covariance(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.save(tmp_path / "program.json")
        graph.set_run_profile(profile)
        runs = 0
        while not graph.sources_exhausted():
            assert runs < 9
            graph.run()
            runs += 1
            block = graph.blocks["statistics"]
            if case in ("ticket_mismatch", "memory_limit"):
                assert block.state == "error" and block.error_kind == "invalid", block.error
                assert not block.current_emissions and not block.memory["streams"]
            else:
                assert block.last_result is not None, block.error
                assert block.last_result.outputs["ready"] is (runs > 1)
        assert runs == 8
        if case in ("ticket_mismatch", "memory_limit"):
            return
        result = block.last_result
        prefix = "color" if case == "channel_covariance" else "x"
        x = np.stack(
            [
                cv2.imread(str(BASE / f"assets/{prefix}_{i:02d}.png"), cv2.IMREAD_UNCHANGED)
                for i in range(8)
            ]
        ).astype(np.float64)
        if case == "channel_covariance":
            y, x = x[..., 1], x[..., 0]
        elif case == "paired_sources":
            y = np.stack(
                [
                    cv2.imread(str(BASE / f"assets/y_{i:02d}.png"), cv2.IMREAD_UNCHANGED)
                    for i in range(8)
                ]
            ).astype(np.float64)
        else:
            y = x
        weights = 0.25 * 0.75 ** np.arange(7, -1, -1) if case == "exponential" else np.ones(8)
        denominator = weights.sum() - (weights**2).sum() / weights.sum()
        mx, my = (np.average(values, weights=weights, axis=0) for values in (x, y))
        vx = np.sum(weights[:, None, None] * (x - mx) ** 2, axis=0) / denominator
        covariance = np.sum(weights[:, None, None] * (x - mx) * (y - my), axis=0) / denominator
        valid = result.outputs["valid"] != 0
        if case == "masked_history":
            yy, xx = np.indices(valid.shape)
            assert valid[24, 32] and not valid[0, 0]
            assert not valid[(xx - 32) ** 2 + (yy - 24) ** 2 > 17**2].any()
        else:
            assert valid.all()
        for name, expected in (("mean_x", mx), ("variance_x", vx), ("covariance", covariance)):
            np.testing.assert_allclose(result.outputs[name][valid], expected[valid], atol=2e-10)
            assert not result.outputs[name][~valid].any()
        assert np.all(result.outputs["count"][valid] == 8)
        assert bool(result.graphics) is (profile == "debug")
        if case == "stationary_noise":
            np.testing.assert_allclose(result.outputs["std_x"][:, :32], 2.0, atol=1e-12)
            np.testing.assert_allclose(result.outputs["std_x"][:, 32:], 4.0, atol=1e-12)
    finally:
        graph.close()


def test_original_signals_pixels_hashes_and_regeneration_are_reproducible():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    module = runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    for prefix, frames in zip(("x", "y", "color"), module["references"](), strict=True):
        for index, image in enumerate(frames):
            np.testing.assert_array_equal(
                cv2.imread(str(BASE / f"assets/{prefix}_{index:02d}.png"), cv2.IMREAD_UNCHANGED),
                image,
            )
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
