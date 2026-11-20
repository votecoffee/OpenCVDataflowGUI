"""Validate Census costs, signed support, confidence and saved stereo graphs."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import binary_stereo
from opencv_dataflow.binary_stereo import BinaryStereoDisparity
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import digest
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/binary_stereo/binary_stereo.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_example_compares_integer_binary_and_native_sgbm_on_same_masked_pair(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for name in ("census", "sgbm", "census_errors", "sgbm_errors"):
            block = graph.blocks[name]
            assert block.last_result and block.last_result.passed, block.error
        block = graph.blocks["census"]
        result = block.last_result
        outputs = result.outputs
        native = graph.blocks["sgbm"].last_result.outputs
        np.testing.assert_array_equal(outputs["support_mask"], native["support_mask"])
        valid = outputs["valid_mask"] != 0
        assert int(valid.sum()) == 14040
        assert outputs["report"]["supported_pixels"] == 14684
        assert outputs["report"]["left_right_rejected_pixels"] == 644
        assert np.all(outputs["disparity"][valid] == 8)
        assert np.isnan(outputs["disparity"][~valid]).all()
        assert np.isfinite(outputs["confidence"][outputs["support_mask"] != 0]).all()
        assert np.isnan(outputs["best_cost"][outputs["support_mask"] == 0]).all()
        source = graph.blocks["left"].last_result
        for field in ("spatial", "ticket"):
            assert result.payloads["disparity"][field] == source.payloads["image"][field]
        census = graph.blocks["census_errors"].last_result.outputs["report"]
        assert census["coverage"] == 14040 / 19200 and census["mae_px"] == 0
        sgbm = graph.blocks["sgbm_errors"].last_result.outputs["report"]
        assert sgbm["coverage"] == 14684 / 19200 and sgbm["mae_px"] < 0.005
        if profile == "debug":
            assert result.graphics["Binary disparity"].shape == (100, 192, 3)
        else:
            assert not result.graphics
    finally:
        graph.close()


def _scalar_cost(a, b, x, y, d):
    total = 0
    for j in range(y - 1, y + 2):
        for i in range(x - 1, x + 2):
            neighborhood_a = a[j - 1 : j + 2, i - 1 : i + 2] < a[j, i]
            neighborhood_b = b[j - 1 : j + 2, i - d - 1 : i - d + 2] < b[j, i - d]
            total += np.count_nonzero(neighborhood_a != neighborhood_b)
    return total


def test_costs_and_signed_support_match_independent_scalar_neighborhood_calculation():
    rng = np.random.default_rng(745)
    h, w = 12, 25
    a, b = rng.integers(0, 20, (2, h, w), dtype=np.uint8)
    lm = np.full(a.shape, 255, np.uint8)
    rm = lm.copy()
    lm[4, 11] = 0
    rm[8, 12] = 0
    block = BinaryStereoDisparity(
        parameters={
            "census_size": "3",
            "aggregation_size": 3,
            "min_disparity": -3,
            "num_disparities": 6,
            "left_right_check": False,
            "minimum_confidence": 0,
        }
    )
    result = block.execute({"left": a, "right": b, "left_mask": lm, "right_mask": rm})
    assert result and result.passed, block.error
    out = result.outputs
    expected = np.full((h, w, 4), np.nan)
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            candidates = list(range(-3, 3))
            if not lm[y - 2 : y + 3, x - 2 : x + 3].all():
                continue
            if any(
                not 2 <= x - d < w - 2 or not rm[y - 2 : y + 3, x - d - 2 : x - d + 3].all()
                for d in candidates
            ):
                continue
            ranked = sorted((_scalar_cost(a, b, x, y, d), d) for d in candidates)
            (best, winner), (second, _) = ranked[:2]
            expected[y, x] = (
                best,
                second,
                (second - best) / second if second else 0,
                winner if second > best else np.nan,
            )
    for index, name in enumerate(("best_cost", "second_best_cost", "confidence", "disparity")):
        np.testing.assert_allclose(out[name], expected[..., index], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(out["support_mask"], np.isfinite(expected[..., 0]) * 255)
    np.testing.assert_array_equal(out["valid_mask"], np.isfinite(expected[..., 3]) * 255)


def _negative_shift():
    rng = np.random.default_rng(523)
    a = rng.integers(0, 120, (45, 80), dtype=np.uint8)
    b = np.zeros_like(a)
    b[:, 6:] = a[:, :-6] + 61
    return a, b


def test_depth_bounded_signed_search_preserves_sealed_grid_and_exact_negative_disparity():
    a, b = _negative_shift()
    h, w = a.shape
    rec = {
        "model": "stereo_rectification_v1",
        "units": "m",
        "image_size": [w, h],
        "roi_left": [2, 1, w - 4, h - 2],
        "roi_right": [3, 2, w - 6, h - 4],
        "P2": [[200, 0, 0, 24], [0, 200, 0, 0], [0, 0, 1, 0]],
        "Q": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 200], [0, 0, -1 / 0.12, 0]],
    }
    inputs = {"rectification": rec}
    for side, image in (("left", a), ("right", b)):
        spatial = frame((w, h), side)
        stamp = {"sha256": digest(rec), "side": side, "spatial": spatial}
        inputs[side] = ValuePacket(image, {"spatial": spatial, "stereo_calibration_grid": stamp})
    block = BinaryStereoDisparity(
        parameters={"search_mode": "depth", "minimum_depth": 3.5, "maximum_depth": 5.5}
    )
    result = block.execute(inputs)
    assert result and result.passed, block.error
    outputs = result.outputs
    report = outputs["report"]
    assert (report["min_disparity"], report["max_disparity"], report["num_disparities"]) == (
        -7,
        -4,
        4,
    )
    assert report["expected_disparity_px"] == pytest.approx([-24 / 3.5, -24 / 5.5])
    assert report["matched_pixels"] > 1900
    assert np.all(outputs["disparity"][outputs["valid_mask"] != 0] == -6)
    assert result.payloads["disparity"] == inputs["left"].payload
    changed = {**rec, "roi_left": [0, 0, w, h]}
    assert block.execute({**inputs, "rectification": changed}) is None
    assert "differs from the sealed" in block.error


def test_flat_texture_is_ambiguous_even_at_zero_threshold():
    a = np.full((25, 53), 30, np.uint8)
    block = BinaryStereoDisparity(
        parameters={"minimum_confidence": 0, "num_disparities": 7, "aggregation_size": 1}
    )
    result = block.execute({"left": a, "right": a})
    assert result and not result.passed and not block.error
    assert result.fail_reasons[0]["code"] == "binary_stereo_no_matches"
    outputs = result.outputs
    supported = outputs["support_mask"] != 0
    assert supported.any()
    assert outputs["report"]["ambiguous_pixels"] == supported.sum()
    assert not outputs["valid_mask"].any()
    assert (outputs["best_cost"][supported] == 0).all()
    assert (outputs["second_best_cost"][supported] == 0).all()
    assert (outputs["confidence"][supported] == 0).all()


def test_empty_support_and_work_budget_are_checked_before_census(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("No descriptors should be computed")

    monkeypatch.setattr(binary_stereo, "_census", unexpected)
    monkeypatch.setattr(binary_stereo, "_MAX_COMPARISONS", 1)
    a, b = _negative_shift()
    block = BinaryStereoDisparity(parameters={"min_disparity": -8, "num_disparities": 6})
    inputs = {"left": a, "right": b}
    empty = block.execute({**inputs, "right_mask": np.zeros_like(a)})
    assert empty and not empty.passed
    assert empty.fail_reasons[0]["code"] == "stereo_search_no_support"
    assert np.isnan(empty.outputs["confidence"]).all()
    checkerboard = ((np.indices(a.shape).sum(axis=0) % 2) * 255).astype(np.uint8)
    sparse = block.execute({**inputs, "right_mask": checkerboard})
    assert sparse and not sparse.passed and not block.error
    assert sparse.fail_reasons[0]["code"] == "stereo_search_no_support"
    assert block.execute(inputs) is None
    assert "work limit" in block.error


def test_wrong_mask_frame_and_conflicting_tickets_are_rejected():
    a, b = _negative_shift()
    size = a.shape[::-1]
    clock = TicketClock("stereo")
    left_ticket, right_ticket = (clock.issue(key, "fixed", key) for key in ("a", "b"))
    block = BinaryStereoDisparity(parameters={"min_disparity": -8, "num_disparities": 6})
    assert (
        block.execute(
            {
                "left": ValuePacket(a, {"ticket": left_ticket}),
                "right": ValuePacket(b, {"ticket": right_ticket}),
            }
        )
        is None
    )
    assert "same ticket" in block.error
    assert (
        block.execute(
            {
                "left": ValuePacket(a, {"spatial": frame(size, "left")}),
                "right": ValuePacket(b, {"spatial": frame(size, "right")}),
                "right_mask": ValuePacket(np.full_like(a, 255), {"spatial": frame(size, "left")}),
            }
        )
        is None
    )
    assert "coordinate frame" in block.error
