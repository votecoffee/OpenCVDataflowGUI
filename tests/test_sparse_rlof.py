"""Known motion, native sparse RLOF parity, and saved correspondence graphs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.match_review import paired_points
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/sparse_rlof"
TYPE = "cv.sparse_rlof"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture(autouse=True)
def native_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


@pytest.mark.parametrize(("support", "with_guess"), [
    ("fixed", False), ("fixed", True), ("cross", False), ("cross", True),
])
def test_wired_tracking_parity_with_native_row_vector(registry, support, with_guess):
    shape = (96, 128) if support == "fixed" else (96, 128, 3)
    source = np.random.default_rng(31).integers(0, 256, shape, dtype=np.uint8)
    target = np.roll(source, (-3, 5), axis=(0, 1))
    points = np.array([[25, 25], [48, 32], [80, 22], [90, 74]], np.float32).reshape(1, -1, 2)
    guess = points + [4, -2] if with_guess else None
    inputs = {"image": source, "target": target, "points": points[0].tolist()}
    if guess is not None:
        inputs["initial_points"] = guess[0].tolist()
    block = registry.create(TYPE, parameters={
        "support": support, "solver": "standard", "max_level": 3, "fb_tolerance": 2.0,
    })
    result = block.execute(inputs)
    assert result and result.passed, block.error

    optflow = cv2.optflow
    native = optflow.RLOFOpticalFlowParameter_create()
    native.setSupportRegionType(optflow.SR_FIXED if support == "fixed" else optflow.SR_CROSS)
    native.setSolverType(optflow.ST_STANDART)
    native.setSmallWinSize(9)
    native.setLargeWinSize(21)
    native.setCrossSegmentationThreshold(25)
    native.setUseGlobalMotionPrior(False)
    native.setUseIlluminationModel(True)
    native.setUseMEstimator(False)
    native.setMaxLevel(3)
    native.setMaxIteration(30)
    native.setMinEigenValue(0.0001)
    native.setUseInitialFlow(with_guess)
    found, statuses, errors = optflow.calcOpticalFlowSparseRLOF(
        source, target, points, None if guess is None else guess.astype(np.float32),
        rlofParam=native, forwardBackwardThreshold=2.0,
    )
    tracks = result.outputs["tracks"]["items"]
    np.testing.assert_allclose([r["target"] for r in tracks], found[0], atol=1e-6)
    np.testing.assert_array_equal([r["status"] for r in tracks], statuses.ravel().astype(bool))
    np.testing.assert_allclose([r["fb_error"] for r in tracks], errors.ravel(), atol=1e-6)
    np.testing.assert_allclose(result.outputs["report"]["median_displacement"], [5, -3], atol=0.06)
    assert result.drawings["sparse tracks"].source == "image"


def test_rejection_keeps_input_order_and_reverses_only_eligible_points(registry, monkeypatch):
    calls = []

    def solve(a, b, points, guess, **kwargs):
        calls.append(points[0].tolist())
        assert points.shape[0] == 1 and kwargs["forwardBackwardThreshold"] == 0
        if len(calls) == 1:
            return np.array([[[16, 14], [-10, 12], [91, 60], [55, 27], [np.nan, 20]]], np.float32), None, None
        assert calls[-1] == [[16, 14], [55, 27]]
        return np.array([[[14, 14], [58, 27]]], np.float32), None, None

    monkeypatch.setattr(cv2.optflow, "calcOpticalFlowSparseRLOF", solve)
    image = np.zeros((80, 112), np.uint8)
    mask = np.full(image.shape, 255, np.uint8)
    mask[49, 46] = 0
    points = [[14, 14], [24, 12], [26, 25], [54, 27], [59, 20], [46, 49]]
    block = registry.create(TYPE, parameters={"fb_tolerance": 0.8, "max_displacement": 10, "minimum_tracks": 2})
    result = block.execute({"image": image, "target": image, "mask": mask, "points": points})
    assert result and not result.passed and block.error == ""
    rows = result.outputs["tracks"]["items"]
    assert [r["index"] for r in rows] == list(range(6))
    assert [r["reason"] for r in rows] == [
        None, "outside_target", "displacement_limit", "round_trip_limit", "nonfinite_estimate", "outside_selection",
    ]
    assert [r["fb_error"] for r in rows] == [0, None, None, 4, None, None]
    matches = result.outputs["matches"]
    assert matches["source_indices"] == [0] and matches["query_points"] == [[14, 14]]
    assert result.outputs["target_points"] == [[16, 14]]
    assert result.fail_reasons[0]["code"] == "sparse_rlof_insufficient_tracks"


def test_source_and_target_point_ports_keep_distinct_tickets_and_frames(registry):
    shape = (80, 100)
    source = np.random.default_rng(20).integers(0, 256, shape, dtype=np.uint8)
    target = np.roll(source, 3, axis=1)
    clock = TicketClock("points")
    contexts = [
        {"ticket": clock.issue("source", "fixed", key), "spatial": frame([100, 80], key)}
        for key in ("a", "b")
    ]
    inputs = {
        "image": ValuePacket(source, contexts[0]),
        "target": ValuePacket(target, contexts[1]),
        "points": ValuePacket([[28, 41], [43, 60]], contexts[0]),
        "initial_points": ValuePacket([[31, 41], [46, 60]], contexts[1]),
    }
    tracking = registry.create(TYPE, parameters={"fb_tolerance": 0, "support": "fixed"})
    result = tracking.execute(inputs)
    assert result and result.passed, tracking.error
    assert result.payloads["target_points"]["spatial"] == contexts[1]["spatial"]
    assert result.payloads["target_points"]["ticket"] == contexts[1]["ticket"]
    result_match = result.outputs["matches"]
    assert result_match["query_context"]["ticket"] == contexts[0]["ticket"]
    assert result_match["train_context"]["ticket"] == contexts[1]["ticket"]
    assert all(row["fb_error"] is None for row in result.outputs["tracks"]["items"])
    np.testing.assert_allclose(result.outputs["target_points"], [[31, 41], [46, 60]], atol=0.06)
    assert tracking.execute({**inputs, "points": ValuePacket([[28, 41]], contexts[1])}) is None
    assert "coordinate frame" in tracking.error
    same_grid_wrong_ticket = {**contexts[0], "ticket": contexts[1]["ticket"]}
    assert tracking.execute({**inputs, "points": ValuePacket([[28, 41]], same_grid_wrong_ticket)}) is None
    assert "different ticket" in tracking.error


@pytest.mark.parametrize(("settings", "shape", "points", "guess", "error"), [
    ({"global_motion_prior": True}, (29, 64), [[20, 10]], None, "at least 40"),
    ({"global_motion_prior": True}, (32, 64), [[20, 10]], [[24, 10]], "initial points or"),
    ({}, (32, 64), [[20, 10]], [], "counts must match"),
    ({}, (32, 64), [[float("inf"), 10]], None, "JSON compliant"),
    ({}, (32, 64), [[63.999999999, 10]], None, "rounds outside"),
    ({"small_window": 31}, (32, 64), [[20, 10]], None, "larger than"),
])
def test_invalid_inputs_never_enter_native_solver(registry, monkeypatch, settings, shape, points, guess, error):
    monkeypatch.setattr(cv2.optflow, "calcOpticalFlowSparseRLOF", lambda *a, **k: pytest.fail("entered native solver"))
    block = registry.create(TYPE, parameters=settings)
    args = {"image": np.zeros(shape, np.uint8), "target": np.zeros(shape, np.uint8), "points": points}
    if guess is not None:
        args["initial_points"] = guess
    assert block.execute(args) is None
    assert error in block.error


def test_empty_point_list_reports_normal_inspection_failure(registry, monkeypatch):
    monkeypatch.setattr(cv2.optflow, "calcOpticalFlowSparseRLOF", lambda *a, **k: pytest.fail("entered native solver"))
    block = registry.create(TYPE)
    image = np.zeros((32, 64), np.uint8)
    result = block.execute({"image": image, "target": image, "points": []})
    assert result and not result.passed and block.error == ""
    assert result.outputs["target_points"] == []
    assert result.outputs["tracks"]["items"] == []
    assert result.outputs["report"]["mean_fb_error"] is None


@pytest.mark.parametrize("mode", ("debug", "lean"))
def test_saved_pair_and_occlusion_pass_fail_without_plugins(registry, mode):
    for name, accepted, count in (("translated", True, 36), ("occluded", False, 20)):
        graph = Graph.load(EXAMPLE / f"{name}.json", registry)
        try:
            graph.set_run_profile(mode)
            graph.set_branch_workers(3)
            graph.run()
            block = graph.blocks["track"]
            assert block_entry(block).path[:2] == ("OpenCV", "Optical Flow Algorithms (optflow)")
            result = block.last_result
            assert result and result.passed == accepted, block.error
            assert bool(result.drawings) == (mode == "debug")
            report = result.outputs["report"]
            assert report["input_count"] == 36 and report["accepted_count"] == count
            assert report["mean_fb_error"] < 0.02
            np.testing.assert_allclose(report["median_displacement"], [5, -3], atol=0.01)
            matches = result.outputs["matches"]
            source, target = paired_points(matches, [240, 160], [240, 160])
            assert source.shape == target.shape == (count, 2)
            assert graph.blocks["review"].last_result.passed
        finally:
            graph.close()
