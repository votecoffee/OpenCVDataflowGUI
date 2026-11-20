"""Qualify signed native stereo pairs, support footprints and calibrated reprojection."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.quasi_dense_stereo import QuasiDenseDisparity
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/quasi_dense_stereo"


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("direction", "profile", "sign"),
    [
        ("forward", "debug", 1),
        ("reverse", "lean", -1),
    ],
)
def test_saved_example_retains_sign_and_reprojects_both_planes(direction, profile, sign):
    graph = Graph.load(EXAMPLE / f"quasi_dense_{direction}.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for block in graph.processing_blocks():
            assert block.last_result and block.last_result.passed, (block.id, block.error)
        match = graph.blocks["matcher"].last_result
        report = match.outputs["report"]
        assert report["native_sparse_candidates"] > 100
        assert report["matched_pixels"] > 40 * report["native_sparse_candidates"]
        assert report["rejected_by_support_or_geometry"] > 0
        values = match.outputs["disparity"]
        observed = match.outputs["valid_mask"] != 0
        np.testing.assert_array_equal(np.isfinite(values), observed)
        np.testing.assert_array_equal(
            np.unique(values[observed]), np.sort(np.array([6, 12]) * sign)
        )
        assert np.all(match.outputs["vertical_offset"][observed] == 0)
        assert np.isnan(match.outputs["vertical_offset"][~observed]).all()
        # The accepted patches must avoid a hole in the right-view mask.
        right = graph.blocks["exclusion"].last_result.outputs["mask"]
        kernel = np.ones((11, 11), np.uint8)
        right = cv2.erode(right, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        y, x = np.nonzero(observed)
        assert np.all(right[y, x - values[y, x].astype(int)] != 0)
        pairs = match.outputs["matches"]
        assert len(pairs["query_points"]) == len(pairs["train_points"]) == 256
        assert pairs["total_matches"] == int(observed.sum())
        q, t = (np.asarray(pairs[name]) for name in ("query_points", "train_points"))
        np.testing.assert_array_equal(values[q[:, 1], q[:, 0]], q[:, 0] - t[:, 0])
        rectified = graph.blocks["rectify"].last_result.payloads["left"]
        assert match.payloads["disparity"]["spatial"] == rectified["spatial"]
        assert match.payloads["disparity"]["ticket"] == rectified["ticket"]

        measured = graph.blocks["evaluate"].last_result.outputs["report"]
        assert measured["coverage"] > 0.70 and measured["rmse_px"] < 0.15
        assert measured["bad_fraction"] < 0.001
        depth = graph.blocks["cloud"].last_result.outputs["depth"]
        np.testing.assert_allclose(np.unique(depth[observed]), [1.5, 3.0], rtol=1e-6)
        assert np.isnan(depth[~observed]).all()
        assert bool(match.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_flat_input_and_empty_support_report_no_matches(monkeypatch):
    blank = np.full((72, 96), 64, np.uint8)
    block = QuasiDenseDisparity()
    result = block.execute({"left": blank, "right": blank})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "quasi_dense_no_correspondences"
    assert result.outputs["report"]["native_sparse_candidates"] == 0
    assert not result.outputs["valid_mask"].any()

    def unexpected(*_):
        pytest.fail("Empty support must bypass the native matcher")

    monkeypatch.setattr("opencv_dataflow.quasi_dense_stereo._native_pairs", unexpected)
    result = block.execute({"left": blank, "right": blank, "right_mask": np.zeros_like(blank)})
    assert (
        result and not result.passed and result.fail_reasons[0]["code"] == "quasi_dense_no_support"
    )
    assert np.isnan(result.outputs["disparity"]).all()


def test_native_guards_and_mismatched_input_provenance(monkeypatch):
    def unexpected(*_):
        pytest.fail("Invalid inputs or unsafe settings must stop before matching")

    monkeypatch.setattr("opencv_dataflow.quasi_dense_stereo._native_pairs", unexpected)
    image = np.full((72, 96), 64, np.uint8)
    block = QuasiDenseDisparity(parameters={"border_x": 8})
    assert block.execute({"left": image, "right": image}) is None
    assert "border must be at least" in block.error
    block.parameters["border_x"] = 15
    clock = TicketClock()
    a = ValuePacket(image, {"ticket": clock.issue("left")})
    b = ValuePacket(image, {"ticket": clock.issue("right")})
    assert block.execute({"left": a, "right": b}) is None
    assert "same ticket" in block.error
    left = ValuePacket(image, {"spatial": frame([96, 72], "left")})
    mask = ValuePacket(image, {"spatial": frame([96, 72], "unrelated")})
    assert block.execute({"left": left, "right": image, "left_mask": mask}) is None
    assert "coordinate frame" in block.error


def test_geometric_and_reciprocal_checks_filter_native_pairs(monkeypatch):
    forward = np.array(
        [
            [30, 25, 24, 25],  # Correct reciprocal shift.
            [33, 25, 27, 25],  # Reverse map points to a different pixel.
            [36, 25, 30, 27],  # Vertical mismatch.
            [39, 25, 27, 25],  # Outside the signed interval.
        ],
        np.int32,
    )
    reverse = np.array([[24, 25, 30, 25], [27, 25, 35, 25]], np.int32)
    calls = iter((forward, reverse))
    monkeypatch.setattr(
        "opencv_dataflow.quasi_dense_stereo._native_pairs", lambda *_: (next(calls), 2)
    )
    image = np.full((72, 96), 12, np.uint8)
    block = QuasiDenseDisparity(parameters={"min_disparity": 6, "max_disparity": 8})
    result = block.execute({"left": image, "right": image})
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["rejected_by_support_or_geometry"] == 2
    assert report["rejected_by_left_right"] == 1
    assert report["matched_pixels"] == 1
    assert result.outputs["disparity"][25, 30] == 6
    assert np.isnan(result.outputs["disparity"][25, [33, 36, 39]]).all()
