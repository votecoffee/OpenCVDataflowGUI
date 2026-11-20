"""Verify pixel-only stereo rectification, valid support and frame provenance."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame

TYPE_ID = "3d.rectify_uncalibrated_stereo"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/uncalibrated_stereo"
F_RECTIFIED = [[0, 0, 0], [0, 0, 1], [0, -1, 0]]


def project(h, xy):
    xy = np.asarray(xy, dtype=float)
    mapped = np.c_[xy, np.ones(len(xy))] @ np.asarray(h).T
    return mapped[:, :2] / mapped[:, 2, None]


def rms_dy(left, right):
    return float(np.sqrt(np.mean((np.asarray(left)[:, 1] - np.asarray(right)[:, 1]) ** 2)))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_aligns_independent_points_and_supports_both_views(profile):
    graph = Graph.load(EXAMPLE / "uncalibrated_stereo.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["rectify"]
        result = block.last_result
        assert result and result.passed, (block.error, result)
        out = result.outputs
        truth = json.loads((EXAMPLE / "assets/truth.json").read_bytes())
        assert sorted(set(range(56)) - set(out["indices"])) == truth["outlier_indices"]
        report = out["report"]
        assert report["fit_pairs"] == 52 and report["vertical_rms_px"] < 0.25
        assert report["input_vertical_rms_px"] > 20

        data = out["rectification"]
        assert data["model"] == "uncalibrated_stereo_rectification_v1"
        assert data["left_size"] == [760, 550] and data["right_size"] == [700, 510]
        assert data["image_size"] == [640, 480] and "Q" not in data
        a, b = truth["heldout_pixels"]
        left, right = project(data["H1"], a), project(data["H2"], b)
        assert rms_dy(a, b) > 20 and rms_dy(left, right) < 0.35

        for i, (side, heldout) in enumerate((("left", left), ("right", right)), 1):
            original = graph.blocks[f"{side}_matches"].last_result.outputs["value"]
            np.testing.assert_allclose(
                out[f"points_{side}"], project(data[f"H{i}"], np.asarray(original)[out["indices"]])
            )
            image = out[side]
            assert image.shape == (480, 640, 3) and image.dtype == np.uint8
            valid = out[f"{side}_valid"]
            assert valid.shape == (480, 640) and set(np.unique(valid)) == {0, 255}
            x, y = np.rint(heldout).astype(int).T
            assert np.all(valid[y, x] == 255)
            # The held-out marker centers land on drawn features in the warped raster.
            assert np.count_nonzero(np.max(image[y, x], axis=1) > 120) >= 18
            assert report[f"{side}_support"] == pytest.approx(np.count_nonzero(valid) / valid.size)

        common = out["common_valid"]
        np.testing.assert_array_equal(common, out["left_valid"] & out["right_valid"])
        assert report["common_support"] == pytest.approx(np.count_nonzero(common) / common.size)
        assert 0.85 < report["common_support"] < 0.90
    finally:
        graph.close()


@pytest.fixture
def rectified_inputs():
    a = [[x, y] for y in (8, 13, 18) for x in (8, 14, 21)]
    b = [[x - 2, y] for x, y in a]
    left_context = {
        "spatial": frame([32, 24], "left-raw"),
        "stereo_calibration_grid": {"side": "left"},
    }
    right_context = {"spatial": frame([32, 24], "right-raw")}
    return {
        "points_left": ValuePacket(a, left_context),
        "points_right": ValuePacket(b, right_context),
        "matrix": F_RECTIFIED,
        "left_image": ValuePacket(
            ImageArray(np.zeros((24, 32, 3), np.uint16), "rgb"), left_context
        ),
        "right_image": ValuePacket(np.zeros((24, 32), np.uint8), right_context),
    }


@pytest.mark.parametrize(("interpolation", "right_edge"), [("linear", 2), ("nearest", 1)])
def test_raster_support_excludes_border_and_maps_to_each_source(
    rectified_inputs, monkeypatch, interpolation, right_edge
):
    left = np.array([[1, 0, 0.75], [0, 1, 0], [0, 0, 1.0]])
    right = np.array([[1, 0, -1.25], [0, 1, 0], [0, 0, 1.0]])
    monkeypatch.setattr(
        cv2, "stereoRectifyUncalibrated", lambda *args, **kwargs: (True, left, right)
    )
    block = builtin_registry().create(TYPE_ID, parameters={"interpolation": interpolation})
    result = block.execute(rectified_inputs)
    assert result and result.passed
    out = result.outputs
    assert image_format(out["left"]) == "rgb" and out["left"].dtype == np.uint16
    assert image_format(out["right"]) == "gray" and out["right"].dtype == np.uint8
    assert np.all(out["left_valid"][:, 0] == 0) and np.all(out["left_valid"][:, 1:] == 255)
    assert np.all(out["right_valid"][:, -right_edge:] == 0)
    assert np.all(out["right_valid"][:, :-right_edge] == 255)
    assert out["report"]["common_support"] == pytest.approx((31 - right_edge) / 32)
    assert np.count_nonzero(out["left"]) == np.count_nonzero(out["right"]) == 0
    for side, h in (("left", left), ("right", right)):
        for port in (side, f"{side}_valid", f"points_{side}"):
            payload = result.payloads[port]
            assert payload["spatial"]["root_id"] == f"{side}-raw"
            transform = np.array(payload["spatial"]["to_root"])
            np.testing.assert_allclose(transform / transform[2, 2], np.linalg.inv(h))
            assert "stereo_calibration_grid" not in payload
    assert result.payloads["common_valid"]["spatial"] == result.payloads["left"]["spatial"]


def test_prefilters_by_supplied_mask_and_sampson_error(rectified_inputs, monkeypatch):
    original = rectified_inputs["points_left"].value
    right = rectified_inputs["points_right"].value
    original.extend([[24, 22], [26, 20]])
    right.extend([[21, 13], [23, 20]])
    rectified_inputs["inliers"] = [True] * 10 + [False]

    def solve(a, b, f, size, *, threshold):
        assert len(a) == len(b) == 9 and len(f) == 3
        assert size == (32, 24) and threshold == 0
        return True, np.eye(3), np.eye(3)

    monkeypatch.setattr(cv2, "stereoRectifyUncalibrated", solve)
    result = builtin_registry().create(TYPE_ID).execute(rectified_inputs)
    assert result and result.passed
    assert result.outputs["indices"] == list(range(9))


@pytest.mark.parametrize(
    "failure",
    [
        "insufficient_rectification_points",
        "uncalibrated_stereo_solver_failed",
        "invalid_homography",
        "projective_horizon_intersects_source",
        "rectification_vertical_limit",
        "rectification_common_support",
    ],
)
def test_failed_geometry_withholds_images_and_model(rectified_inputs, monkeypatch, failure):
    left, right = np.eye(3), np.eye(3)
    ok = failure != "uncalibrated_stereo_solver_failed"
    settings = {}
    match failure:
        case "insufficient_rectification_points":
            rectified_inputs["inliers"] = [True] * 7 + [False] * 2
        case "invalid_homography":
            left[0] = 0
        case "projective_horizon_intersects_source":
            left[2] = [1, 0, -16]
        case "rectification_vertical_limit":
            right[1, 2] = 3
        case "rectification_common_support":
            right[0, 2] = 32
            settings["minimum_common_support"] = 0
    monkeypatch.setattr(cv2, "stereoRectifyUncalibrated", lambda *args, **kwargs: (ok, left, right))
    block = builtin_registry().create(TYPE_ID, parameters=settings)
    result = block.execute(rectified_inputs)
    assert result and not result.passed, block.error
    assert result.outputs["report"]["status"] == failure
    assert result.outputs["left"] is result.outputs["right"] is NO_VALUE
    assert result.outputs["common_valid"] is result.outputs["rectification"] is NO_VALUE


def test_rejects_matrix_frame_mismatch_invalid_mask_and_oversized_canvas(rectified_inputs):
    registry = builtin_registry()
    block = registry.create(TYPE_ID)
    bad_matrix = ValuePacket(
        F_RECTIFIED, {"epipolar_frames": {"left": frame([32, 24], "different")}}
    )
    assert block.execute({**rectified_inputs, "matrix": bad_matrix}) is None
    assert "different frames" in block.error
    assert block.execute({**rectified_inputs, "inliers": [1] * 9}) is None
    assert "boolean mask" in block.error
    big = registry.create(TYPE_ID, parameters={"width": 8192, "height": 8192})
    assert big.execute(rectified_inputs) is None
    assert "canvas exceeds" in big.error
