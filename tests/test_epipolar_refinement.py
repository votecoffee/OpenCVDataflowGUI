"""Check Sampson metrics, true correspondence error and refinement gates."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED

TYPE_ID = "3d.refine_epipolar"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/epipolar_refinement"
F_RECTIFIED = [[0, 0, 0], [0, 0, 1], [0, -1, 0]]


def rms(delta):
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_rejects_wrong_matches_and_improves_true_pixel_error(profile):
    graph = Graph.load(EXAMPLE / "epipolar_refinement.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["refine"]
        result = block.last_result
        assert result and result.passed, block.error
        out = result.outputs
        report = out["report"]
        assert report["accepted_count"] == 29 and out["inlier_ratio"] == 29 / 32
        fixture = json.loads((EXAMPLE / "assets/truth.json").read_bytes())
        assert np.flatnonzero(~np.asarray(out["inliers"])).tolist() == fixture["outlier_indices"]
        assert all(report["status"][j] == "sampson_limit" for j in fixture["outlier_indices"])

        selected = out["indices"]
        truth = np.asarray(fixture["clean"])[:, selected]
        raw = np.asarray(
            [
                graph.blocks[f"points_{side}"].last_result.outputs["value"]
                for side in ("left", "right")
            ]
        )
        corrected = np.asarray([out["points_left"], out["points_right"]])
        assert rms(corrected - truth) < 0.75
        assert rms(corrected - truth) < rms(raw[:, selected] - truth) * 0.9
        assert 0.5 < report["accepted_input_rms_px"] < 0.8
        assert report["accepted_output_rms_px"] < 1e-9
        assert graph.blocks["check"].last_result.outputs["inliers"] == [True] * 29
        assert len(result.drawings) == len(result.graphics) == (2 if profile == "debug" else 0)

        f = np.asarray(graph.blocks["fundamental"].last_result.outputs["value"])
        for a, b, observed in zip(raw[0], raw[1], report["sampson_px"], strict=True):
            native = cv2.sampsonDistance(np.r_[a, 1.0], np.r_[b, 1.0], f)
            assert observed == pytest.approx(np.sqrt(native), abs=1e-11)
    finally:
        graph.close()


def test_rectified_pair_is_corrected_at_analytic_midpoint_and_keeps_both_frames():
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_inlier_ratio": 0.6})
    block.exposed_outputs.append(RESULT_GRAPHICS)
    a = [[70, 20], [80, 40], [90, 70]]
    b = [[60, 20], [70, 41], [80, 78]]
    left, right = {"spatial": frame([130, 90], "left")}, {"spatial": frame([110, 100], "right")}
    inputs = {
        "points_left": ValuePacket(a, left),
        "points_right": ValuePacket(b, right),
        "left_image": ValuePacket(np.zeros((90, 130), np.uint8), left),
        "right_image": ValuePacket(np.zeros((100, 110), np.uint8), right),
        "matrix": ValuePacket(
            (-1e-90 * np.asarray(F_RECTIFIED)).tolist(),
            {"epipolar_frames": {"left": left["spatial"], "right": right["spatial"]}},
        ),
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["indices"] == [0, 1]
    assert result.outputs["inliers"] == [True, True, False]
    np.testing.assert_allclose(result.outputs["points_left"], [[70, 20], [80, 40.5]])
    np.testing.assert_allclose(result.outputs["points_right"], [[60, 20], [70, 40.5]])
    report = result.outputs["report"]
    assert report["left_shift_px"] == report["right_shift_px"] == [0, 0.5, None]
    assert report["output_sampson_px"] == [0, 0, None]
    for side, known in (("left", left), ("right", right)):
        assert result.payloads[f"points_{side}"]["spatial"] == known["spatial"]
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert [layer.payload["spatial"]["root_id"] for layer in layers] == ["left", "right"]
    assert {obj.color for obj in layers[0].objects} == {ACCEPTED, REJECTED}
    block.parameters["mode"] = "evaluate"
    score = block.execute(inputs)
    assert score and score.passed
    assert score.outputs["points_left"] == a[:2]
    assert score.outputs["points_right"] == b[:2]
    assert score.outputs["report"]["output_sampson_px"][1] == pytest.approx(2**-0.5)


def test_filters_exact_epipoles_in_either_view_with_defined_source_indices():
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_inlier_ratio": 0.3})
    matrix = [[0, -1, 12], [1, 0, -24], [-12, 24, 0]]
    result = block.execute(
        {
            "points_left": [[24, 12], [15, 15], [31, 18]],
            "points_right": [[30, 30], [10.5, 16.5], [24, 12]],
            "matrix": matrix,
        }
    )
    assert result and result.passed, block.error
    assert result.outputs["indices"] == [1]
    assert result.outputs["report"]["status"] == [
        "undefined_epiline",
        "accepted",
        "undefined_epiline",
    ]
    assert result.outputs["report"]["sampson_px"] == [None, 0.0, None]


@pytest.mark.parametrize("reason", ["shift_limit", "outside_grid"])
def test_correction_does_not_emit_a_pair_that_shifts_too_far_or_leaves_its_view(reason):
    options = {"maximum_sampson_px": 20, "maximum_shift_px": 0.1}
    y = 10
    if reason == "outside_grid":
        options["maximum_shift_px"] = 20
        y = 20
    block = builtin_registry().create(TYPE_ID, parameters=options)
    result = block.execute(
        {
            "points_left": ValuePacket([[7, 9]], {"spatial": frame([40, 10], "left")}),
            "points_right": ValuePacket([[8, y]], {"spatial": frame([40, 30], "right")}),
            "matrix": F_RECTIFIED,
        }
    )
    assert result and not result.passed
    assert result.outputs["points_left"] is result.outputs["points_right"] is NO_VALUE
    assert result.outputs["report"]["status"] == [reason]
    assert not result.outputs["indices"] and result.outputs["report"]["accepted_count"] == 0


def test_native_nonfinite_or_inaccurate_candidate_is_isolated(monkeypatch):
    def stub(matrix, left, right):
        assert left.shape == right.shape == (1, 3, 2)
        fixed_left, fixed_right = left.copy(), right.copy()
        fixed_left[0, 0, 0] = np.nan
        # Preserve the second pair's residual. Set the third pair exactly onto its line.
        fixed_right[0, 2, 1] = fixed_left[0, 2, 1]
        return fixed_left, fixed_right

    monkeypatch.setattr(cv2, "correctMatches", stub)
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_inlier_ratio": 0.3})
    result = block.execute(
        {
            "points_left": [[10, 10], [11, 11], [12, 12]],
            "points_right": [[12, 10], [14, 12], [16, 13]],
            "matrix": F_RECTIFIED,
        }
    )
    assert result and result.passed
    assert result.outputs["indices"] == [2]
    assert result.outputs["report"]["status"] == [
        "nonfinite_correction",
        "closure_limit",
        "accepted",
    ]


def test_refinement_uses_fundamental_matrix_frames():
    block = builtin_registry().create(TYPE_ID)
    value = block.execute(
        {
            "points_left": ValuePacket([[5, 4]], {"spatial": frame([30, 30], "rectified")}),
            "points_right": [[6, 5]],
            "matrix": ValuePacket(
                F_RECTIFIED, {"epipolar_frames": {"left": frame([30, 30], "raw")}}
            ),
        }
    )
    assert value is None and "different frames" in block.error
