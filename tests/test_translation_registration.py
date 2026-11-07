"""Test translation-only consensus, diagnostics, and saved graph wiring."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED

TYPE_ID = "inspect.translation"


def _pairs(origin=0):
    query = origin + np.array(
        [[8, 11], [15, 20], [28, 12], [40, 40], [30, 35], [21, 27], [18, 30], [24, 41]],
        dtype=np.float64,
    )
    deltas = np.array(
        [
            [3.125, -4.25],
            [3.09375, -4.25],
            [3.15625, -4.25],
            [3.125, -4.21875],
            [3.125, -4.28125],
            [3.125, -4.25],
            [-16, 40],
            [36, 17],
        ]
    )
    return {"query_points": query.tolist(), "train_points": (query + deltas).tolist()}, deltas


@pytest.mark.parametrize("method", ("ransac", "lmeds"))
def test_robust_fit_recovers_subpixel_motion_at_large_image_origin(method):
    data, deltas = _pairs(origin=1e9)
    block = builtin_registry().create(
        TYPE_ID, parameters={"method": method, "reprojection_pixels": 0.2}
    )
    result = block.execute({"matches": data})
    assert result and result.passed, block.error
    assert result.outputs["inliers"] == 6
    assert result.score == 0.75
    fit = result.outputs["fit"]
    np.testing.assert_allclose(fit["shift_pixels"], [3.125, -4.25], atol=1e-8)
    assert fit["inlier_mask"] == [True] * 6 + [False] * 2
    residuals = np.linalg.norm(deltas - fit["shift_pixels"], axis=1)
    np.testing.assert_allclose(fit["residuals_pixels"], residuals)
    assert fit["rms_pixels"] == pytest.approx(np.sqrt(np.mean(residuals[:6] ** 2)))
    matrix = result.outputs["transform"]["source_to_target"]
    np.testing.assert_allclose(matrix, [[1, 0, 3.125], [0, 1, -4.25], [0, 0, 1]])


def test_acceptance_failure_keeps_diagnostics_and_colored_match_markers():
    data, _ = _pairs()
    data["train_size"] = [96, 96]
    block = builtin_registry().create(
        TYPE_ID, parameters={"min_inliers": 7, "min_inlier_ratio": 0.9}
    )
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"matches": data, "train_image": np.zeros((96, 96), np.uint8)})
    assert result and not result.passed, block.error
    assert result.outputs["transform"] is NO_VALUE
    assert result.outputs["inliers"] == 6
    assert result.outputs["fit"]["status"] == "below_acceptance"
    assert {reason["code"] for reason in result.fail_reasons} == {
        "translation_below_min_inliers",
        "translation_below_min_inlier_ratio",
    }
    markers = result.outputs["result_graphics"].layers[0].objects
    assert [marker.color for marker in markers] == [ACCEPTED] * 6 + [REJECTED] * 2
    assert [marker.points[0] for marker in markers] == list(map(tuple, data["train_points"]))


def test_missing_correspondences_and_native_failure_withhold_transform(monkeypatch):
    block = builtin_registry().create(TYPE_ID)
    empty = block.execute({"matches": {"query_points": [], "train_points": []}})
    assert empty and not empty.passed
    assert empty.outputs["transform"] is NO_VALUE
    assert empty.outputs["fit"]["status"] == "translation_no_matches"
    monkeypatch.setattr(
        cv2, "estimateTranslation2D", lambda *args, **kwargs: ((float("nan"), float("nan")), None)
    )
    data, _ = _pairs()
    unavailable = block.execute({"matches": data})
    assert unavailable and not unavailable.passed
    assert unavailable.outputs["transform"] is NO_VALUE
    assert unavailable.outputs["fit"]["status"] == "translation_no_consensus"


def test_bad_pair_lengths_and_wrong_preview_dimensions_are_validated():
    block = builtin_registry().create(TYPE_ID)
    data, _ = _pairs()
    assert block.execute({"matches": {**data, "train_points": data["train_points"][:-1]}}) is None
    assert "equal lists of paired" in block.error
    assert (
        block.execute(
            {
                "matches": {**data, "train_size": [96, 96]},
                "train_image": np.zeros((96, 95), np.uint8),
            }
        )
        is None
    )
    assert "match the train image" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_graph_recovers_and_applies_translation_from_sift_matches(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools/translation_registration/translation_registration.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    fit = graph.blocks["estimate"].last_result.outputs["fit"]
    assert fit["shift_pixels"] == pytest.approx([9, 13], abs=0.05)
    assert fit["rms_pixels"] < 0.2
    assert fit["inlier_ratio"] > 0.9
    expected = graph.blocks["shift"].last_result.outputs["image"].astype(np.int16)
    actual = graph.blocks["warp"].last_result.outputs["image"]
    assert np.mean(np.abs(expected - actual)) < 0.5


@pytest.mark.parametrize("warp_type", ("cv.warp_affine", "cv.warp_perspective"))
def test_rejected_fit_stops_connected_warp_and_rerun_recovers(warp_type):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools/translation_registration/translation_registration.json"
    )
    registry = builtin_registry()
    data = Graph.load(project, registry).to_dict()
    warp = next(block for block in data["blocks"] if block["id"] == "warp")
    warp.update(type=warp_type, parameters={})
    graph = Graph.from_dict(data, registry, project_dir=project.parent)
    graph.run()
    assert graph.blocks["warp"].last_result and graph.blocks["warp"].state == "pass"
    graph.blocks["estimate"].parameters["min_inliers"] = 100_000
    graph.run()
    assert graph.blocks["estimate"].state == "fail"
    assert graph.blocks["warp"].state == "inactive"
    assert graph.blocks["warp"].last_result is None
    assert not graph.blocks["warp"].last_output_images
    graph.blocks["estimate"].parameters["min_inliers"] = 8
    graph.run()
    assert graph.blocks["warp"].state == "pass"
    assert graph.blocks["warp"].last_result is not None
