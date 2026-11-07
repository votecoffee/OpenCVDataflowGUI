"""Validate fixed lens profiles on independent grid observations."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED

_PROJECT = (
    Path(__file__).parents[1]
    / "examples/custom tools/validate_lens_calibration/validate_lens_calibration.json"
)


def _inputs():
    assets = _PROJECT.parent / "assets"
    image = cv2.imdecode(
        np.frombuffer((assets / "held_out_grid.png").read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    return {
        "image": image,
        "calibration": json.loads((assets / "fitted_profile.json").read_bytes()),
    }


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_validation_example_measures_and_draws_held_out_residuals(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    result = graph.blocks["validate"].last_result
    report = result.outputs["report"]
    points = report["points"]
    assert report["pose_direction"] == "board_to_camera"
    assert report["translation_units"] == "mm"
    assert len(points) == 24 and points[-1]["grid"] == [3, 5]
    errors = [
        np.linalg.norm(np.asarray(point["projected"]) - point["observed"]) for point in points
    ]
    assert errors == pytest.approx([point["error_pixels"] for point in points])
    assert result.outputs["rms"] == pytest.approx(np.sqrt(np.mean(np.square(errors))))
    assert result.outputs["max_error"] == pytest.approx(max(errors))
    assert result.outputs["rms"] < 0.1
    assert report["worst_point_index"] == int(np.argmax(errors))
    assert [len(layer.objects) for layer in result.outputs["result_graphics"].layers] == [24, 24]
    output = graph.blocks["overlay"].last_result.outputs["image"]
    assert output.shape == (420, 540, 3) and np.any(output[..., 0] != output[..., 1])


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("rms_limit", "point_limit", "code"),
    [
        (0.3, 0, "calibration_validation_rms_exceeds_limit"),
        (1.0, 0.6, "calibration_validation_point_exceeds_limit"),
    ],
)
def test_altered_profile_fails_selected_limit_and_remains_fixed(rms_limit, point_limit, code):
    tool = Graph.load(_PROJECT, builtin_registry()).blocks["validate"]
    tool.parameters.update(max_rms_pixels=rms_limit, max_point_error_pixels=point_limit)
    inputs = _inputs()
    inputs["calibration"]["distortion"] = [0.7, 0.0, 0.01, 0.0, 0.0]
    before = copy.deepcopy(inputs["calibration"])
    result = tool.execute(inputs)
    assert result is not None and not result.passed, tool.error
    assert [reason["code"] for reason in result.fail_reasons] == [code]
    assert inputs["calibration"] == before
    assert result.outputs["rms"] > 0.4 and result.outputs["max_error"] > 1.0
    if point_limit:
        markers = result.drawings["Observed grid"].objects
        for point, marker in zip(result.outputs["report"]["points"], markers):
            assert marker.color == (REJECTED if point["error_pixels"] > point_limit else ACCEPTED)


@pytest.mark.local_examples
def test_missing_target_reports_failure_without_emitting_numeric_errors():
    tool = Graph.load(_PROJECT, builtin_registry()).blocks["validate"]
    inputs = _inputs()
    inputs["image"] = np.full_like(inputs["image"], 255)
    result = tool.execute(inputs)
    assert result is not None and not result.passed, tool.error
    assert result.fail_reasons[0]["code"] == "calibration_validation_grid_not_found"
    assert result.outputs["rms"] is result.outputs["max_error"] is NO_VALUE
    assert not result.drawings


@pytest.mark.local_examples
def test_validation_rejects_mismatched_image_size():
    tool = Graph.load(_PROJECT, builtin_registry()).blocks["validate"]
    inputs = _inputs()
    inputs["image"] = inputs["image"][:-1]
    assert tool.execute(inputs) is None
    assert "dimensions must match" in tool.error
