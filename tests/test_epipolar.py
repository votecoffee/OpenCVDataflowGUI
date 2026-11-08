"""Test bidirectional epipolar distances and image-space diagnostics."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED

_F = [[0, 0, 0], [0, 0, 1], [0, -1, 0]]


def test_check_epipolar_reports_point_distances_and_layers_in_each_view():
    registry = builtin_registry()
    block = registry.create("3d.check_epipolar", parameters={"minimum_inlier_ratio": 0.6})
    block.exposed_outputs.append(RESULT_GRAPHICS)
    image = np.zeros((90, 130), np.uint8)
    left = {"spatial": frame([130, 90], "left")}
    right = {"spatial": frame([130, 90], "right")}
    result = block.execute(
        {
            "points_left": ValuePacket([[70, 20], [80, 40], [90, 70]], left),
            "points_right": ValuePacket([[60, 20], [70, 41], [80, 78]], right),
            "matrix": _F,
            "left_image": ValuePacket(image, left),
            "right_image": ValuePacket(image, right),
        }
    )
    assert result is not None and result.passed, block.error
    assert result.outputs["inliers"] == [True, True, False]
    assert result.outputs["inlier_ratio"] == pytest.approx(2 / 3)
    report = result.outputs["report"]
    assert report["distance_left_pixels"] == report["distance_right_pixels"] == [0, 1, 8]
    np.testing.assert_allclose(result.outputs["lines_right"][1], [0, 1, -40])
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert [layer.payload["spatial"]["root_id"] for layer in layers] == ["left", "right"]
    assert {obj.color for obj in layers[0].objects} == {ACCEPTED, REJECTED}
    for side, context in (("left", left), ("right", right)):
        assert result.payloads[f"{side}_overlay"]["spatial"] == context["spatial"]
        assert np.any(result.outputs[f"{side}_overlay"] != 0)


def test_epipolar_rejects_mismatched_geometry_and_bad_matrix():
    block = builtin_registry().create("3d.check_epipolar")
    left = {"spatial": frame([130, 90], "left")}
    inputs = {
        "points_left": ValuePacket([[70, 20]], left),
        "points_right": [[60, 20]],
        "matrix": ValuePacket(_F, {"epipolar_frames": {"left": frame([130, 90], "old-left")}}),
    }
    assert block.execute(inputs) is None
    assert "different frames" in block.error
    inputs["matrix"] = np.eye(3).tolist()
    assert block.execute(inputs) is None
    assert "rank two" in block.error


def test_epipolar_reports_undefined_lines_at_an_epipole():
    block = builtin_registry().create("3d.check_epipolar")
    f = [[0, -1, 12], [1, 0, -24], [-12, 24, 0]]
    result = block.execute({"points_left": [[24, 12]], "points_right": [[30, 30]], "matrix": f})
    assert result is not None and not result.passed, block.error
    assert result.outputs["inliers"] == [False]
    assert result.outputs["lines_right"] == [None]
    assert result.outputs["report"]["distance_right_pixels"] == [None]
    assert result.outputs["report"]["undefined"] == 1
    assert result.outputs["left_overlay"] is NO_VALUE
