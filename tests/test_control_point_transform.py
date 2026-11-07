"""Validate exact native transforms and their use by the existing warp blocks."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def test_affine_control_points_keep_subpixel_precision_at_large_coordinates():
    source = np.array([[0, 0], [18, 1], [3, 23]], np.float64) + [7_000_000, -4_000_000]
    target = source + [0.125, -0.375]
    block = builtin_registry().create("cv.control_point_transform")
    result = block.execute({"source_points": source.tolist(), "target_points": target.tolist()})
    assert result is not None, block.error
    matrix = result.outputs["transform"]["source_to_target"]
    np.testing.assert_allclose(matrix, [[1, 0, .125], [0, 1, -.375], [0, 0, 1]], atol=1e-8)
    assert result.outputs["rms_pixels"] < 1e-8


@pytest.mark.parametrize("solver", ["lu", "svd"])
def test_perspective_control_points_wire_into_warp_and_point_transform(solver):
    registry = builtin_registry()
    graph = Graph()
    size = 75
    source = [[8, 9], [54, 5], [62, 55], [3, 64]]
    target = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    fit = graph.add(registry.create("cv.control_point_transform", parameters={
        "model": "perspective", "source_points": source, "target_points": target, "solver": solver,
    }))
    warp = graph.add(registry.create("cv.warp_perspective", parameters={
        "width": size, "height": size,
    }))
    points = graph.add(registry.create("cv.transform_points", parameters={"pixel_points": source}))
    graph.connect(fit.id, "transform", warp.id, "transform")
    graph.connect(fit.id, "transform", points.id, "transform")
    image = np.arange(68 * 69, dtype=np.uint8).reshape(68, 69)
    graph.run(injected={(warp.id, "image"): image})
    assert all(block.state == "pass" for block in graph.blocks.values())
    assert fit.last_result.outputs["rms_pixels"] < 1e-4
    np.testing.assert_allclose(points.last_result.outputs["points"], target, atol=1e-4)
    expected = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(np.float32(source), np.float32(target)), (size, size)
    )
    error = np.abs(warp.last_output_images["image"].astype(np.int16) - expected)
    assert error.mean() < 0.05
    graph.close()


@pytest.mark.parametrize("target", [
    [[0, 0], [1, 1], [2, 2], [0, 3]],
    [[0, 0], [2, 2], [2, 0], [0, 2]],
    [[0, 0], [2, 0], [2, 2], [0, float("nan")]],
])
def test_perspective_rejects_collinear_crossed_and_nonfinite_control_points(target):
    block = builtin_registry().create("cv.control_point_transform", parameters={
        "model": "perspective", "source_points": [[0, 0], [2, 0], [2, 2], [0, 2]],
    })
    assert block.execute({"target_points": target}) is None
    assert block.state == "error"
    assert not block.current_emissions
