"""Validate exact native transforms and their use by the existing warp blocks."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame, metadata


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

def test_rectify_quadrilateral_auto_size_keeps_both_endpoint_pixels():
    image = np.arange(25 * 35, dtype=np.uint16).reshape(25, 35)
    block = builtin_registry().create("cv.rectify_quadrilateral")
    corners = [[6, 4], [23, 4], [23, 15], [6, 15]]
    context = {"spatial": frame([35, 25], "inspection")}
    result = block.execute({
        "image": ValuePacket(image, context),
        "corners": ValuePacket(corners, context),
    })
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], image[4:16, 6:24])
    geometry = result.outputs["geometry"]
    assert geometry["output_size"] == [18, 12]
    transform = geometry["output_to_input"]
    np.testing.assert_allclose(transform, [[1, 0, 6], [0, 1, 4], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(metadata(result.outputs["image"])["spatial"]["to_root"], transform)


def test_rectify_quadrilateral_uses_longer_edges_or_an_explicit_size():
    image = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
    corners = [[8, 4], [23, 4], [28, 20], [3, 20]]
    block = builtin_registry().create("cv.rectify_quadrilateral")
    result = block.execute({"image": image, "corners": corners})
    assert result is not None, block.error
    assert result.outputs["geometry"]["output_size"] == [26, 18]
    block.parameters.update(width=19, height=13, solver="svd")
    result = block.execute({"image": image, "corners": corners})
    assert result is not None, block.error
    assert result.outputs["image"].shape == (13, 19)
    expected = cv2.perspectiveTransform(
        np.float64([[[0, 0], [18, 0], [18, 12], [0, 12]]]),
        np.asarray(result.outputs["geometry"]["output_to_input"]),
    )
    np.testing.assert_allclose(expected[0], corners, atol=1e-5)


@pytest.mark.parametrize(("corners", "error"), [
    ([[2, 2], [2, 15], [16, 15], [16, 2]], "ordered clockwise"),
    ([[2, 2], [16, 2], [5, 4], [2, 15]], "convex corners"),
    ([[2, 2], [23, 2], [16, 15], [2, 15]], "inside the source image"),
])
def test_rectify_quadrilateral_rejects_unsafe_corner_sets(corners, error):
    block = builtin_registry().create("cv.rectify_quadrilateral")
    assert block.execute({"image": np.zeros((20, 20), np.uint8), "corners": corners}) is None
    assert error in block.error
    assert not block.current_emissions


def test_rectify_quadrilateral_rejects_corners_from_another_frame():
    block = builtin_registry().create("cv.rectify_quadrilateral")
    image = ValuePacket(np.zeros((20, 20), np.uint8), {"spatial": frame([20, 20], "a")})
    corners = ValuePacket([[2, 2], [16, 2], [16, 15], [2, 15]],
                          {"spatial": frame([20, 20], "b")})
    assert block.execute({"image": image, "corners": corners}) is None
    assert "coordinate frame" in block.error
