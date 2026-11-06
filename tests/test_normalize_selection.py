"""Verify color normalization modes, selections, and legacy defaults."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


def test_joint_and_independent_masked_color_normalization():
    image = np.array(
        [[[10, 100, 200], [30, 120, 240], [90, 2, 20]]],
        dtype=np.uint8,
    )
    mask = np.array([[255, 255, 0]], np.uint8)
    source = ImageArray(image, "rgb")
    spatial = frame([3, 1], "camera")
    block = builtin_registry().create("cv.normalize", parameters={"outside_selection": "preserve"})
    result = block.execute({"image": ValuePacket(source, {"spatial": spatial}), "mask": mask})
    assert result is not None, block.error
    joint = cv2.normalize(image[mask != 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    np.testing.assert_array_equal(result.outputs["image"][0, :2], joint)
    assert image_format(result.outputs["image"]) == "rgb"
    assert result.payloads["image"]["spatial"] == spatial

    block.parameters["channel_mode"] = "per_channel"
    result = block.execute({"image": source, "mask": mask})
    assert result is not None, block.error
    np.testing.assert_array_equal(
        result.outputs["image"], [[[0, 0, 0], [255, 255, 255], [90, 2, 20]]]
    )


def test_normalize_alpha_retains_opacity_and_selected_depth():
    rgba = ImageArray(np.array([[[10, 20, 30, 64], [30, 40, 50, 128]]], np.uint8), "rgba")
    block = builtin_registry().create(
        "cv.normalize",
        parameters={
            "pixel_depth": "float64",
            "maximum": 1,
            "preserve_alpha": True,
        },
    )
    result = block.execute({"image": rgba})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert output.dtype == np.float64 and image_format(output) == "rgba"
    np.testing.assert_allclose(output[..., 3], np.array([[64, 128]]) / 255)
    np.testing.assert_allclose(output[..., :3], (np.asarray(rgba)[..., :3] - 10) / 40)

    block.parameters.update(pixel_depth="same", outside_selection="zero")
    result = block.execute({"image": rgba, "mask": np.zeros((1, 2), np.uint8)})
    assert result is not None, block.error
    assert result.outputs["image"].dtype == np.uint8
    assert not np.any(result.outputs["image"])


def test_legacy_normalize_keeps_zero_background_and_joint_scaling():
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create("cv.normalize"))
    saved = graph.to_dict()
    for key in ("outside_selection", "channel_mode", "preserve_alpha"):
        del saved["blocks"][0]["parameters"][key]
    block = Graph.from_dict(saved, registry).blocks[block.id]
    image = np.array([[50, 100, 150, 200]], np.uint16)
    mask = np.array([[255, 255, 255, 0]], np.uint8)
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    expected = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=mask)
    np.testing.assert_array_equal(result.outputs["image"], expected)
