"""Native blob detection and cloning through the normal block contracts."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.payloads import frame, image_with_payload
from opencv_dataflow.restoration import SeamlessClone


def test_simple_blob_detector_filters_by_shape_and_mask():
    image = np.full((120, 180), 226, np.uint8)
    cv2.circle(image, (36, 35), 12, 25, -1)
    cv2.circle(image, (126, 82), 18, 25, -1)
    cv2.rectangle(image, (61, 35), (111, 45), 25, -1)
    mask = np.zeros_like(image)
    mask[:, :90] = 255
    block = builtin_registry().create(
        "inspect.simple_blob_detector",
        parameters={"filter_circularity": True, "min_circularity": 0.65, "minimum_count": 2},
    )
    block.execute({"image": image})
    assert block.state == "pass"
    outputs = block.last_result.outputs
    assert sorted(outputs["points"]) == [[36.0, 35.0], [126.0, 82.0]]
    circles = validate_collection(outputs["collection"])
    assert circles["kind"] == "circles"
    assert len(block.last_result.drawings["blob keypoints"].objects) == 2

    block.execute({"image": image, "mask": mask})
    assert block.state == "fail"
    assert block.last_result.outputs["points"] == [[36.0, 35.0]]
    assert [reason["code"] for reason in block.last_result.fail_reasons] == [
        "simple_blob_count_below_minimum"
    ]


@pytest.fixture
def cloning_images():
    rows, cols = np.indices((43, 61))
    source = np.stack((cols * 4, rows * 4, (rows * cols) % 256), axis=-1).astype(np.uint8)
    rows, cols = np.indices((100, 140))
    destination = np.stack((40 + cols, 70 + rows, 180 - cols), axis=-1).astype(np.uint8)
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[5:34, :30] = 255
    return source, destination, mask


@pytest.mark.parametrize("mode", tuple(SeamlessClone.modes))
def test_seamless_clone_matches_native_and_preserves_destination_frame(mode, cloning_images):
    source, destination, mask = cloning_images
    tagged = cv2.cvtColor(destination, cv2.COLOR_BGR2RGBA)
    tagged[..., 3] = np.arange(destination.shape[1], dtype=np.uint8)
    spatial = frame([destination.shape[1], destination.shape[0]], "destination")
    target = image_with_payload(ImageArray(tagged, "rgba"), {"spatial": spatial})
    block = builtin_registry().create("cv.seamless_clone", parameters={"mode": mode})
    block.execute({"source": source, "destination": target, "mask": mask})
    assert block.state == "pass", block.error
    output = block.last_result.outputs["image"]
    native = cv2.seamlessClone(
        np.pad(source, ((1, 1), (1, 1), (0, 0))),
        destination,
        np.pad(mask, 1),
        (70, 50),
        SeamlessClone.modes[mode],
    )
    assert np.any(native != destination)
    assert output.color_format == "rgba"
    np.testing.assert_array_equal(output[..., :3], native[..., ::-1])
    np.testing.assert_array_equal(output[..., 3], tagged[..., 3])
    assert block.last_result.payloads["image"]["spatial"] == spatial


def test_seamless_clone_empty_selection_and_bounds(cloning_images):
    source, destination, mask = cloning_images
    block = builtin_registry().create("cv.seamless_clone", parameters={"center_x": 0})
    block.execute({"source": source, "destination": destination, "mask": mask})
    assert block.state == "error"
    assert block.error_kind == "invalid"
    assert "must fit" in block.error

    block.execute({"source": source, "destination": destination, "mask": np.zeros_like(mask)})
    assert block.state == "pass"
    np.testing.assert_array_equal(block.last_result.outputs["image"], destination)
