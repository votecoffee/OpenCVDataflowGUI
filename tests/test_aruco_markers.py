"""Verify marker selection, camera-space graphics, and ordered correspondences."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


def _panel(right_id=7):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((200, 340), 255, np.uint8)
    for number, x in ((28, 35), (right_id, 220)):
        image[42:114, x : x + 72] = cv2.aruco.generateImageMarker(dictionary, number, 72)
    return image


def test_aruco_points_follow_id_order_and_retain_camera_frame():
    image = _panel()
    spatial = frame(image.shape[::-1], "camera-view")
    detector = builtin_registry().create(
        "inspect.aruco_markers", parameters={"required_ids": [28, 7]}
    )
    detector.exposed_outputs.append(RESULT_GRAPHICS)
    result = detector.execute({"image": ValuePacket(image, {"spatial": spatial})})
    assert result is not None and result.passed, detector.error
    assert [marker["id"] for marker in result.outputs["markers"]] == [7, 28]
    assert result.outputs["count"] == 2
    np.testing.assert_allclose(result.outputs["points"], [[70.5, 77.5], [255.5, 77.5]], atol=0.6)
    assert result.payloads["points"]["spatial"] == spatial
    (layer,) = result.outputs[RESULT_GRAPHICS].layers
    assert layer.payload["spatial"] == spatial
    assert {obj.text for obj in layer.objects if obj.kind == "label"} == {"7", "28"}

    detector.parameters.update(point="corner_2", draw_overlay=False)
    result = detector.execute({"image": image})
    assert result is not None and result.passed, detector.error
    np.testing.assert_allclose(result.outputs["points"], [[106, 113], [291, 113]], atol=0.6)
    assert result.outputs["overlay"] is NO_VALUE


def test_aruco_center_follows_perspective_projection():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    patch = cv2.aruco.generateImageMarker(dictionary, 16, 120)
    source = np.float32([[0, 0], [119, 0], [119, 119], [0, 119]])
    target = np.float32([[49, 47], [223, 24], [196, 206], [85, 184]])
    transform = cv2.getPerspectiveTransform(source, target)
    image = cv2.warpPerspective(patch, transform, (270, 240), borderValue=255)
    detector = builtin_registry().create("inspect.aruco_markers")
    result = detector.execute({"image": image})
    assert result is not None and result.passed, detector.error
    expected = cv2.perspectiveTransform(np.float32([[[59.5, 59.5]]]), transform).reshape(1, 2)
    np.testing.assert_allclose(result.outputs["points"], expected, atol=1.0)


def test_aruco_rejects_missing_and_ambiguous_correspondences():
    detector = builtin_registry().create(
        "inspect.aruco_markers", parameters={"required_ids": [28, 7]}
    )
    image = _panel()
    mask = np.zeros_like(image)
    mask[:, :165] = 255
    result = detector.execute({"image": image, "mask": mask})
    assert result is not None and not result.passed, detector.error
    assert result.outputs["count"] == 1
    assert result.outputs["points"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "aruco_missing_ids"

    detector.parameters["required_ids"] = [28]
    result = detector.execute({"image": _panel(right_id=28)})
    assert result is not None and not result.passed, detector.error
    assert result.outputs["count"] == 2
    assert result.outputs["points"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "aruco_duplicate_ids"


@pytest.mark.parametrize("ids", ([28, 28], [50], [True]))
def test_aruco_validates_required_ids_against_dictionary(ids):
    block = builtin_registry().create("inspect.aruco_markers")
    with pytest.raises(ValidationError) as error:
        block.validate_parameters({**block.parameters, "required_ids": ids})
    assert error.value.fields == ("required_ids",)
