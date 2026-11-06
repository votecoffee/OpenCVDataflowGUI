"""Exercise Hough detections, pose provenance, and selected candidate ranking."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


@pytest.fixture
def repeated_shape():
    template = np.zeros((36, 44), np.uint8)
    polygon = np.array([[5, 6], [32, 6], [37, 14], [31, 15], [31, 28], [19, 28], [19, 22], [5, 22]])
    cv2.fillPoly(template, [polygon.astype(np.int32)], 230)
    cv2.circle(template, (13, 13), 4, 0, -1)
    image = np.zeros((120, 170), np.uint8)
    image[13:49, 18:62] = template
    image[61:97, 94:138] = template
    return image, template


def test_hough_template_reports_ranked_anchors_and_pose_graphics(repeated_shape):
    image, template = repeated_shape
    spatial = frame([170, 120], "camera")
    block = builtin_registry().create("inspect.hough_template", parameters={"minimum_votes": 122})
    result = block.execute(
        {"image": ValuePacket(image, {"spatial": spatial}), "template": template}
    )
    assert result is not None, block.error
    assert result.passed and result.outputs["count"] == 2
    matches = result.outputs["matches"]
    assert [m["center"] for m in matches] == [[40, 31], [116, 79]]
    assert [m["votes"] for m in matches] == [122, 122]
    assert [m["bounds"] for m in matches] == [[18, 13, 44, 36], [94, 61, 44, 36]]
    assert result.outputs["poses"][0]["target_payload"]["spatial"] == spatial
    assert len(result.drawings["Hough matches"].objects) == 6
    assert result.outputs["overlay"].shape == (120, 170, 3)
    assert np.any(result.outputs["overlay"][13, 18] != 0)


def test_roi_filters_anchors_before_minimum_distance_suppression(repeated_shape):
    image, template = repeated_shape
    registry = builtin_registry()
    region = registry.create("region.shape", parameters={"rectangle": [97, 65, 7, 7]})
    selection = region.execute({"image": image})
    assert selection is not None, region.error
    block = registry.create(
        "inspect.hough_template",
        parameters={"template_anchor": [6, 7], "min_center_distance": 100, "draw_overlay": False},
    )
    result = block.execute(
        {"image": image, "template": template, "region": selection.outputs["region"]}
    )
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert result.outputs["matches"][0]["center"] == [100, 68]
    assert result.outputs["matches"][0]["bounds"] == [94, 61, 44, 36]
    assert result.outputs["overlay"] is NO_VALUE


def test_hough_template_handles_no_matches_and_invalid_templates(repeated_shape):
    image, template = repeated_shape
    block = builtin_registry().create("inspect.hough_template")
    result = block.execute({"image": np.zeros_like(image), "template": template})
    assert result is not None, block.error
    assert not result.passed and result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "hough_match_count_below_minimum"
    assert block.execute({"image": image, "template": np.zeros_like(template)}) is None
    assert "no edges" in block.error
    block.parameters["template_anchor"] = [100, 100]
    assert block.execute({"image": image, "template": template}) is None
    assert "anchor must be inside" in block.error
