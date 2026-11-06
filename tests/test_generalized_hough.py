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


def test_guil_template_recovers_rotation_scale_and_oriented_graphics(repeated_shape):
    _, template = repeated_shape
    scene = np.zeros((135, 180), np.uint8)
    scene[13:49, 13:57] = template
    transform = cv2.getRotationMatrix2D((22, 18), -90, 1.25)
    transform[:, 2] += [117 - 22, 81 - 18]
    scene = np.maximum(scene, cv2.warpAffine(template, transform, (180, 135)))
    mask = np.zeros_like(scene)
    mask[68:94, 103:132] = 255
    block = builtin_registry().create(
        "inspect.guil_template",
        parameters={
            "dp": 1,
            "min_angle": 80,
            "max_angle": 101,
            "angle_step": 5,
            "min_scale": 1.15,
            "max_scale": 1.29,
            "scale_step": 0.05,
            "angle_epsilon": 3,
            "min_angle_votes": 200,
            "min_scale_votes": 25,
            "minimum_votes": 25,
            "max_buffer_size": 32,
            "maximum_matches": 1,
        },
    )
    result = block.execute({"image": scene, "template": template, "mask": mask})
    assert result is not None, block.error
    assert result.passed and result.outputs["count"] == 1
    match = result.outputs["matches"][0]
    np.testing.assert_allclose(match["center"], [117, 81], atol=2, rtol=0)
    assert match["angle_degrees"] == pytest.approx(-90)
    assert match["scale"] == pytest.approx(1.25)
    assert match["votes"] >= 25 and match["scale_votes"] >= 25 and match["angle_votes"] >= 200
    pose = np.array(match["pose"]["source_to_target"])
    np.testing.assert_allclose(pose[:2, :2], [[0, -1.25], [1.25, 0]], atol=1e-12)
    local = np.array([[-22, -18, 1], [21, -18, 1], [21, 17, 1], [-22, 17, 1]])
    np.testing.assert_allclose(match["corners"], (local @ pose.T)[:, :2])
    assert result.drawings["Hough matches"].objects[0].points == tuple(map(tuple, match["corners"]))


def test_guil_template_rejects_invalid_ranges_and_overlarge_search(repeated_shape):
    image, template = repeated_shape
    block = builtin_registry().create(
        "inspect.guil_template", parameters={"min_scale": 2, "max_scale": 1}
    )
    assert block.execute({"image": image, "template": template}) is None
    assert "Minimum scale must be below maximum scale" in block.error
    block.parameters.update(min_scale=0.05, max_scale=8, scale_step=0.01, angle_step=0.5)
    assert block.execute({"image": image, "template": template}) is None
    assert "Guil search is too large" in block.error


def test_guil_template_can_find_a_reduced_part_in_a_smaller_scene(repeated_shape):
    _, template = repeated_shape
    transform = cv2.getRotationMatrix2D((22, 18), -90, 0.5)
    transform[:, 2] += [17 - 22, 16 - 18]
    scene = cv2.warpAffine(template, transform, (34, 32))
    block = builtin_registry().create(
        "inspect.guil_template",
        parameters={
            "dp": 1,
            "min_angle": 80,
            "max_angle": 101,
            "angle_step": 5,
            "min_scale": 0.45,
            "max_scale": 0.54,
            "scale_step": 0.05,
            "min_angle_votes": 20,
            "min_scale_votes": 5,
            "minimum_votes": 3,
            "angle_epsilon": 5,
            "maximum_matches": 1,
        },
    )
    result = block.execute({"image": scene, "template": template})
    assert result is not None, block.error
    match = result.outputs["matches"][0]
    np.testing.assert_allclose(match["center"], [17, 16], atol=2, rtol=0)
    assert match["scale"] == pytest.approx(0.5)
    assert match["angle_degrees"] == pytest.approx(-90)
