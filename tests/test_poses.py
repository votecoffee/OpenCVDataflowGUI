import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import pose_inspection_graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, packet_for
from opencv_dataflow.poses import local_offset, make_pose
from opencv_dataflow.selection import rasterize


def tool(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state == "pass", block.error
    return block


def pose(image, **parameters):
    return tool("pose.define", {"image": image}, **parameters).last_result.outputs["pose"]


def test_local_offset_composes_after_rotation_scale_and_maps_points():
    image = np.zeros((200, 240), np.uint8)
    original = pose(image, x=80, y=70, angle_degrees=90, scale=2)
    changed = tool(
        "pose.offset", {"pose": original}, pose_offset_x=10, pose_offset_y=5, pose_offset_angle=30
    ).last_result.outputs
    np.testing.assert_allclose(changed["points"], [[90, 50]])
    assert changed["angle_degrees"] == pytest.approx(120)
    np.testing.assert_allclose(original["origin"], [80, 70])
    result = tool(
        "cv.transform_points", {"points": [[0, 0], [1, 0]], "transform": changed["pose"]}
    ).last_result
    matrix = np.asarray(changed["pose"]["source_to_target"])
    np.testing.assert_allclose(result.outputs["points"], [[90, 50], (matrix @ [1, 0, 1])[:2]])


@pytest.mark.parametrize("shape", ["rectangle", "circle", "polygon"])
def test_negative_local_regions_rotate_scale_and_serialize_without_clipping(shape):
    image = np.zeros((160, 200), np.uint8)
    anchor = pose(image, x=80, y=70, angle_degrees=30, scale=1.5)
    parameters = {
        "shape": shape,
        "rectangle": [-12, -9, 24, 18],
        "center": [-3, 2],
        "radius": 8,
        "vertices": [[-12, -8], [9, -8], [3, 11]],
    }
    outputs = tool(
        "region.pose", {"image": image, "pose": anchor}, **parameters
    ).last_result.outputs
    # Independent local canvas at offset (-40,-40) retains negative coordinates.
    local = np.zeros((80, 80), np.uint8)
    if shape == "rectangle":
        cv2.rectangle(local, (28, 31), (51, 48), 255, -1)
    elif shape == "circle":
        cv2.circle(local, (37, 42), 8, 255, -1)
    else:
        cv2.fillPoly(local, [np.array([[28, 32], [49, 32], [43, 51]])], 255)
    matrix = np.asarray(anchor["source_to_target"]) @ local_offset(-40, -40, 0)
    expected = cv2.warpPerspective(local, matrix, (200, 160), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(outputs["mask"], expected)
    np.testing.assert_array_equal(
        rasterize(copy.deepcopy(outputs["region"]), image.shape), expected
    )
    inverted = tool(
        "region.pose", {"image": image, "pose": anchor}, **parameters, invert=True
    ).last_result.outputs["mask"]
    np.testing.assert_array_equal(inverted, 255 - expected)


def test_blob_and_contour_feature_poses_handle_orientation_and_offsets():
    image = np.zeros((100, 140), np.uint8)
    image[20:40, 50:90] = 255
    blobs = tool("inspect.blobs", {"image": image})
    features = blobs.last_result.outputs["collection"]
    outputs = tool(
        "pose.from_feature",
        {"image": image, "features": features},
        angle_mode="feature",
        pose_offset_y=5,
    ).last_result.outputs
    np.testing.assert_allclose(outputs["points"], [[69.5, 34.5]], atol=1e-8)
    contours = tool("inspect.contours", {"image": image}, offset=[100, 200], expected_count=1)
    from_contour = tool(
        "pose.from_feature",
        {"image": image, "features": packet_for(contours.last_result, "collection")},
    )
    np.testing.assert_allclose(from_contour.last_result.outputs["points"], [[69.5, 29.5]])
    # Sort/filter first to choose a ranked object; out-of-range picks cannot reuse stale poses.
    picker = builtin_registry().create("pose.from_feature", parameters={"index": 1})
    picker.execute({"image": image, "features": features})
    assert picker.state == "error" and "index" in picker.error.lower()


def test_feature_with_no_stable_orientation_requires_fixed_angle():
    image = np.zeros((80, 80), np.uint8)
    cv2.circle(image, (40, 40), 12, 255, -1)
    features = tool("inspect.blobs", {"image": image}).last_result.outputs["collection"]
    picker = builtin_registry().create("pose.from_feature", parameters={"angle_mode": "feature"})
    picker.execute({"image": image, "features": features})
    assert picker.state == "error" and "stable orientation" in picker.error
    picker.parameters["angle_mode"] = "fixed"
    picker.execute({"image": image, "features": features})
    assert picker.state == "pass"


def test_pose_rebases_between_image_levels_and_rejects_unrelated_roots():
    image = ImageArray(
        np.zeros((160, 200), np.uint8), "gray", {"spatial": frame([200, 160], "camera")}
    )
    image[60:100, 80:120] = 255
    small = tool("cv.resize", {"image": image}, width=100, height=80).last_result.outputs["image"]
    blobs = tool("inspect.blobs", {"image": small}).last_result.outputs["collection"]
    result = tool("pose.from_feature", {"image": image, "features": blobs}).last_result.outputs
    np.testing.assert_allclose(result["points"], [[99.5, 79.5]])
    assert result["pose"]["scale_x"] == pytest.approx(2)
    small_pose = pose(small, x=20, y=30)
    region = tool(
        "region.pose", {"image": image, "pose": small_pose}, rectangle=[0, 0, 5, 5]
    ).last_result.outputs
    # The full image position and local size both use the pyramid/image geometry.
    assert region["mask"][61, 41] == 255 and region["mask"][68, 48] == 255
    different = image.copy()
    different.metadata = {"spatial": frame([200, 160], "other-camera")}
    block = builtin_registry().create("region.pose")
    block.execute({"image": different, "pose": small_pose})
    assert block.state == "error" and "coordinate roots" in block.error


@pytest.mark.parametrize("kind", ["shape", "texture"])
def test_pattern_offsets_preserve_match_geometry_and_ranked_pose(kind):
    images = tool("input.pattern_demo", {}).last_result.outputs
    reference = images[f"{kind}_reference"]
    matcher = builtin_registry().create(
        f"pattern.match_{kind}",
        parameters={"pose_offset_x": 12.0, "pose_offset_y": -7.0, "pose_offset_angle": 25.0},
    )
    model = matcher.train({"image": reference})
    matcher.execute({"image": reference})
    assert matcher.state == "pass", matcher.error
    outputs = matcher.last_result.outputs
    center = (np.array(model["size"]) - 1) / 2
    np.testing.assert_allclose(outputs["points"][0], center, atol=0.1)
    np.testing.assert_allclose(outputs["pose"]["origin"], center + [12, -7], atol=0.1)
    assert outputs["pose"]["angle_degrees"] == pytest.approx(25, abs=0.1)
    selected = tool("pose.from_feature", {"image": reference, "features": outputs["matches"]})
    np.testing.assert_allclose(selected.last_result.outputs["points"], [outputs["pose"]["origin"]])
    matcher.execute({"image": np.zeros_like(reference)})
    assert matcher.state == "fail" and matcher.last_result.outputs["pose"] == {}
    invalid = builtin_registry().create("region.pose")
    invalid.execute({"image": reference, "pose": {}})
    assert invalid.state == "error" and "no pose" in invalid.error


def test_projective_pose_keeps_full_mapping_and_rejects_crossing_horizon():
    image = np.zeros((120, 160), np.uint8)
    matrix = np.array([[1, 0.1, 50], [0.05, 1, 40], [0.001, 0.002, 1]])
    anchor = make_pose(matrix, [160, 120], {})
    outputs = tool("pose.offset", {"pose": anchor}, pose_offset_x=10).last_result.outputs
    np.testing.assert_allclose(
        outputs["pose"]["source_to_target"], (matrix @ local_offset(10, 0, 0)) / 1.01
    )
    bad = make_pose([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]], [160, 120], {})
    block = builtin_registry().create("region.pose")
    block.execute({"image": image, "pose": bad})
    assert block.state == "error" and "horizon" in block.error


def test_saved_pose_inspection_runs_repeatedly_and_reloads(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(
        Path(__file__).parents[1] / "examples/inspection/pose_inspection/pose_inspection.json",
        registry,
    )
    assert graph.to_dict() == pose_inspection_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
    pose_value = graph.blocks["pose_offset"].last_result.outputs["pose"]
    expected = cv2.getRotationMatrix2D((55.5, 47.5), 10, 1)
    expected[:, 2] += [70, 80]
    center = cv2.transform(np.array([[[76.0, 30.0]]]), expected)[0, 0]
    np.testing.assert_allclose(pose_value["origin"], center, atol=1)
    path = tmp_path / "saved.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    restored.run()
    assert all(b.state == "pass" for b in restored.blocks.values())


@pytest.mark.parametrize(
    "parameters",
    [
        {"rectangle": [0, 0, 0, 10]},
        {"shape": "circle", "radius": -2},
        {"shape": "polygon", "vertices": [[0, 0], [1, 1]]},
        {"rectangle": [0, 0, 20000, 10]},
        {"center": [0, float("inf")], "shape": "circle"},
    ],
)
def test_invalid_relative_region_clears_stale_results(parameters):
    image = np.zeros((80, 100), np.uint8)
    anchor = pose(image, x=40, y=40)
    block = tool("region.pose", {"image": image, "pose": anchor})
    block.parameters.update(parameters)
    block.execute({"image": image, "pose": anchor})
    assert block.state == "error" and block.last_result is None


def test_line_feature_angle_converts_image_clockwise_to_pose_counterclockwise():
    from opencv_dataflow.objects import collection, line_item

    image = np.zeros((100, 100), np.uint8)
    features = collection("lines", [line_item([[40, 10], [40, 70]], 0)], image, {})
    result = tool(
        "pose.from_feature",
        {"image": image, "features": features},
        angle_mode="feature",
        pose_offset_x=5,
    ).last_result.outputs
    np.testing.assert_allclose(result["points"], [[40, 45]])
    assert result["angle_degrees"] == pytest.approx(-90)
