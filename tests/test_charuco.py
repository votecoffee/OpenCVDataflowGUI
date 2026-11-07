"""Verify identified board coordinates and pose wiring from partial ChArUco views."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, packet_for

_PROJECT = Path(__file__).parents[1] / "examples/opencv tools/charuco_board/charuco_board.json"


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_cropped_board_graph_identifies_points_and_recovers_pose(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    detect = graph.blocks["detect"].last_result
    ids = detect.outputs["corner_ids"]
    assert ids == sorted(set(ids))
    assert 6 <= len(ids) < 24 and 0 not in ids
    expected = [[(index % 6 + 1) * 0.03, (index // 6 + 1) * 0.03, 0] for index in ids]
    np.testing.assert_allclose(detect.outputs["object_points"], expected, atol=1e-7)
    assert detect.payloads["object_points"]["object_point_units"] == "m"
    assert "spatial" not in detect.payloads["object_points"]
    assert detect.outputs["observation"]["image_size"] == [640, 480]
    assert detect.outputs["observation"]["visible_marker_ids"]
    assert len(detect.outputs["result_graphics"].layers) == 2
    pose = graph.blocks["pose"].last_result.outputs
    np.testing.assert_allclose(pose["pose"]["translation"], [-0.145, -0.082, 0.285], atol=0.001)
    assert pose["rms"] < 0.3
    image = graph.blocks["overlay"].last_result.outputs["image"]
    assert image.shape == (480, 640, 3) and np.any(image[..., 0] != image[..., 1])


def test_custom_marker_ids_and_legacy_board_survive_image_rotation():
    marker_ids = list(range(20, 32))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((6, 4), 0.03, 0.022, dictionary, np.int32(marker_ids))
    board.setLegacyPattern(True)
    image = board.generateImage((424, 296), marginSize=20)
    detector = builtin_registry().create(
        "calibration.charuco",
        parameters={"columns": 6, "rows": 4, "legacy_pattern": True, "marker_ids": marker_ids},
    )
    original = detector.execute({"image": image})
    rotated = detector.execute({"image": cv2.rotate(image, cv2.ROTATE_180)})
    assert original is not None and original.passed and rotated is not None and rotated.passed
    assert original.outputs["corner_ids"] == rotated.outputs["corner_ids"] == list(range(15))
    assert original.outputs["object_points"] == rotated.outputs["object_points"]
    assert rotated.outputs["observation"]["board"]["marker_ids"] == marker_ids
    np.testing.assert_allclose(
        rotated.outputs["image_points"],
        [423, 295] - np.asarray(original.outputs["image_points"]),
        atol=0.1,
    )


def test_missing_and_collinear_observations_suppress_point_outputs(monkeypatch):
    detector = builtin_registry().create("calibration.charuco", parameters={"minimum_corners": 4})
    image = np.full((180, 240), 255, np.uint8)
    missing = detector.execute({"image": image})
    assert missing is not None and not missing.passed
    assert missing.outputs["corner_count"] == 0
    assert missing.fail_reasons[0]["code"] == "charuco_corner_count"

    class CollinearDetector:
        def __init__(self, *args):
            pass

        def detectBoard(self, gray):
            corners = np.float32([[20, 20], [40, 20], [60, 20], [80, 20]])
            return corners, np.arange(4, dtype=np.int32), (), None

    monkeypatch.setattr(cv2.aruco, "CharucoDetector", CollinearDetector)
    collinear = detector.execute({"image": image})
    assert collinear is not None and not collinear.passed
    assert collinear.fail_reasons[0]["code"] == "charuco_collinear"
    assert collinear.outputs["corner_ids"] == [0, 1, 2, 3]
    for result in (missing, collinear):
        assert result.outputs["image_points"] is result.outputs["object_points"] is NO_VALUE


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"marker_length": 0.04}, "less than square length"),
        ({"marker_ids": [1] * 17}, "unique marker IDs"),
        ({"columns": 20, "rows": 20}, "fewer IDs"),
    ],
)
def test_invalid_board_configuration_is_reported(changes, message):
    detector = builtin_registry().create("calibration.charuco", parameters=changes)
    assert detector.execute({"image": np.zeros((80, 100), np.uint8)}) is None
    assert message in detector.error


def test_pnp_rejects_conflicting_charuco_object_point_units():
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.run()
    detect = graph.blocks["detect"].last_result
    camera = graph.blocks["camera"].last_result.outputs["camera"]
    pose = graph.blocks["pose"]
    result = pose.execute(
        {
            "camera": {**camera, "units": "mm"},
            "image_points": packet_for(detect, "image_points"),
            "object_points": packet_for(detect, "object_points"),
        }
    )
    assert result is None
    assert "object point units" in pose.error
