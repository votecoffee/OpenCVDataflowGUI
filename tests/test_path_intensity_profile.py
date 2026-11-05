"""Exercise the arc-length profile through its ordinary graph block contract."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame


def test_path_profiles_along_a_bend_in_source_coordinates():
    source = np.arange(25, dtype=np.uint16).reshape(5, 5) * 10
    block = builtin_registry().create(
        "cv.path_intensity_profile",
        parameters={"vertices": [[0, 0], [2, 0], [2, 2]], "sample_spacing": 0.5},
    )
    result = block.execute({"image": source})
    assert result is not None, block.error
    profile = result.outputs["profile"]
    assert profile["distances"] == list(np.arange(9) / 2)
    assert profile["points"][4] == [2, 0] and profile["points"][-1] == [2, 2]
    assert profile["channels"][0]["values"] == [0, 5, 10, 15, 20, 45, 70, 95, 120]
    assert result.outputs["path_length"] == 4
    assert result.outputs["selected_count"] == 9
    assert result.graphics["profile"].shape == (300, 640, 3)
    assert result.drawings["sampling path"].objects[0].points == ((0, 0), (2, 0), (2, 2))


def test_path_keeps_mask_gaps_and_outside_image_samples():
    image = np.tile(np.arange(5, dtype=np.float64), (5, 1))
    mask = np.ones((5, 5), np.uint8)
    mask[:, 2] = 0
    block = builtin_registry().create(
        "cv.path_intensity_profile",
        parameters={"vertices": [[-0.5, 2], [3.5, 2]], "sample_spacing": 0.5},
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["profile"]["channels"][0]["values"] == [
        None,
        0,
        0.5,
        1,
        None,
        None,
        None,
        3,
        3.5,
    ]
    assert result.outputs["selected_count"] == 5


def test_closed_wired_color_path_respects_spatial_frame_and_sample_limit():
    image = ImageArray(np.full((5, 5, 4), [10, 20, 30, 40], dtype=np.uint8), "rgba")
    payload = {"spatial": frame([5, 5], "camera")}
    path = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
    source = ValuePacket(image, payload)
    block = builtin_registry().create(
        "cv.path_intensity_profile",
        parameters={"closed": True, "include_alpha": True, "interpolation": "nearest"},
    )
    result = block.execute({"image": source, "path": ValuePacket(path, payload)})
    assert result is not None, block.error
    assert result.outputs["path_length"] == pytest.approx(8)
    assert result.outputs["points"].count([0, 0]) == 1
    assert result.outputs["profile"]["distances"] == list(range(8))
    channels = result.outputs["profile"]["channels"]
    assert [(row["channel"], row["values"][0]) for row in channels] == [
        ("R", 10),
        ("G", 20),
        ("B", 30),
        ("A", 40),
    ]
    assert result.payloads["points"]["spatial"] == payload["spatial"]
    wrong = ValuePacket(path, {"spatial": frame([5, 5], "other")})
    assert block.execute({"image": source, "path": wrong}) is None
    assert "share the image coordinate frame" in block.error
    assert block.execute({"image": source, "path": [[0, 0], [3000, 0], [3000, 3000]]}) is None
    assert "8192 samples" in block.error
