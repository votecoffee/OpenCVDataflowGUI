"""Verify MSER polarity, selection, and the native color detector."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE


@pytest.fixture
def extrema():
    y, x = np.mgrid[:140, :220]
    dark = np.exp(-((x - 52) ** 2 + (y - 65) ** 2) / 200)
    bright = np.exp(-((x - 155) ** 2 + (y - 75) ** 2) / 200)
    return np.rint(130 - 100 * dark + 100 * bright).astype(np.uint8)


@pytest.mark.parametrize(
    "polarity,locations",
    [
        ("dark", {(52.0, 65.0)}),
        ("bright", {(155.0, 75.0)}),
        ("both", {(52.0, 65.0), (155.0, 75.0)}),
    ],
)
def test_mser_grayscale_polarity(extrema, polarity, locations):
    block = builtin_registry().create(
        "inspect.mser", parameters={"polarity": polarity, "min_area": 35, "max_area": 3000}
    )
    result = block.execute({"image": extrema})
    assert result is not None, block.error
    items = result.outputs["regions"]["items"]
    assert {tuple(item["center"]) for item in items} == locations
    assert result.outputs["count"] == len(items)
    assert len(result.drawings["MSER regions"].objects) == len(items)
    assert all(result.outputs["mask"][int(y), int(x)] == 255 for x, y in locations)


def test_mser_roi_keeps_only_whole_regions_and_reports_truncation(extrema):
    block = builtin_registry().create(
        "inspect.mser", parameters={"min_area": 35, "max_area": 3000, "draw_overlay": False}
    )
    selection = np.zeros_like(extrema)
    selection[:, :110] = 255
    result = block.execute({"image": extrema, "mask": selection})
    assert result is not None and result.outputs["count"] == 2, block.error
    assert result.outputs["overlay"] is NO_VALUE
    assert not result.outputs["mask"][:, 110:].any()
    # The center remains selected, but both regions cross this mask boundary.
    selection[:, 58:] = 0
    result = block.execute({"image": extrema, "mask": selection})
    assert result.outputs["count"] == 0
    block.parameters["max_regions"] = 1
    result = block.execute({"image": extrema})
    assert result.outputs["count"] == 1
    assert result.outputs["regions"]["truncated"]


def test_mser_color_uses_native_regions_for_rgb_input():
    rng = np.random.default_rng(2)
    bgr = cv2.GaussianBlur(rng.integers(0, 256, (100, 140, 3), dtype=np.uint8), (5, 5), 1)
    native_regions, native_boxes = cv2.MSER_create(5, 8, 2000).detectRegions(bgr)
    block = builtin_registry().create(
        "inspect.mser", parameters={"mode": "color", "min_area": 8, "max_area": 2000}
    )
    result = block.execute({"image": ImageArray(bgr[..., ::-1], "rgb")})
    assert result is not None, block.error
    assert result.outputs["count"] == len(native_regions) > 0
    assert [r["bounds"] for r in result.outputs["regions"]["items"]] == native_boxes.tolist()
