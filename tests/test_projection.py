"""Verify profile reductions and selection against direct sample calculations."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("statistic", ["sum", "mean", "minimum", "maximum", "sum_of_squares"])
def test_masked_profiles_preserve_pixel_indices_and_channels(axis, statistic):
    gray = np.arange(20, dtype=np.uint16).reshape(4, 5)
    image = ImageArray(np.stack([gray, gray + 100, gray + 200, gray + 300], axis=2), "rgba")
    mask = np.full(gray.shape, 255, np.uint8)
    mask[2] = mask[:, 1] = 0
    mask[0, 3] = 0
    block = builtin_registry().create(
        "cv.intensity_profile", parameters={"axis": axis, "statistic": statistic}
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    profile = result.outputs["profile"]
    assert profile["axis"] == axis
    assert profile["image_size"] == [5, 4]
    assert [channel["channel"] for channel in profile["channels"]] == ["R", "G", "B"]
    assert result.outputs["selected_count"] == np.count_nonzero(mask)
    assert result.graphics["profile"].shape == (300, 640, 3)
    reduction = {
        "sum": np.sum,
        "mean": np.mean,
        "minimum": np.min,
        "maximum": np.max,
        "sum_of_squares": lambda values: np.sum(values**2),
    }[statistic]
    count = gray.shape[0 if axis == "rows" else 1]
    for index in range(count):
        line = image[index] if axis == "rows" else image[:, index]
        selected = mask[index] != 0 if axis == "rows" else mask[:, index] != 0
        assert profile["sample_counts"][index] == np.count_nonzero(selected)
        for channel, data in enumerate(profile["channels"]):
            if selected.any():
                expected = reduction(np.asarray(line[selected, channel], dtype=np.float64))
                assert data["values"][index] == pytest.approx(expected)
                if statistic in ("minimum", "maximum"):
                    pixel = int(np.flatnonzero(selected & (line[:, channel] == expected))[0])
                    point = [pixel, index] if axis == "rows" else [index, pixel]
                    assert data["locations"][index] == point
            else:
                assert data["values"][index] is None
                if statistic in ("minimum", "maximum"):
                    assert data["locations"][index] is None
    if statistic in ("minimum", "maximum"):
        locations = profile["channels"][0]["locations"]
        assert result.outputs["points"] == [point for point in locations if point is not None]
    else:
        assert result.outputs["points"] is NO_VALUE
        assert all("locations" not in channel for channel in profile["channels"])


def test_profile_includes_optional_alpha_and_handles_empty_selection():
    image = ImageArray(np.array([[[8, 9, 10, 11]]], np.uint8), "bgra")
    block = builtin_registry().create("cv.intensity_profile", parameters={"include_alpha": True})
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["profile"]["channels"][-1] == {"channel": "A", "values": [11.0]}
    empty = block.execute({"image": image, "mask": np.zeros((1, 1), np.uint8)})
    assert empty is not None, block.error
    assert empty.outputs["selected_count"] == 0
    assert empty.outputs["profile"]["channels"][-1]["values"] == [None]
    assert empty.graphics["profile"].any()


def test_profile_reports_nonfinite_reduction_as_invalid():
    block = builtin_registry().create("cv.intensity_profile", parameters={"statistic": "sum"})
    assert block.execute({"image": np.full((2, 1), np.finfo(np.float64).max)}) is None
    assert block.error_kind == "invalid"
    assert "overflowed" in block.error


@pytest.mark.parametrize("tie,expected", [("first", [[0, 0], [2, 0]]), ("last", [[0, 2], [2, 1]])])
def test_extrema_ties_use_selected_channel_and_ignore_empty_columns(tie, expected):
    red = np.array([[9, 5, 0], [1, 9, 3], [2, 2, 8]], np.float64)
    blue = np.array([[4, 4, 8], [4, 4, 8], [4, 4, 1]], np.float64)
    image = ImageArray(np.stack((red, red, blue), axis=2), "rgb")
    mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]], np.uint8)
    block = builtin_registry().create(
        "cv.intensity_profile",
        parameters={"statistic": "maximum", "extrema_tie": tie, "points_channel": "B"},
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["points"] == expected
    channels = result.outputs["profile"]["channels"]
    assert channels[0]["locations"] == [[0, 0], None, [2, 2]]
    assert channels[-1]["locations"] == [expected[0], None, expected[1]]
