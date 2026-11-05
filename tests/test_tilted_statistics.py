"""Verify the 45-degree sampling lattice against direct pixel selection."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.images import ImageArray


def tilted_pixels(shape, window):
    x, y, width, height = window
    yy, xx = np.indices(shape)
    u, v = xx + yy, xx - yy
    start = x + y - 1
    stop = x - y
    return (u >= start) & (u < start + 2 * width) & (v >= stop - 2 * height) & (v < stop)


@pytest.mark.parametrize("masked", [False, True])
def test_tilted_statistics_match_direct_masks_and_boundary_windows(masked):
    rng = np.random.default_rng(45)
    source = rng.integers(0, 1000, size=(10, 11, 4), dtype=np.uint16)
    image = ImageArray(source, "rgba")
    windows = [[5, 1, 2, 3], [3, 0, 4, 3], [9, 3, 2, 5], [1, 0, 1, 1]]
    mask = (rng.random(source.shape[:2]) > 0.35).astype(np.uint8)
    block = builtin_registry().create("cv.tilted_statistics", parameters={"include_alpha": True})
    block.exposed_outputs.append(RESULT_GRAPHICS)
    inputs = {"image": image, "windows": windows}
    if masked:
        inputs["mask"] = mask
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["count"] == len(windows)
    assert len(result.outputs[RESULT_GRAPHICS].layers[0].objects) == len(windows)
    for window, record in zip(windows, result.outputs["measurements"]["windows"], strict=True):
        expected = tilted_pixels(source.shape[:2], window)
        if masked:
            expected &= mask != 0
        samples = source[expected]
        assert record["sample_count"] == len(samples)
        assert [row["channel"] for row in record["channels"]] == ["R", "G", "B", "A"]
        for channel, measured in enumerate(record["channels"]):
            values = samples[:, channel]
            assert measured["sum"] == values.sum()
            if len(values):
                assert measured["mean"] == pytest.approx(values.mean())
                assert measured["standard_deviation"] == pytest.approx(values.std())
            else:
                assert measured["mean"] is None
                assert measured["standard_deviation"] is None


def test_tilted_statistics_keeps_low_float_contrast_and_reports_empty_windows():
    window = [4, 0, 3, 3]
    image = 1e12 + np.arange(64).reshape(8, 8) / 4
    block = builtin_registry().create("cv.tilted_statistics", parameters={"windows": [window]})
    result = block.execute({"image": image})
    assert result is not None, block.error
    samples = image[tilted_pixels(image.shape, window)]
    summary = result.outputs["measurements"]["windows"][0]["channels"][0]
    assert summary["standard_deviation"] == pytest.approx((samples - 1e12).std())
    empty = block.execute({"image": image, "mask": np.zeros(image.shape, np.uint8)})
    assert empty is not None, block.error
    record = empty.outputs["measurements"]["windows"][0]
    assert record["sample_count"] == 0
    assert record["channels"][0] == {
        "channel": "Gray",
        "sum": 0.0,
        "mean": None,
        "standard_deviation": None,
    }
    assert block.execute({"image": image, "windows": [[1, 0, 3, 3]]}) is None
    assert block.error_kind == "invalid" and "outside" in block.error
