"""Verify selected summaries, locations and scalar output contracts."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


def test_masked_channel_summary_and_extrema_graphics_keep_image_coordinates():
    gray = np.array([[25, 0, 4], [9, 12, 90]], np.uint16)
    image = ImageArray(np.stack([gray, gray + 10, gray + 20, gray + 30], axis=2), "rgba")
    mask = np.array([[0, 255, 255], [255, 255, 0]], np.uint8)
    context = frame([3, 2], "fixture")
    block = builtin_registry().create(
        "cv.image_statistics",
        parameters={
            "channel": "R",
            "statistic": "percentile",
            "value_percentile": 75,
            "percentiles": [0, 100],
            "percentile_method": "linear",
        },
    )
    block.exposed_outputs.append(RESULT_GRAPHICS)
    result = block.execute({"image": ValuePacket(image, {"spatial": context}), "mask": mask})
    assert result is not None, block.error
    summary = result.outputs["summary"]
    assert summary["image_size"] == [3, 2] and result.outputs["selected_count"] == 4
    assert [channel["channel"] for channel in summary["channels"]] == ["R", "G", "B"]
    samples = np.array([0, 4, 9, 12], np.float64)
    red = summary["channels"][0]
    assert red["minimum"] == 0 and red["maximum"] == 12
    assert red["mean"] == pytest.approx(samples.mean())
    assert red["standard_deviation"] == pytest.approx(samples.std())
    assert red["rms"] == pytest.approx(np.sqrt(np.mean(samples**2)))
    assert red["sum"] == 25 and red["nonzero_count"] == 3
    assert result.outputs["value"] == pytest.approx(np.percentile(samples, 75))
    assert red["percentiles"][:2] == [
        {"percentile": 0, "value": 0},
        {"percentile": 100, "value": 12},
    ]
    assert result.outputs["minimum_location"] == [1, 0]
    assert result.outputs["maximum_location"] == [1, 1]
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert [layer.objects[0].points for layer in layers] == [((1.0, 0.0),), ((1.0, 1.0),)]
    assert all(layer.payload["spatial"] == context for layer in layers)


def test_summary_handles_empty_selection_and_preserves_small_float_variation():
    image = 1e12 + np.array([[0, 0.25, 0.5]])
    block = builtin_registry().create(
        "cv.image_statistics", parameters={"statistic": "standard_deviation"}
    )
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["value"] == pytest.approx(np.std([0, 0.25, 0.5]))
    empty = block.execute({"image": image, "mask": np.zeros(image.shape, np.uint8)})
    assert empty is not None, block.error
    assert empty.outputs["selected_count"] == 0
    assert empty.outputs["value"] is NO_VALUE
    assert empty.outputs["minimum_location"] is NO_VALUE
    assert empty.outputs["summary"]["channels"][0]["mean"] is None
    assert not empty.drawings


def test_summary_checks_percentile_fields_and_allows_alpha_selection():
    image = ImageArray(np.array([[[0, 4, 8, 12], [0, 4, 8, 20]]], np.uint8), "bgra")
    block = builtin_registry().create(
        "cv.image_statistics",
        parameters={"channel": "A", "statistic": "percentile", "percentile_method": "higher"},
    )
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid" and "unavailable" in block.error
    block.parameters["include_alpha"] = True
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["value"] == 20
    block.parameters["percentiles"] = [0, 101]
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid" and "between 0 and 100" in block.error
