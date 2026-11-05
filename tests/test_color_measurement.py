"""Check Lab ROI measurements and color-difference reference cases."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray


def test_lab_measurement_selects_pixels_and_preserves_color_order():
    image = np.zeros((12, 20, 3), np.uint8)
    image[:, :10] = [0, 0, 255]
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, :10] = 255
    block = builtin_registry().create("inspect.measure_lab")
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    red = cv2.cvtColor(np.float32([[[0, 0, 1]]]), cv2.COLOR_BGR2Lab)[0, 0]
    np.testing.assert_allclose(result.outputs["lab"], red)
    assert result.outputs["statistics"]["count"] == 120
    np.testing.assert_allclose(result.outputs["statistics"]["standard_deviation"], 0)
    assert result.outputs["swatch"][0, 0, 2] > 250
    rgb = block.execute({"image": ImageArray(image[..., ::-1], "rgb"), "mask": mask})
    assert rgb is not None, block.error
    np.testing.assert_allclose(rgb.outputs["lab"], red)


def test_uint8_lab_uses_opencv_encoding_and_color_comparison_checks_tolerance():
    image = ImageArray(np.full((4, 5, 3), [255, 128, 130], np.uint8), "lab")
    registry = builtin_registry()
    measure = registry.create("inspect.measure_lab")
    measured = measure.execute({"image": image})
    assert measured is not None, measure.error
    np.testing.assert_allclose(measured.outputs["lab"], [100, 0, 2])
    compare = registry.create(
        "inspect.compare_lab",
        parameters={
            "metric": "cie76",
            "maximum_delta_e": 2,
            "reference_lab": [100, 0, 0],
        },
    )
    result = compare.execute({"sample": measured.outputs["lab"]})
    assert result.passed and result.outputs["delta_e"] == 2
    result = compare.execute({"sample": measured.outputs["lab"], "reference": [100, 0, -1]})
    assert not result.passed and result.outputs["delta_e"] == 3
    assert result.fail_reasons[0]["code"] == "color_difference_above_maximum"


# Sharma, Wu and Dalal's supplementary dataset includes neutral and hue-wrap cases.
@pytest.mark.parametrize(
    "reference,sample,expected",
    [
        ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425),
        ([50, 0, 0], [50, -1, 2], 2.3669),
        ([50, 2.49, -0.001], [50, -2.49, 0.001], 7.1792),
        ([50, 2.49, -0.001], [50, -2.49, 0.0011], 7.2195),
    ],
)
def test_ciede2000_reference_pairs(reference, sample, expected):
    block = builtin_registry().create("inspect.compare_lab")
    result = block.execute({"sample": sample, "reference": reference})
    assert result is not None, block.error
    assert result.outputs["delta_e"] == pytest.approx(expected, abs=5e-5)


def test_color_match_example_measures_and_compares_two_regions():
    path = Path(__file__).resolve().parents[1] / "examples/lab_color_match.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        compare = graph.blocks["compare"]
        assert compare.state == "pass", compare.error
        assert compare.last_result.outputs["delta_e"] == 0
    finally:
        graph.close()
