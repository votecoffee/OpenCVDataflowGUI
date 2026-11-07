"""Check Lab ROI measurements and color-difference reference cases."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE


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


@pytest.mark.local_examples
def test_color_match_example_measures_and_compares_two_regions():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/applications/lab_color_match/lab_color_match.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        compare = graph.blocks["compare"]
        assert compare.state == "pass", compare.error
        assert compare.last_result.outputs["delta_e"] == 0
    finally:
        graph.close()


def test_lab_classifier_ranks_a_wired_palette_and_reports_a_margin():
    registry = builtin_registry()
    block = registry.create(
        "inspect.classify_lab",
        parameters={"metric": "cie76", "maximum_delta_e": 2, "minimum_margin": 1},
    )
    result = block.execute(
        {
            "sample": [50, 1, 0],
            "references": {"other": [50, -4, 0], "target": [50, 0, 0], "third": [50, 20, 0]},
        }
    )
    assert result is not None and result.passed, block.error
    assert result.outputs["label"] == "target"
    assert result.outputs["delta_e"] == 1
    detail = result.outputs["classification"]
    assert detail["margin"] == 4 and detail["status"] == "accepted"
    assert [item["label"] for item in detail["ranking"]] == ["target", "other", "third"]
    assert result.outputs["swatches"].shape == (24, 128, 3)


@pytest.mark.parametrize(
    "sample,reason",
    [
        ([50, 0, 0], "color_class_ambiguous"),
        ([50, 20, 0], "color_class_distance_above_maximum"),
    ],
)
def test_lab_classifier_rejects_without_emitting_a_label(sample, reason):
    block = builtin_registry().create(
        "inspect.classify_lab",
        parameters={"metric": "cie76", "references": {"a": [50, -1, 0], "b": [50, 1, 0]}},
    )
    result = block.execute({"sample": sample})
    assert result is not None and not result.passed, block.error
    assert result.outputs["label"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == reason
    assert result.outputs["classification"]["candidate_label"]


def test_lab_classifier_handles_single_reference_and_rejects_bad_palette():
    block = builtin_registry().create(
        "inspect.classify_lab",
        parameters={"maximum_delta_e": 3, "references": {"blue": [50, 2.6772, -79.7751]}},
    )
    result = block.execute({"sample": [50, 0, -82.7485]})
    assert result is not None and result.passed, block.error
    assert result.outputs["delta_e"] == pytest.approx(2.0425, abs=5e-5)
    assert result.outputs["classification"]["margin"] is None
    assert block.execute({"sample": [50, 0, 0], "references": {"bad": [101, 0, 0]}}) is None
    assert block.error_kind == "invalid"
    assert "Invalid Lab value" in block.error
