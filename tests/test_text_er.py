"""Validate native ER grouping and contrast-aware component extraction."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ocr import text_rectangle
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.text_er import TextExtremalRegions
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/native_text_regions"
MODELS = Path(__file__).parents[1] / "src/opencv_dataflow/models/text_er"


def iou(a, b):
    x, y, w, h = a
    u, v, p, q = b
    intersection = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return intersection / (w * h + p * q - intersection)


@pytest.fixture
def image():
    return cv2.imread(str(EXAMPLE / "assets/text_chart.png"), cv2.IMREAD_COLOR)


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_text_groups_match_independent_word_bounds(profile, workers, image):
    truth = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))["words"]
    graph = Graph.load(EXAMPLE / "native_text_regions.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["groups"]
        assert block_entry(block).path[:2] == ("OpenCV", "Scene Text Detection (text)")
        result = block.last_result
        assert result and result.passed, block.error
        regions = result.outputs["regions"]
        assert regions["model"] == "ocr_regions_v1"
        assert len(regions["items"]) == result.outputs["count"] == len(truth) == 3
        for word, group in zip(truth, regions["items"], strict=True):
            assert iou(word["bounds"], group["bounds"]) > 0.90
            assert text_rectangle(group["corners"]).shape == (4, 2)
        np.testing.assert_array_equal(result.outputs["image"], image)
        assert result.payloads["regions"]["spatial"]["root_id"] == "source:image"
        assert result.payloads["regions"]["ticket"]["value"] == "text_chart.png"
        assert graph.blocks["parts"].last_result.outputs["count"] == 25
        assert graph.blocks["filmstrip"].last_result.passed
        if profile == "debug":
            assert len(result.drawings["Text regions"].objects) == 3
        else:
            assert not result.drawings
    finally:
        graph.close()


def native_filters():
    return (
        cv2.text.createERFilterNM1(str(MODELS / "trained_classifierNM1.xml"), 8),
        cv2.text.createERFilterNM2(str(MODELS / "trained_classifierNM2.xml")),
    )


def test_grouping_matches_native_rgb_and_respects_same_grid_selection(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    native = cv2.text.detectRegions(
        rgb, *native_filters(), method=cv2.text.ERGROUPING_ORIENTATION_HORIZ
    )
    source = ImageArray(rgb.astype(np.uint16) * 257, "rgb")
    context = frame([430, 270], "camera")
    roi = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [0, 89, 280, 170],
        "image_size": [430, 270],
        "coordinate_context": context,
    }
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:180] = 255
    block = TextExtremalRegions()
    result = block.execute(
        {
            "image": ValuePacket(source, {"spatial": context}),
            "mask": ValuePacket(mask, {"spatial": context}),
            "region": roi,
        }
    )
    assert result and result.passed, block.error
    expected = sorted(native, key=lambda bounds: (bounds[1], bounds[0]))[1]
    np.testing.assert_array_equal(result.outputs["regions"]["items"][0]["bounds"], expected)
    assert result.outputs["count"] == 1
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    assert result.payloads["regions"]["spatial"] == context


@pytest.mark.parametrize(
    ("mode", "flag"),
    [
        ("gray", None),
        ("rgb_lightness_gradient", cv2.text.ERFILTER_NM_RGBLGrad),
        ("hsv_gradient", cv2.text.ERFILTER_NM_IHSGrad),
    ],
)
def test_component_channels_match_native_bounds(image, mode, flag):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channel_images = (
        (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),)
        if flag is None
        else (cv2.text.computeNMChannels(rgb, _mode=flag))
    )
    native = set()
    filters = native_filters()
    for plane in channel_images:
        native.update(
            box
            for contour in cv2.text.detectRegions(cv2.bitwise_not(plane), *filters)
            if (box := cv2.boundingRect(contour))[2] > 1 and box[3] > 1
        )
    block = TextExtremalRegions(
        parameters={
            "mode": "components",
            "component_channels": mode,
            "polarity": "bright",
        }
    )
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    regions = result.outputs["regions"]["items"]
    assert {tuple(region["bounds"]) for region in regions} == native
    assert all(region["polarity"] == "bright" for region in regions)


def test_blank_and_truncated_results_keep_useful_failure_reasons(image):
    block = TextExtremalRegions(parameters={"maximum_regions": 1})
    blank = block.execute({"image": np.zeros_like(image)})
    assert blank and not blank.passed
    assert blank.outputs["regions"]["items"] == []
    assert blank.fail_reasons[0]["code"] == "text_er.count"
    limited = block.execute({"image": image})
    assert limited and not limited.passed
    assert limited.outputs["count"] == 1 and limited.outputs["report"]["selected_regions"] == 3
    assert limited.fail_reasons[0]["code"] == "text_er.truncated"


def test_classifier_override_validates_stage_and_reports_digest(tmp_path, image):
    path = tmp_path / "classifier.xml"
    block = TextExtremalRegions(parameters={"nm2_model_path": path.name})
    block.project_dir = tmp_path
    path.write_bytes((MODELS / "trained_classifierNM1.xml").read_bytes())
    assert block.execute({"image": image}) is None
    assert "expected text classifier stage" in block.error
    path.write_bytes((MODELS / "trained_classifierNM2.xml").read_bytes())
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    assert result.outputs["report"]["models"]["NM2"] == {
        "source": "local_file",
        "sha256": "5e712fefc183584a9ac51544a291a53e070ba2ad785370168e774a905e2220a3",
    }
