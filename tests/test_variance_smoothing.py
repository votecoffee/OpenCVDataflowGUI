"""Verify native samples, output selection and a measured color step."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "cv.variance_adaptive_smoothing"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/variance_smoothing"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("encoding", ("bgr", "rgb", "bgra", "rgba", "gray"))
def test_matches_native_samples_and_retains_channel_encoding(registry, encoding):
    rng = np.random.default_rng(78)
    bgr = rng.integers(102, 141, (22, 31, 3), dtype=np.uint8)
    source = bgr
    work = bgr
    if encoding == "gray":
        source = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        work = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif encoding.startswith("rgb"):
        source = bgr[..., ::-1].copy()
    expected = cv2.ximgproc.edgePreservingFilter(work, 4, 38.0)
    if encoding == "gray":
        expected = expected[..., 0]
    elif encoding.startswith("rgb"):
        expected = expected[..., ::-1]
    if encoding in ("rgba", "bgra"):
        alpha = rng.integers(0, 256, bgr.shape[:2], dtype=np.uint8)
        source = np.dstack((source, alpha))
        expected = np.dstack((expected, alpha))
    block = registry.create(TYPE, parameters={"diameter": 4, "threshold": 38.0})
    source = ImageArray(source, encoding)
    before = source.copy()
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint8 and output.color_format == encoding
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(output[-1], before[-1])
    np.testing.assert_array_equal(output[:, -1], before[:, -1])
    np.testing.assert_array_equal(source, before)
    assert np.any(output != before)


def test_mask_and_region_limit_writes_with_source_frame_and_ticket(registry):
    source = np.random.default_rng(779).integers(90, 130, (20, 30, 3), dtype=np.uint8)
    spatial = frame([30, 20], "camera")
    ticket = TicketClock("smoothing").issue("camera", "fixed", "part")
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[1:14, 3:23] = 255
    region = {
        "model": "image_region_v1", "shape": "rectangle",
        "image_size": [30, 20], "rectangle": [10, 5, 18, 8],
    }
    block = registry.create(TYPE)
    result = block.execute(
        {
            "image": ValuePacket(source, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    selected = np.zeros(mask.shape, bool)
    selected[5:13, 10:23] = True
    expected = cv2.ximgproc.edgePreservingFilter(source, 7, 30.0)
    actual = result.outputs["image"]
    np.testing.assert_array_equal(actual[selected], expected[selected])
    np.testing.assert_array_equal(actual[~selected], source[~selected])
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == spatial


def test_zero_threshold_is_an_exact_copy_without_native_filter(registry, monkeypatch):
    monkeypatch.delattr(cv2.ximgproc, "edgePreservingFilter")
    source = np.random.default_rng(87).integers(0, 256, (18, 19), dtype=np.uint8)
    block = registry.create(TYPE, parameters={"threshold": 0})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], source)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("small", "larger than the diameter"),
        ("work", "work exceeds the limit"),
        ("pixels", "up to 1,048,576 pixels"),
        ("depth", "needs 8-bit input"),
        ("encoding", "Convert color"),
        ("premultiplied", "Unassociate premultiplied alpha"),
        ("threshold", "Threshold must be"),
    ],
)
def test_invalid_inputs_are_rejected_before_native_call(registry, monkeypatch, case, message):
    def unexpected(*args):
        raise AssertionError("Invalid input reached OpenCV")

    monkeypatch.setattr(cv2.ximgproc, "edgePreservingFilter", unexpected)
    source = np.zeros((24, 32, 3), np.uint8)
    params = {}
    if case == "small":
        source = source[:7]
    elif case == "work":
        source = np.zeros((256, 256), np.uint8)
        params["diameter"] = 31
    elif case == "pixels":
        source = np.zeros((1025, 1025), np.uint8)
    elif case == "depth":
        source = source.astype(np.uint16)
    elif case == "encoding":
        source = ImageArray(source, "lab")
    elif case == "premultiplied":
        source = ImageArray(
            np.zeros((24, 32, 4), np.uint8), "rgba", {"alpha_representation": "premultiplied"}
        )
    elif case == "threshold":
        params["threshold"] = -1
    block = registry.create(TYPE, parameters=params)
    assert block.execute({"image": source}) is None
    assert message in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_reports_noise_and_edge_strength(registry, profile):
    graph = Graph.load(EXAMPLE / "variance_smoothing.json", registry)
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        node = graph.blocks["adaptive"]
        assert block_entry(node).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        assert all(block.state == "pass" for block in graph.processing_blocks())
        clean_step = np.mean([164 - 58, 184 - 70, 209 - 88])
        half = np.mean([58, 70, 88, 164, 184, 209])
        for key, rmse, retained in (
            ("adaptive", 1.24, 0.990), ("gaussian", 2.29, 0.279), ("noisy", 11.14, 0.982),
        ):
            measure = graph.blocks[f"{key}_rmse"].last_result.outputs["value"]
            assert measure == pytest.approx(rmse, abs=0.02)
            record = graph.blocks[f"{key}_profile"].last_result.outputs["profile"]
            series = np.asarray([channel["values"] for channel in record["channels"]]).mean(axis=0)
            contrast = series[80] - series[79]
            assert contrast / clean_step == pytest.approx(retained, abs=0.004)
            center = 79 + (half - series[79]) / contrast
            assert center == pytest.approx(79.5, abs=0.02)
    finally:
        graph.close()
