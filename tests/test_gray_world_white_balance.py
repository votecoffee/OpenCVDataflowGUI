"""Qualify native gray-world output, sample selection and failed estimates."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.gray_world_white_balance"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/gray_world_white_balance"


def tool(threshold):
    return builtin_registry().create(TYPE, parameters={"saturation_threshold": threshold})


@pytest.mark.parametrize(("dtype", "cutoff"), [(np.uint8, 0.63), (np.uint16, 0.62)])
def test_native_parity_and_reported_fixed_point_gains(dtype, cutoff):
    peak = np.iinfo(dtype).max
    rng = np.random.default_rng(478)
    source = rng.integers(0, peak + 1, (43, 61, 4), dtype=dtype)
    tagged = ImageArray(source[..., [2, 1, 0, 3]][:, ::2], "rgba")
    before = np.asarray(tagged).copy()
    block = tool(cutoff)
    result = block.execute({"image": tagged})
    assert result and result.passed, block.error
    image = result.outputs["image"]
    bgr = np.ascontiguousarray(before[..., [2, 1, 0]])
    native = cv2.xphoto.createGrayworldWB()
    native.setSaturationThreshold(cutoff)
    expected = native.balanceWhite(bgr)
    assert image_format(image) == "rgba" and image.dtype == dtype
    np.testing.assert_array_equal(image[..., :3], expected[..., ::-1])
    np.testing.assert_array_equal(image[..., 3], before[..., 3])
    np.testing.assert_array_equal(tagged, before)
    scale = peak + 1
    gains = np.array([result.outputs["gains"][key] for key in "BGR"])
    reconstructed = (bgr.astype(np.uint64) * np.rint(gains * scale).astype(np.uint64)) // scale
    np.testing.assert_array_equal(reconstructed, expected)
    admitted = result.graphics["Estimation support"] != 0
    highest = bgr.max(axis=2).astype(np.uint64)
    lowest = bgr.min(axis=2).astype(np.uint64)
    cutoff_code = round(float(np.float32(cutoff) * np.float32(peak)))
    expected_mask = ((highest - lowest) * peak <= cutoff_code * highest) & (highest != 0)
    np.testing.assert_array_equal(admitted, expected_mask)
    report = result.outputs["report"]
    assert report["used_pixels"] == int(expected_mask.sum())
    assert (
        report["rejected_saturated_pixels"] + report["used_pixels"] + report["black_pixels"]
        == bgr.shape[0] * bgr.shape[1]
    )


def test_region_and_mask_intersect_then_gains_apply_to_entire_ticketed_image():
    bgr = np.full((16, 24, 3), [180, 140, 60], np.uint8)
    bgr[2:12, 4:19] = [40, 60, 80]
    bgr[2:4, 4:10] = [5, 250, 15]
    selection = np.zeros((16, 24), np.uint8)
    selection[:9, :13] = 255
    context = frame([24, 16], "ticket-image")
    ticket = TicketClock("test").issue("camera", "fixed", "part-17")
    region = {
        "model": "image_region_v1",
        "image_size": [24, 16],
        "shape": "rectangle",
        "rectangle": [4, 2, 15, 10],
        "coordinate_context": context,
    }
    block = tool(0.6)
    result = block.execute(
        {
            "image": ValuePacket(bgr, {"spatial": context, "ticket": ticket}),
            "mask": ValuePacket(selection, {"spatial": context}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == context
    report = result.outputs["report"]
    assert report["candidate_pixels"] == 63
    assert report["used_pixels"] == 51 and report["rejected_saturated_pixels"] == 12
    assert report["application"] == "selected_gains"
    np.testing.assert_array_equal(result.outputs["image"][5, 10], [40, 40, 40])
    np.testing.assert_array_equal(result.outputs["image"][15, 23], [180, 93, 30])


def test_threshold_endpoints_and_insufficient_estimation_support():
    samples = np.array([[[45, 45, 45], [50, 100, 180], [0, 0, 0]]], np.uint8)
    block = tool(0)
    result = block.execute({"image": samples})
    assert result and result.passed, block.error
    assert result.outputs["report"]["used_pixels"] == 1
    assert result.outputs["report"]["black_pixels"] == 1
    np.testing.assert_array_equal(result.outputs["image"], samples)

    failed = block.execute({"image": samples[:, 1:]})
    assert failed and not failed.passed
    assert failed.fail_reasons[0]["code"] == "gray_world_no_channel_support"
    assert failed.outputs["image"] is NO_VALUE and failed.outputs["gains"] is NO_VALUE

    all_saturated = np.full((5, 6, 3), [0, 30, 80], np.uint8)
    result = tool(1.0).execute({"image": all_saturated})
    assert result and not result.passed
    assert result.outputs["report"]["used_pixels"] == 30
    assert result.outputs["report"]["status"] == "insufficient_channel_support"
    assert result.outputs["image"] is NO_VALUE


def test_bad_image_or_selection_is_invalid():
    block = tool(0.6)
    gray = np.full((4, 4), 80, np.uint8)
    assert block.execute({"image": gray}) is None and "three color channels" in block.error
    color = np.full((4, 4, 3), 80, np.uint8)
    assert block.execute({"image": color.astype(np.float32)}) is None
    assert "8- or 16-bit" in block.error
    assert (
        block.execute(
            {
                "image": ValuePacket(color, {"spatial": frame([4, 4], "image")}),
                "mask": ValuePacket(gray, {"spatial": frame([4, 4], "other")}),
            }
        )
        is None
    )
    assert "coordinate frame" in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize(("depth", "profile"), [("uint8", "debug"), ("uint16", "lean")])
def test_saved_gray_world_projects_neutralize_patch(depth, profile):
    graph = Graph.load(EXAMPLE / f"gray_world_{depth}.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        for key, count in (
            ("all_pixels", 14144),
            ("saturation_filtered", 5824),
            ("region_filtered", 1456),
        ):
            block = graph.blocks[key]
            result = block.last_result
            assert result and result.passed, block.error
            image = result.outputs["image"]
            assert str(image.dtype) == depth
            assert result.outputs["report"]["used_pixels"] == count
            patch = image[53, 36]
            if key == "all_pixels":
                assert np.ptp(patch) > 3
            else:
                assert np.ptp(patch) <= 1
            assert bool(result.graphics) == (profile == "debug")
    finally:
        graph.close()
