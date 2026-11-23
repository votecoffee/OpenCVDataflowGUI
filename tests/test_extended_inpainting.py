"""Qualify xphoto mask polarity, typed image output, and recovery quality."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.extended_inpaint"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/extended_inpainting"


def tool(method="fsr_fast"):
    return builtin_registry().create(TYPE, parameters={"method": method})


def waves():
    y, x = np.indices((40, 48))
    samples = np.rint(115 + 19 * np.cos(2 * np.pi * x / 12) + 14 * np.sin(2 * np.pi * y / 10))
    return np.stack((samples - 23, samples, samples + 45), axis=2).astype(np.uint8)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_frequency_profiles_use_byte_destinations_and_keep_source_type(dtype):
    canonical = waves()
    mask = np.zeros(canonical.shape[:2], np.uint8)
    mask[12:18, 20:28] = 255
    selected = mask != 0
    bgr = canonical.astype(dtype)
    if dtype == np.uint16:
        bgr = bgr * 257 + 17
    elif dtype in (np.float32, np.float64):
        bgr = (bgr + np.asarray(0.2, dtype=dtype)) / 255
    original = ImageArray(bgr[..., ::-1], "rgb")
    original[selected] = 1 if np.issubdtype(dtype, np.floating) else 3
    byte_source = canonical.copy()
    byte_source[selected] = 0
    expected = np.empty_like(byte_source)
    cv2.xphoto.inpaint(byte_source, ~mask, expected, cv2.xphoto.INPAINT_FSR_FAST)
    node = tool()
    result = node.execute({"image": original, "mask": mask})
    assert result and result.passed, node.error
    output = result.outputs["image"]
    assert output.dtype == dtype and image_format(output) == "rgb"
    if dtype == np.uint16:
        expected = expected.astype(dtype) * 257
    elif dtype in (np.float32, np.float64):
        expected = expected.astype(dtype) / 255
    np.testing.assert_allclose(output[selected], expected[..., ::-1][selected], rtol=1e-7)
    np.testing.assert_array_equal(output[~selected], original[~selected])
    np.testing.assert_array_equal(result.outputs["repair_mask"], mask)


def test_mask_region_intersection_and_ticket_keep_alpha_and_measured_pixels():
    signal = waves()
    height, width = signal.shape[:2]
    alpha = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    image = np.dstack((signal.astype(np.uint16) * 257, alpha))
    selection = np.zeros((height, width), np.uint8)
    selection[8:18, 13:24] = 11
    context = frame([width, height], "registered")
    ticket = TicketClock("inpaint").issue("camera", "fixed", "plate")
    region = {
        "model": "image_region_v1",
        "image_size": [width, height],
        "shape": "rectangle",
        "rectangle": [16, 10, 13, 15],
        "coordinate_context": context,
    }
    block = tool("fsr_best")
    result = block.execute(
        {
            "image": ValuePacket(ImageArray(image, "bgra"), {"spatial": context, "ticket": ticket}),
            "mask": ValuePacket(selection, {"spatial": context}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    wanted = np.zeros_like(selection)
    wanted[10:18, 16:24] = 255
    np.testing.assert_array_equal(result.outputs["repair_mask"], wanted)
    np.testing.assert_array_equal(result.outputs["image"][wanted == 0], image[wanted == 0])
    np.testing.assert_array_equal(result.outputs["image"][..., 3], alpha)
    for port in ("image", "repair_mask"):
        assert result.payloads[port]["spatial"] == context
        assert result.payloads[port]["ticket"] == ticket


def test_shift_map_keeps_high_depth_and_limits_native_graph():
    source = waves().astype(np.uint16) * 257 + 89
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[10:16, 24:30] = 255
    measured = source.copy()
    measured[mask != 0] = 0
    node = tool("shift_map")
    result = node.execute({"image": measured, "mask": mask})
    assert result and result.passed, node.error
    native = np.empty_like(measured)
    cv2.xphoto.inpaint(measured, ~mask, native, cv2.xphoto.INPAINT_SHIFTMAP)
    out = result.outputs["image"]
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out[mask != 0], native[mask != 0])
    np.testing.assert_array_equal(out[mask == 0], measured[mask == 0])

    assert node.execute({"image": measured[:15], "mask": mask[:15]}) is None
    assert "32 × 32" in node.error
    striped = np.zeros_like(mask)
    striped[:, ::2] = 255
    assert node.execute({"image": measured, "mask": striped}) is None
    assert "fully known 8" in node.error
    wide_mask = np.zeros((250, 250), np.uint8)
    wide_mask[18:233, 18:233] = 255
    assert node.execute({"image": wide_mask, "mask": wide_mask}) is None
    assert "40,000" in node.error


def test_missing_full_empty_and_off_frame_selection():
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    source = ValuePacket(image, {"spatial": frame([6, 5], "source")})
    mask = np.zeros((5, 6), np.uint8)
    node = tool()
    assert node.execute({"image": source}) is None
    assert "Connect an inpainting mask or region" in node.error
    result = node.execute({"image": source, "mask": mask})
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["image"], image)
    np.testing.assert_array_equal(result.outputs["repair_mask"], mask)
    mask[:] = 255
    assert node.execute({"image": source, "mask": mask}) is None
    assert "needs known pixels" in node.error
    other = ValuePacket(mask, {"spatial": frame([6, 5], "different")})
    assert node.execute({"image": source, "mask": other}) is None
    assert "coordinate frame" in node.error
    mask[0, 0] = 0
    assert node.execute({"image": image.astype(np.float32) + 2, "mask": mask}) is None
    assert "Known floating-point" in node.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_recovers_known_periodic_surface(profile):
    graph = Graph.load(EXAMPLE / "extended_inpainting.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        clean = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_UNCHANGED)
        source = graph.blocks["damaged"].last_result.outputs["image"]
        selected = np.zeros(clean.shape[:2], bool)
        selected[22:29, 30:38] = True
        assert np.sqrt(np.mean((source[selected].astype(float) - clean[selected]) ** 2)) > 100
        for key in ("fsr_fast", "fsr_best", "shift_map"):
            block = graph.blocks[key]
            result = block.last_result
            assert result and result.passed, block.error
            output = result.outputs["image"]
            rms = np.sqrt(np.mean((output[selected].astype(float) - clean[selected]) ** 2))
            assert rms < 2
            np.testing.assert_array_equal(output[~selected], source[~selected])
            np.testing.assert_array_equal(result.outputs["repair_mask"] != 0, selected)
    finally:
        graph.close()
