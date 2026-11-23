"""Qualify native rolling guidance and fixed zero-border iteration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.rolling_guidance_filter"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/rolling_guidance_filter"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


@pytest.mark.parametrize(
    ("iteration_count", "border", "native_border"),
    ((1, "replicate", cv2.BORDER_REPLICATE), (5, "reflect_edge", cv2.BORDER_REFLECT)),
)
def test_normalized_working_image_matches_native_and_keeps_alpha(
    iteration_count, border, native_border
):
    rng = np.random.default_rng(8329)
    original = rng.integers(8400, 54000, (27, 45, 4), dtype=np.uint16)
    source = ImageArray(original, "rgba")
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[5:25, 6:29] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [45, 27],
        "shape": "rectangle",
        "rectangle": [10, 2, 23, 18],
    }
    node = tool(
        diameter=0, sigma_color=0.13, sigma_space=2.8, iterations=iteration_count, border=border
    )
    result = node.execute({"image": source, "mask": mask, "region": region})
    assert result and result.passed, node.error
    output = result.outputs["image"]
    native = cv2.ximgproc.rollingGuidanceFilter(
        (original[..., :3] / 65535).astype(np.float32),
        d=0,
        sigmaColor=0.13,
        sigmaSpace=2.8,
        numOfIter=iteration_count,
        borderType=native_border,
    )
    expected = np.rint(np.clip(native.astype(float), 0, 1) * 65535).astype(np.uint16)
    selected = np.zeros_like(mask, dtype=bool)
    selected[5:20, 10:29] = True
    assert output.dtype == np.uint16 and image_format(output) == "rgba"
    np.testing.assert_array_equal(output[..., :3][selected], expected[selected])
    np.testing.assert_array_equal(output[~selected], original[~selected])
    np.testing.assert_array_equal(output[..., 3], original[..., 3])


def test_constant_border_keeps_original_signal_and_zeros_each_iteration():
    source = np.array(
        [[0.86, 0.88, 0.89, 0.91], [0.87, 0.94, 0.92, 0.93], [0.95, 0.96, 0.89, 0.88]], np.float32
    )
    radius, color, space, iterations = 2, 0.86, 1.5, 4
    node = tool(
        diameter=5, sigma_color=color, sigma_space=space, iterations=iterations, border="constant"
    )
    result = node.execute({"image": source})
    assert result and result.passed, node.error

    signal = np.pad(source.astype(float), radius)
    offsets = np.array([(y, x) for y in range(-2, 3) for x in range(-2, 3) if x * x + y * y <= 4])
    spatial = (offsets**2).sum(axis=1) / space**2
    guide = source.astype(float)
    for _ in range(iterations):
        g = np.pad(guide, radius)
        for iy, ix in np.ndindex(guide.shape):
            y, x = iy + radius, ix + radius
            ly, lx = y + offsets[:, 0], x + offsets[:, 1]
            distance = (g[ly, lx] - g[y, x]) ** 2 / color**2
            weights = np.exp(-0.5 * (spatial + distance))
            guide[iy, ix] = np.average(signal[ly, lx], weights=weights)
    filtered = result.outputs["image"]
    np.testing.assert_allclose(filtered, guide, rtol=0, atol=3e-6)
    assert float(filtered[0, 0]) < 0.78


def test_zero_passes_keep_source_coordinates_and_support_requested_depth():
    sample = np.arange(60, dtype=np.uint8).reshape(6, 10)
    ticket = TicketClock("rolling").issue("camera", "fixed", "part-228")
    spatial = frame([10, 6], "current-frame")
    source = ValuePacket(sample, {"spatial": spatial, "ticket": ticket})
    node = tool(iterations=0, output_depth="float32")
    result = node.execute({"image": source})
    assert result and result.passed, node.error
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_array_equal(result.outputs["image"], (sample / 255).astype(np.float32))
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == spatial


def test_invalid_auto_window_and_source_range_report_validation():
    image = np.ones((3, 8), np.float32)
    node = tool(diameter=0, sigma_space=85)
    assert node.execute({"image": image}) is None
    assert "Automatic neighborhood" in node.error
    node = tool()
    image[0, 0] = 3e38
    assert node.execute({"image": image}) is None
    assert node.error_kind == "invalid" and "Source values" in node.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_reduces_texture_and_preserves_large_edges(profile):
    graph = Graph.load(EXAMPLE / "rolling_guidance_filter.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        keys = ("textured", "single", "rolling", "selected")
        results = {key: graph.blocks[key].last_result for key in keys}
        assert all(result and result.passed for result in results.values())
        images = {key: result.outputs["image"] for key, result in results.items()}
        std = {key: image[38:108, 63:84].std() for key, image in images.items()}
        assert std["rolling"] < 0.3 * std["single"]
        assert std["rolling"] < 0.1 * std["textured"]
        im = images["rolling"].astype(float)
        assert np.mean(im[48:103, 53] - im[48:103, 52]) == pytest.approx(118, abs=6)
        cutout = im[73:81, 131:135].mean() - im[73:81, 124:127].mean()
        assert cutout == pytest.approx(93, abs=6)
        write = np.zeros(im.shape, dtype=bool)
        write[28:112, 20:196] = True
        np.testing.assert_array_equal(images["selected"][write], images["rolling"][write])
        np.testing.assert_array_equal(images["selected"][~write], images["textured"][~write])
    finally:
        graph.close()
