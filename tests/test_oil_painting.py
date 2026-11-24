"""Qualify OpenCV modal filtering, selection halos, and the saved comparison."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.oil_painting import OilPaintingFilter
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/oil_painting"


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_comparison_retains_a_sharper_boundary(profile, workers):
    graph = Graph.load(EXAMPLE / "oil_painting.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["smooth"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Computational Photography (xphoto)")
        results = {}
        for key in ("source", "gaussian", "smooth", "selected", "filmstrip"):
            node = graph.blocks[key]
            results[key] = node.last_result
            assert node.last_result and node.last_result.passed, node.error
        source, gaussian, modal, selected = (
            results[key].outputs["image"] for key in ("source", "gaussian", "smooth", "selected")
        )
        grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
                 for image in (source, gaussian, modal)]
        assert 9.7 < grays[0][40:70, 65:88].std() < 9.9
        assert grays[1][40:70, 65:88].std() == grays[2][40:70, 65:88].std() == 0
        edge_slopes = [np.diff(gray[32:73, 52:61].mean(axis=0)) for gray in grays]
        assert edge_slopes[1].max() == 18 and edge_slopes[2].max() == 86
        assert int(edge_slopes[2].argmax()) == int(edge_slopes[0].argmax()) - 1

        region = np.zeros(source.shape[:2], bool)
        region[24:93, 23:118] = True
        np.testing.assert_array_equal(selected[region], modal[region])
        np.testing.assert_array_equal(selected[~region], source[~region])
        assert results["selected"].payloads["image"]["spatial"]["root_id"] == "source:image"
        assert results["selected"].payloads["image"]["ticket"]["value"] == "textured.png"
    finally:
        graph.close()


@pytest.mark.parametrize(("plane", "code"), [
    ("grayscale", cv2.COLOR_BGR2GRAY),
    ("lab_lightness", cv2.COLOR_BGR2Lab),
    ("hsv_hue", cv2.COLOR_BGR2HSV),
])
@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba"])
def test_histogram_modes_with_channel_order_and_alpha(plane, code, encoding):
    bgr = np.random.default_rng(375).integers(0, 256, (13, 16, 3), dtype=np.uint8)
    colors = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if encoding == "gray" else bgr
    if encoding == "gray" or plane == "grayscale":
        expected = cv2.xphoto.oilPainting(colors, 3, 9)
    else:
        first = cv2.cvtColor(bgr, code)[..., 0]
        bins = np.rint(first / 9).astype(np.int64)
        expected = np.empty_like(bgr)
        for y, x in np.ndindex(bgr.shape[:2]):
            window = np.s_[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
            neighborhood = bins[window]
            mode = np.bincount(neighborhood.ravel()).argmax()
            expected[y, x] = np.rint(bgr[window][neighborhood == mode].mean(axis=0))
    if encoding in ("rgb", "rgba"):
        colors, expected = colors[..., ::-1], expected[..., ::-1]
    if encoding in ("bgra", "rgba"):
        alpha = np.broadcast_to(np.arange(16, dtype=np.uint8)[None, :], (13, 16))
        colors, expected = np.dstack((colors, alpha)), np.dstack((expected, alpha))
    image = ImageArray(colors, encoding)
    before = image.copy()
    block = OilPaintingFilter(parameters={"histogram_plane": plane, "radius": 3, "dynamic_ratio": 9})
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint8 and output.color_format == encoding
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(image, before)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[6:10, 8:15] = 255
    selected = block.execute({"image": image, "mask": mask})
    assert selected and selected.passed, block.error
    np.testing.assert_array_equal(selected.outputs["image"][mask != 0], expected[mask != 0])
    np.testing.assert_array_equal(selected.outputs["image"][mask == 0], image[mask == 0])


def test_mask_and_circle_intersect_and_use_neighbors_across_the_selection_boundary():
    y, x = np.indices((28, 42))
    image = np.random.default_rng(72).integers(0, 256, x.shape, dtype=np.uint8)
    mask = np.where(x + y < 19, 255, 0).astype(np.uint8)
    spatial = frame([42, 28], "camera")
    region = {
        "model": "image_region_v1", "image_size": [42, 28], "shape": "circle",
        "center": [7, 7], "radius": 9, "coordinate_context": spatial,
    }
    block = OilPaintingFilter()
    result = block.execute({
        "image": ValuePacket(image, {"spatial": spatial}),
        "mask": ValuePacket(mask, {"spatial": spatial}),
        "region": region,
    })
    assert result and result.passed, block.error
    expected = cv2.xphoto.oilPainting(image, 4, 12)  # Native whole image, including the top edge.
    keep = ((x - 7)**2 + (y - 7)**2 <= 9**2) & (mask != 0)
    np.testing.assert_array_equal(result.outputs["image"], np.where(keep, expected, image))
    assert result.payloads["image"]["spatial"] == spatial
    shifted = ValuePacket(mask, {"spatial": frame([42, 28], "different_camera")})
    assert block.execute({"image": ValuePacket(image, {"spatial": spatial}), "mask": shifted}) is None
    assert "coordinate frame differs" in block.error


def test_empty_selection_does_not_call_native(monkeypatch):
    def unexpected(*_args):
        pytest.fail("Empty selection reached OpenCV")

    monkeypatch.setattr(cv2.xphoto, "oilPainting", unexpected)
    image = np.full((12, 18, 4), 112, np.uint8)
    block = OilPaintingFilter()
    result = block.execute({"image": image, "mask": np.zeros(image.shape[:2], np.uint8)})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], image)


@pytest.mark.parametrize(("value", "parameters", "error"), [
    (np.zeros((4, 4), np.uint16), {}, "Unsupported image depth"),
    (ImageArray(np.zeros((4, 4, 3), np.uint8), "hsv"), {}, "gray or BGR/RGB"),
    (ImageArray(np.zeros((4, 4, 3), np.uint8), "rgba"), {}, "gray or BGR/RGB"),
    (np.zeros((4, 4), np.uint8), {"dynamic_ratio": 128}, "between 1 and 127"),
    (np.zeros((4, 4), np.uint8), {"radius": 0}, "between 1 and 127"),
    (np.zeros((1100, 1100), np.uint8), {"radius": 127}, "smaller region"),
])
def test_invalid_settings_are_reported(value, parameters, error):
    tool = OilPaintingFilter(parameters=parameters)
    assert tool.execute({"image": value}) is None
    assert error in tool.error
