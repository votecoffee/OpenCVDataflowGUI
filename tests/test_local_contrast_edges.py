"""Local illumination subtraction and source-grid edge continuity."""

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

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/local_contrast_edges"
TYPE = "cv.local_contrast_edges"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(("mode", "border_name", "border", "short", "long"), (
    ("box", "reflect_101", cv2.BORDER_REFLECT_101, 1, 17),
    ("gaussian", "reflect_edge", cv2.BORDER_REFLECT, 5, 21),
    ("box", "replicate", cv2.BORDER_REPLICATE, 3, 35),
))
def test_controls_match_native_filter_and_derivative_pipeline(registry, mode, border_name, border, short, long):
    rng = np.random.default_rng(86)
    rgb = rng.integers(5, 250, (53, 78, 3), np.uint16) * np.uint16(257)
    block = registry.create(TYPE, parameters={
        "background": mode, "border": border_name, "short_range": short, "long_range": long,
        "low": 1.5, "high": 4.0,
    })
    result = block.execute({"image": ImageArray(rgb, "rgb")})
    assert result and result.passed, block.error
    mono = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) * np.float32(1 / 257)
    fine = cv2.GaussianBlur(mono, (short, short), 0, borderType=border)
    background = (cv2.blur(mono, (long, long), borderType=border) if mode == "box"
                  else cv2.GaussianBlur(mono, (long, long), 0, borderType=border))
    contrast = fine - background
    dx, dy = (cv2.Sobel(contrast, cv2.CV_32F, x, y, ksize=3, scale=0.125, borderType=border)
              for x, y in [(1, 0), (0, 1)])
    qx, qy = (np.rint(32 * component).astype(np.int16) for component in (dx, dy))
    edge_reference = cv2.Canny(qx, qy, 48.0, 128.0, L2gradient=True)
    np.testing.assert_array_equal(result.outputs["contrast"], contrast)
    np.testing.assert_allclose(result.outputs["strength"], np.hypot(dx, dy), rtol=3e-7, atol=1e-7)
    np.testing.assert_array_equal(result.outputs["image"], edge_reference)


def test_mask_and_region_intersection_preserve_source_grid_and_local_neighborhoods(registry):
    mono = cv2.imread(str(EXAMPLE / "assets/uneven-light.png"), cv2.IMREAD_GRAYSCALE)
    payload = {
        "spatial": frame([256, 192], "fixture"),
        "ticket": TicketClock("local-edges").issue("camera", "fixed", "part-86"),
        "pyramid": {"level": 1, "largest_size": [512, 384], "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]},
    }
    source = ValuePacket(mono.astype(np.float64) / 255, payload)
    mask = np.zeros(mono.shape, np.uint8)
    mask[27:150, 24:129] = 255
    region = {
        "model": "image_region_v1", "image_size": [256, 192], "shape": "rectangle",
        "rectangle": [37, 33, 95, 99], "coordinate_context": payload["spatial"],
    }
    selected = np.zeros(mono.shape, bool)
    selected[33:132, 37:129] = True
    block = registry.create(TYPE, parameters={"low": 0.5, "high": 0.9})
    full = block.execute({"image": source})
    assert full and full.passed, block.error
    result = block.execute({"image": source, "mask": ValuePacket(mask, payload), "region": region})
    assert result and result.passed, block.error
    assert result.outputs["report"]["selected_pixels"] == int(selected.sum())
    assert result.outputs["report"]["edge_pixels"] == cv2.countNonZero(result.outputs["image"])
    for output in ("image", "contrast", "strength"):
        restricted = result.outputs[output]
        assert result.payloads[output]["spatial"] == payload["spatial"]
        assert result.payloads[output]["ticket"] == payload["ticket"]
        assert result.payloads[output]["pyramid"] == payload["pyramid"]
        assert not restricted[~selected].any()
        np.testing.assert_allclose(restricted[selected], full.outputs[output][selected], rtol=2e-7)
    wrong = ValuePacket(mask, {"spatial": frame([256, 192], "different-grid")})
    assert block.execute({"image": source, "mask": wrong}) is None
    assert "coordinate frame" in block.error


def test_uniform_image_and_threshold_toggle(registry):
    block = registry.create(TYPE)
    result = block.execute({"image": np.full((27, 34), 127, np.uint8)})
    assert result and result.passed, block.error
    assert not any(result.outputs[key].any() for key in ("image", "contrast", "strength"))
    image = cv2.imread(str(EXAMPLE / "assets/uneven-light.png"), cv2.IMREAD_GRAYSCALE)
    low = block.execute({"image": image})
    high = registry.create(TYPE, parameters={"high": 100.0}).execute({"image": image})
    assert low.outputs["image"].any() and not high.outputs["image"].any()
    np.testing.assert_allclose(low.outputs["strength"], high.outputs["strength"], rtol=2e-7)


@pytest.mark.parametrize(("settings", "error"), (
    ({"short_range": 4}, "odd"),
    ({"short_range": 25}, "smaller"),
    ({"low": 2.0}, "low <= high"),
    ({"high": 5000}, "between 0"),
))
def test_invalid_ranges_and_thresholds_fail_with_field_errors(registry, settings, error):
    block = registry.create(TYPE, parameters=settings)
    assert block.execute({"image": np.zeros((23, 31), np.uint8)}) is None
    assert error in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_uneven_lighting_fixture_localizes_two_continuous_boundaries(registry, profile):
    graph = Graph.load(EXAMPLE / "compare_uneven_lighting.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(3)
        graph.run()
        local = graph.blocks["local"]
        assert block_entry(local).path[:2] == ("OpenCV", "Image Processing (imgproc)")
        result = local.last_result
        assert result and result.passed, local.error
        precision = graph.blocks["local_check"].last_result.outputs
        baseline = graph.blocks["baseline_check"].last_result.outputs
        assert precision["precision"] >= 0.99 and precision["recall"] >= 0.99
        assert baseline["precision"] < 0.20 and baseline["recall"] >= 0.98
        mask = graph.blocks["selection"].last_result.outputs["mask"] != 0
        edges = result.outputs["image"]
        assert not edges[~mask].any()
        count, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        assert count == 3
        assert sorted(stats[1:, cv2.CC_STAT_HEIGHT].tolist()) == [87, 110]
        for x in (52, 107):
            assert np.all(np.any(edges[45:151, x - 1:x + 2] != 0, axis=1))
    finally:
        graph.close()
