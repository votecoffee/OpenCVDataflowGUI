"""Check exposure fusion, illumination equations and source-grid preservation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.low_light import EnhanceLowLight, _native_function, illumination_system
from opencv_dataflow.payloads import ValuePacket, frame

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/low_light_enhancement"


@pytest.fixture
def chart():
    return cv2.imread(str(EXAMPLE / "assets/shadow_chart.png"), cv2.IMREAD_COLOR)


def chart_measurements(image):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    dark = gray[29:52, 19:36]
    contrast = float(gray[29:52, 49:66].mean() - dark.mean())
    noise = float(dark.std())
    saturated = float(np.mean(np.max(image, axis=2) == 255))
    return contrast, noise, saturated


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_example_improves_shadow_contrast_and_tracks_noise(profile, workers, chart):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.toolbox import block_entry

    graph = Graph.load(EXAMPLE / "low_light_enhancement.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        before_contrast, before_noise, before_clip = chart_measurements(chart)
        gamma = graph.blocks["gamma"].last_result.outputs["image"]
        gamma_contrast, gamma_noise, gamma_clip = chart_measurements(gamma)
        assert gamma_contrast > before_contrast and gamma_noise > before_noise
        assert gamma_clip >= before_clip
        for key in ("automatic", "explicit"):
            block = graph.blocks[key]
            assert block_entry(block).path == (
                "OpenCV", "Intensity Transform (intensity_transform)", "Contrast",
            )
            result = block.last_result
            assert result and result.passed, block.error
            image, report = result.outputs["image"], result.outputs["report"]
            contrast, noise, clipped = chart_measurements(image)
            assert contrast > 1.5 * before_contrast
            assert before_noise < noise < 4 * before_noise
            assert clipped < 0.15
            assert clipped == pytest.approx(report["upper_saturation_after"])
            assert report["mean_after"] > report["mean_before"]
            assert result.payloads["image"]["spatial"]["root_id"] == "source:image"
            assert result.payloads["image"]["ticket"]["value"] == "shadow_chart.png"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_sparse_illumination_system_matches_independent_grid_equations():
    light = np.array([[.1, .12, .20, .3], [.11, .13, .16, .4], [.4, .32, .19, .4]])
    height, width = light.shape
    expected = np.eye(light.size)
    for dr, dc in ((1, 0), (0, 1)):
        differences = np.empty(light.shape)
        for y, x in np.ndindex(light.shape):
            differences[y, x] = light[(y + dr) % height, (x + dc) % width] - light[y, x]
        for y, x in np.ndindex(light.shape):
            total = sum(
                differences[y + dy * dr, x + dy * dc] for dy in range(-2, 3)
                if 0 <= y + dy * dr < height and 0 <= x + dy * dc < width
            )
            a, b = y * width + x, ((y + dr) % height) * width + (x + dc) % width
            weight = .5 / (abs(total * differences[y, x]) + .001)
            expected[a, a] += weight
            expected[b, b] += weight
            expected[a, b] -= weight
            expected[b, a] -= weight
    actual = illumination_system(light).toarray()
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(actual @ np.ones(light.size), np.ones(light.size), atol=1e-10)


def test_preserves_high_depth_alpha_and_only_writes_the_image_selection(chart):
    source = np.empty((*chart.shape[:2], 4), np.uint16)
    source[..., :3] = chart[..., ::-1].astype(np.uint16) * 257
    source[..., 3] = np.arange(176, dtype=np.uint16)[None, :] * 337
    image = ImageArray(source, "rgba")
    mask = np.zeros(chart.shape[:2], np.uint8)
    mask[14:90, 14:62] = 255
    region = {
        "model": "image_region_v1", "image_size": [176, 120],
        "shape": "rectangle", "rectangle": [20, 20, 44, 25],
    }
    selected = np.zeros(mask.shape, bool)
    selected[20:45, 20:62] = True
    block = EnhanceLowLight(parameters={"exposure_mode": "explicit"})
    result = block.execute({
        "image": ValuePacket(image, {"spatial": frame([176, 120], "chart")}),
        "mask": mask, "region": region,
    })
    assert result and result.passed, block.error
    enhanced = result.outputs["image"]
    assert enhanced.dtype == np.uint16 and enhanced.color_format == "rgba"
    assert result.payloads["image"]["spatial"]["root_id"] == "chart"
    report = result.outputs["report"]
    assert report["backend"] == "scipy" and report["exposure_ratio"] == 4
    assert report["selected_pixels"] == selected.sum()
    assert report["illumination_residual"] <= 1e-5
    np.testing.assert_array_equal(enhanced[~selected], source[~selected])
    np.testing.assert_array_equal(enhanced[..., 3], source[..., 3])
    assert enhanced[selected, :3].mean() > source[selected, :3].mean()


def test_flat_frames_and_zero_enhancement_keep_grayscale_values():
    source = np.full((35, 41), 0.08, np.float32)
    block = EnhanceLowLight()
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    assert result.outputs["image"].dtype == np.float32
    assert result.outputs["report"]["exposure_ratio"] == 1.0
    np.testing.assert_array_equal(result.outputs["image"], source)
    source = np.linspace(0, 0.95, source.size, dtype=np.float32).reshape(source.shape)
    block = EnhanceLowLight(parameters={"exposure_mode": "explicit", "enhancement_ratio": 0})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], source)


@pytest.mark.parametrize("automatic", [True, False])
def test_native_binding_parity_on_eigen_build(automatic):
    method = _native_function(automatic)
    if method is None:
        pytest.skip("Native BIMEF requires an Eigen-enabled OpenCV build")
    y, x = np.indices((45, 63))
    signal = np.uint8((5 * x + 7 * y) % 55 + 12)
    rgb = np.stack((signal, signal + 2, signal + 4), axis=2)
    block = EnhanceLowLight(parameters={
        "backend": "opencv", "exposure_mode": "automatic" if automatic else "explicit",
        "exposure_ratio": 3.5, "enhancement_ratio": .7, "response_a": -.31, "response_b": 1.17,
    })
    result = block.execute({"image": ImageArray(rgb, "rgb")})
    assert result and result.passed, block.error
    expected = method(rgb, mu=.7, a=-.31, b=1.17) if automatic else method(rgb, 3.5, .7, -.31, 1.17)
    np.testing.assert_array_equal(expected, result.outputs["image"])
    assert result.outputs["report"]["backend"] == "opencv"
    assert result.outputs["report"]["exposure_ratio"] == (None if automatic else 3.5)


def test_native_unavailability_and_invalid_float_report_errors(monkeypatch, chart):
    monkeypatch.setattr(cv2, "intensity_transform", None)
    block = EnhanceLowLight(parameters={"backend": "opencv"})
    assert block.execute({"image": chart}) is None
    assert "Eigen-enabled" in block.error
    block = EnhanceLowLight()
    assert block.execute({"image": np.full((8, 9), 1.5, np.float32)}) is None
    assert "samples must be in" in block.error
