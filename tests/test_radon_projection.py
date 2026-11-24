"""Native Radon precision, angle and distance mapping, and shipped graphs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.radon_projection import projection_peaks
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/radon_projection"
TYPE = "inspect.radon_projection"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(("shape", "dtype", "crop", "start", "end", "step"), (
    ((25, 25), np.uint8, False, 0, 90, 2.5),
    ((16, 25), np.uint8, True, 90, 61, 4),
    ((23, 16), np.float32, True, -50, 40, 12),
    ((22, 27), np.float64, False, -13, 32, 0.7),
    ((18, 29), np.uint16, False, 60, 25, 8),
))
def test_raw_native_parity_without_mutating_wired_image(registry, shape, dtype, crop, start, end, step):
    rng = np.random.default_rng(87)
    image = rng.integers(0, 240, shape).astype(dtype)
    if image.dtype.kind == "f":
        image /= dtype(307)
    signed_step = step if end > start else -step
    copied = image.copy()
    native = copied.astype(np.float64) if dtype == np.uint16 else copied.copy()
    expected = cv2.ximgproc.RadonTransform(
        native, theta=signed_step, start_angle=start, end_angle=end, crop=crop, norm=False,
    )
    options = {"start_angle": start, "end_angle": end, "angle_step": step, "circle_crop": crop}
    block = registry.create(TYPE, parameters={**options, "maximum_peaks": 0})
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(image, copied)
    np.testing.assert_array_equal(result.outputs["image"], copied)
    np.testing.assert_array_equal(result.outputs["projection"], expected)
    assert result.outputs["projection"].dtype == (np.int32 if dtype == np.uint8 else np.float64)
    assert result.outputs["preview"].dtype == np.uint8
    expected_preview = cv2.normalize(expected, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    np.testing.assert_array_equal(result.outputs["preview"], expected_preview)
    axes = result.outputs["axes"]
    np.testing.assert_allclose(axes["angle_degrees"], start + signed_step*np.arange(expected.shape[1]))
    assert axes["distance_count"] == expected.shape[0]
    assert axes["distance_start_pixels"] == -(expected.shape[0] // 2)
    assert result.outputs["peaks"]["items"] == []


def test_impulse_angles_follow_source_normal_and_signed_distance(registry):
    height, width = 27, 44
    source = np.zeros((height, width), np.float64)
    x, y = 17, 12
    source[y, x] = 22.125
    block = registry.create(TYPE, parameters={"start_angle": 0.0, "end_angle": 180, "angle_step": 90})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    axes = result.outputs["axes"]
    ox, oy = axes["origin"]
    assert axes["angle_degrees"] == [0, 90]
    projection = result.outputs["projection"]
    assert list(np.argmax(projection, axis=0) + axes["distance_start_pixels"]) == [x - ox, oy - y]
    np.testing.assert_array_equal(projection.max(axis=0), [22.125, 22.125])


def test_negative_peak_keeps_full_precision_and_selected_line_span(registry):
    scene = np.zeros((42, 39), np.float64)
    scene[:, 21] = -0.125
    scene[:, 6] = -100
    spatial = frame([39, 42], "measurement-camera")
    ticket = TicketClock("radon").issue("camera", "fixed", "part-87")
    meta = {"spatial": spatial, "ticket": ticket}
    partial = np.zeros(scene.shape, np.uint8)
    partial[3:15, 18:28] = 255
    partial[21:39, 18:28] = 255
    region = {
        "model": "image_region_v1", "image_size": [39, 42], "shape": "rectangle",
        "rectangle": [13, 1, 24, 39], "coordinate_context": spatial,
    }
    settings = {"end_angle": 1, "peak_polarity": "negative", "minimum_strength": 3.0}
    block = registry.create(TYPE, parameters=settings)
    inputs = {
        "image": ValuePacket(scene, meta), "mask": ValuePacket(partial, meta), "region": region,
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    peak, = result.outputs["peaks"]["items"]
    assert peak["projection"] == -0.125 * (12 + 18)
    assert peak["strength"] == 3.75 and peak["line_angle_degrees"] == 90
    assert peak["segment"] == [[21, 21], [21, 38]]
    assert result.outputs["collection"]["context"]["spatial"] == spatial
    assert result.payloads["image"]["spatial"] == spatial
    assert result.payloads["projection"]["ticket"] == ticket
    assert "spatial" not in result.payloads["projection"]
    assert result.payloads["projection"]["radon_axes"] == result.outputs["axes"]
    assert result.drawings["projection lines"].source == "image"

    positive = registry.create(TYPE, parameters={**settings, "peak_polarity": "positive"}).execute(inputs)
    assert positive.outputs["peaks"]["items"] == []
    invalid = ValuePacket(partial, {"spatial": frame([39, 42], "other-camera")})
    assert block.execute({**inputs, "mask": invalid}) is None
    assert "coordinate frame" in block.error


def test_peak_suppression_wraps_line_angle_and_flips_distance(registry):
    raw = np.zeros((21, 3), np.float64)
    raw[13, 0], raw[7, 2], raw[16, 1] = 22.0, 21.0, 19.0
    angles = np.array([0.0, 60.0, 179.0])
    options = {**registry.create(TYPE).parameters, "maximum_peaks": 2, "angle_separation": 2}
    peaks, _ = projection_peaks(raw, angles, (10, 10), (21, 21), None, options)
    assert [(item["normal_angle_degrees"], item["distance_pixels"]) for item in peaks] == [(0, 3), (60, 6)]


@pytest.mark.parametrize("settings,error", (
    ({"start_angle": 6.0, "end_angle": 6.0}, "projection bins"),
    ({"angle_step": 0.01}, "projection bins"),
    ({"circle_crop": 1}, "boolean"),
    ({"angle_step": 0.2}, "rotation budget"),
))
def test_validation_rejects_empty_ranges_and_bounds_native_work(registry, monkeypatch, settings, error):
    monkeypatch.setattr(cv2.ximgproc, "RadonTransform", lambda *a, **k: pytest.fail("entered native Radon"))
    block = registry.create(TYPE, parameters=settings)
    assert block.execute({"image": np.zeros((420, 400), np.uint8)}) is None
    assert error in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_stripes_and_slanted_edge_recover_source_coordinates(registry, profile):
    for filename, expected, shape in (
        ("parallel_stripes", [(38, 0), (38, -22), (38, 22)], (282, 180)),
        ("tilted_edge", [(123, 9)], (179, 120)),
    ):
        graph = Graph.load(EXAMPLE / f"{filename}.json", registry)
        try:
            graph.set_run_profile(profile)
            graph.set_branch_workers(3)
            graph.run()
            radon = graph.blocks["radon"]
            assert block_entry(radon).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
            result = radon.last_result
            assert result and result.passed, radon.error
            assert result.outputs["projection"].shape == shape
            axes = result.outputs["axes"]
            peaks = result.outputs["peaks"]["items"]
            assert [(item["normal_angle_degrees"], item["distance_pixels"]) for item in peaks] == expected
            for item in peaks:
                points = np.asarray(item["segment"])
                np.testing.assert_allclose(
                    (points - axes["origin"]) @ item["normal_xy"], item["distance_pixels"], atol=1e-10,
                )
                assert np.all((points >= [28.5, 15.5]) & (points <= [187.5, 162.5]))
            assert bool(result.drawings) == (profile == "debug")
        finally:
            graph.close()
