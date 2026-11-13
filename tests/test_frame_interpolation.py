"""Known-motion truth, support failures, media lineage and native Farneback integration."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.frame_interpolation import InterpolateFrames, inverse_map
from opencv_dataflow.images import ImageArray
from opencv_dataflow.opencv_reference import opencv_reference
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.sequence_tools import DenseFlow
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/frame_interpolation"


def pair(dtype=np.uint8, channels=1):
    source = np.random.default_rng(130).integers(20, 230, (72, 96), dtype=np.uint8)
    source = cv2.GaussianBlur(source, (5, 5), 1)
    if dtype == np.uint16:
        source = source.astype(dtype) * 257
    elif np.issubdtype(dtype, np.floating):
        source = source.astype(dtype) / 255
    if channels > 1:
        source = np.repeat(source[..., None], channels, axis=2)
    moved = cv2.warpAffine(source, np.float64([[1, 0, 8], [0, 1, 4]]), (96, 72))
    forward = np.full((72, 96, 2), [8, 4], np.float32)
    return source, moved, forward, -forward


def run(a, b, extra=None, **parameters):
    block = builtin_registry().create("cv.interpolate_frames", parameters=parameters)
    result = block.execute({"image_a": a, "image_b": b, **(extra or {})})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize(
    "dtype,channels", [(np.uint8, 1), (np.uint16, 3), (np.float32, 4), (np.float64, 1)]
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_exact_motion_fields_reproduce_independent_intermediate_truth(dtype, channels, fraction):
    a, b, forward, backward = pair(dtype, channels)
    original = a.copy()
    _, result = run(
        a,
        b,
        {"forward": forward, "backward": backward},
        flow_mode="provided",
        fraction=fraction,
        emit_flows=True,
    )
    truth = cv2.warpAffine(a, np.float64([[1, 0, 8 * fraction], [0, 1, 4 * fraction]]), (96, 72))
    supported = result.outputs["coverage"] != 0
    assert supported.mean() > 0.8
    np.testing.assert_array_equal(result.outputs["image"][supported], truth[supported])
    np.testing.assert_array_equal(a, original)
    assert result.outputs["image"].dtype == dtype
    np.testing.assert_array_equal(result.outputs["forward_flow"].vectors, forward)
    np.testing.assert_array_equal(result.outputs["backward_flow"].vectors, backward)


@pytest.mark.parametrize(
    "kernel,margin", [("nearest", 0), ("linear", 0), ("cubic", 2), ("lanczos4", 4)]
)
def test_sampling_kernels_have_conservative_source_footprints(kernel, margin):
    a, b, forward, backward = pair()
    _, result = run(
        a, b, {"forward": forward, "backward": backward}, flow_mode="provided", interpolation=kernel
    )
    support = result.outputs["coverage"] != 0
    assert support[36, 48]
    assert not support[:, :4].any() and not support[-2:].any()
    if margin:
        yy, xx = np.mgrid[:72, :96]
        for name, sx, sy in (("coverage_a", xx - 4, yy - 2), ("coverage_b", xx + 4, yy + 2)):
            valid = result.outputs[name] != 0
            assert np.all((sx[valid] >= margin) & (sx[valid] <= 95 - margin))
            assert np.all((sy[valid] >= margin) & (sy[valid] <= 71 - margin))
    truth = cv2.warpAffine(a, np.float64([[1, 0, 4], [0, 1, 2]]), (96, 72))
    np.testing.assert_array_equal(result.outputs["image"][support], truth[support])


@pytest.mark.parametrize("fraction", [0, 1])
def test_endpoints_copy_exact_pixels_without_estimating_motion(monkeypatch, fraction):
    def forbidden(*args, **kwargs):
        pytest.fail("An exact endpoint must bypass optical flow")

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", forbidden)
    a, b, _, _ = pair(np.uint16, 4)
    _, result = run(a, b, fraction=fraction, flow_mode="provided")
    np.testing.assert_array_equal(result.outputs["image"], a if fraction == 0 else b)
    assert np.all(result.outputs["coverage"] == 255)
    assert not result.outputs["info"]["motion_evaluated"]
    assert result.outputs["forward_flow"] is NO_VALUE


@pytest.mark.parametrize("fill", ["zero", "nearest_endpoint", "crossfade"])
def test_bad_flow_is_not_disguised_by_hole_filling(fill):
    a = np.full((32, 40), 40, np.uint8)
    b = np.full_like(a, 200)
    forward = np.full((32, 40, 2), [8, 0], np.float32)
    backward = np.zeros_like(forward)
    _, result = run(
        a,
        b,
        {"forward": forward, "backward": backward},
        flow_mode="provided",
        uncovered=fill,
        minimum_coverage=0.5,
    )
    assert not result.passed
    assert any(
        reason["code"] == "insufficient_interpolation_coverage" for reason in result.fail_reasons
    )
    assert np.all(result.outputs["coverage"] == 0)
    expected = {"zero": 0, "nearest_endpoint": 40, "crossfade": 120}[fill]
    assert np.all(result.outputs["image"] == expected)


def test_known_occlusion_proxy_and_missing_support_are_reported():
    a = np.full((40, 48), 64, np.uint8)
    b = a.copy()
    b[12:28, 16:32] = 240
    field = np.zeros((40, 48, 2), np.float32)
    _, unchecked = run(a, b, {"forward": field, "backward": field}, flow_mode="provided")
    assert np.all(unchecked.outputs["coverage"] == 255)
    support = np.full(a.shape, 255, np.uint8)
    support[4:8, 4:10] = 0
    _, checked = run(
        a,
        b,
        {"forward": field, "backward": field, "support_a": support},
        flow_mode="provided",
        appearance_tolerance=0.1,
    )
    expected = np.full(a.shape, 255, np.uint8)
    expected[12:28, 16:32] = 0
    expected[4:8, 4:10] = 0
    np.testing.assert_array_equal(checked.outputs["coverage"], expected)
    assert checked.outputs["info"]["uncovered_pixels"] == 16 * 16 + 4 * 6
    assert np.all(checked.outputs["uncertain"][12:28, 16:32] == 255)


def test_nonconvergent_inverse_maps_do_not_claim_support():
    y, x = np.mgrid[:20, :24].astype(np.float32)
    # q + .5*(4*q) = x has a solution, but this fixed-point iteration diverges.
    flow = np.stack((4 * x, np.zeros_like(x)), axis=2)
    p = {**InterpolateFrames.defaults, "inverse_iterations": 3, "inverse_tolerance": 0.01}
    _, _, support = inverse_map(flow, 0.5, np.full(x.shape, 255, np.uint8), x, y, p)
    assert not support[:, 4:].any()


def test_output_selection_preserves_a_outside_and_payloads_keep_both_source_tickets():
    a, b, forward, backward = pair(np.uint16, 3)
    a, b = ImageArray(a, "rgb"), ImageArray(b, "rgb")
    clock = TicketClock()
    ticket_a, ticket_b = clock.issue("frame-a"), clock.issue("frame-b")
    grid = frame([96, 72], "camera")
    grid.update(root_size=[192, 144], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    pyramid = {
        "level": 1,
        "largest_size": [192, 144],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    ca = {
        "ticket": ticket_a,
        "spatial": grid,
        "pyramid": pyramid,
        "encoder": 512,
        "video_frame": {
            "filename": "camera.avi",
            "frame_index": 5,
            "timestamp_ms": 100,
            "fps": 25,
            "timestamp_source": "decoder",
        },
    }
    cb = {
        "ticket": ticket_b,
        "spatial": grid,
        "video_frame": {"filename": "camera.avi", "frame_index": 6, "timestamp_ms": 140},
    }
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[8:62, 12:82] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [96, 72],
        "shape": "circle",
        "center": [48, 36],
        "radius": 25,
    }
    block, result = run(
        ValuePacket(a, ca),
        ValuePacket(b, cb),
        {
            "forward": DenseFlow(forward, ca),
            "backward": DenseFlow(backward, cb),
            "mask": mask,
            "region": region,
        },
        flow_mode="provided",
        emit_flows=True,
    )
    selected = np.zeros(mask.shape, np.uint8)
    cv2.circle(selected, (48, 36), 25, 255, -1)
    selected = (selected != 0) & (mask != 0)
    np.testing.assert_array_equal(result.outputs["image"][~selected], a[~selected])
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["timestamp_ms"] == 120
    assert result.outputs["info"]["sources"][1]["ticket"] == ticket_b
    context = result.payloads["image"]
    assert context["ticket"] == ticket_a and context["spatial"] == grid
    assert context["pyramid"] == pyramid and context["encoder"] == 512
    assert context["video_frame"]["timestamp_ms"] == 120
    assert context["video_frame"]["synthetic"]
    assert "frame_index" not in context["video_frame"]
    assert result.drawings["Frame interpolation"].from_output
    block.set_graphics_mode("off")
    lean = block.execute(
        {
            "image_a": ValuePacket(a, ca),
            "image_b": ValuePacket(b, cb),
            "forward": DenseFlow(forward, ca),
            "backward": DenseFlow(backward, cb),
            "mask": mask,
            "region": region,
        }
    )
    assert lean is not None, block.error
    assert not lean.drawings and not lean.graphics
    assert lean.outputs["overlay"] is NO_VALUE and lean.outputs["coverage_preview"] is NO_VALUE
    np.testing.assert_array_equal(lean.outputs["image"], result.outputs["image"])
    assert lean.outputs["info"] == result.outputs["info"]


def test_explicit_times_work_across_independent_image_sources_without_fake_video_identity():
    a, b, forward, backward = pair()
    _, result = run(
        a,
        b,
        {
            "forward": forward,
            "backward": backward,
            "time_a_ms": np.float32(200),
            "time_b_ms": np.int64(240),
        },
        flow_mode="provided",
        timing_policy="required",
        fraction=0.25,
    )
    assert result.outputs["timestamp_ms"] == 210
    assert result.outputs["info"]["timestamp_source"] == "explicit"
    assert "video_frame" not in result.payloads["image"]


def test_unrelated_stream_times_do_not_retain_a_false_acquisition_timestamp():
    a, b, forward, backward = pair()
    ca = {
        "spatial": frame([96, 72], "a"),
        "video_frame": {
            "filename": "same-name.avi",
            "timestamp_ms": 100,
            "frame_index": 5,
        },
    }
    cb = {
        "spatial": frame([96, 72], "b"),
        "video_frame": {
            "filename": "same-name.avi",
            "timestamp_ms": 140,
            "frame_index": 6,
        },
    }
    _, result = run(
        ValuePacket(a, ca),
        ValuePacket(b, cb),
        {"forward": forward, "backward": backward},
        flow_mode="provided",
        alignment="pixel_grid",
    )
    assert result.outputs["timestamp_ms"] is NO_VALUE
    assert "video_frame" not in result.payloads["image"]
    assert result.outputs["info"]["sources"][0]["video_frame"]["timestamp_ms"] == 100


def test_native_documentation_and_configured_time_fields_are_available():
    a, b, forward, backward = pair()
    block, result = run(
        a,
        b,
        {"forward": forward, "backward": backward},
        flow_mode="provided",
        timing_source="explicit",
        time_a_ms=10,
        time_b_ms=30,
    )
    assert result.outputs["timestamp_ms"] == 20
    assert result.outputs["info"]["timestamp_source"] == "explicit_parameters"
    assert "calcOpticalFlowFarneback" in opencv_reference(block)
    assert "remap" in opencv_reference(block)


@pytest.mark.parametrize("mode,gaussian", [("estimate", False), ("refine", True)])
def test_native_farneback_pair_recovers_a_known_motion(mode, gaussian):
    a, b, forward, backward = pair()
    extra = {} if mode == "estimate" else {"forward": forward, "backward": backward}
    _, result = run(
        a,
        b,
        extra,
        flow_mode=mode,
        gaussian_window=gaussian,
        levels=4,
        window=21,
        iterations=5,
        poly_n="7",
        poly_sigma=1.5,
    )
    truth = cv2.warpAffine(a, np.float64([[1, 0, 4], [0, 1, 2]]), (96, 72))
    inside = np.zeros(a.shape, bool)
    inside[16:-16, 20:-20] = True
    use = inside & (result.outputs["coverage"] != 0)
    assert use.sum() > 0.8 * inside.sum()
    error = np.abs(result.outputs["image"].astype(float) - truth)
    crossfade = np.abs((a.astype(float) + b) / 2 - truth)
    assert error[use].mean() < 2
    assert error[use].mean() < crossfade[use].mean() * 0.3


@pytest.mark.parametrize(
    "parameters,extra,message",
    [
        ({"fraction": 1.1}, {}, "between"),
        ({"window": 4}, {}, "odd"),
        ({"poly_n": "9"}, {}, "one of"),
        ({"flow_mode": "provided"}, {}, "both forward"),
        ({"timing_policy": "required"}, {}, "times are required"),
        ({}, {"time_a_ms": 100}, "both frame times"),
        ({}, {"time_a_ms": 100, "time_b_ms": 90}, "later"),
        ({}, {"time_a_ms": -1, "time_b_ms": 100}, "nonnegative"),
        ({"inverse_iterations": 0}, {}, "between"),
        ({"gaussian_window": "yes"}, {}, "boolean"),
    ],
)
def test_invalid_settings_and_timing_are_contained(parameters, extra, message):
    a, b, _, _ = pair()
    block = InterpolateFrames(parameters=parameters)
    assert block.execute({"image_a": a, "image_b": b, **extra}) is None
    assert block.state == "error" and message in block.error


def test_grid_shape_encoding_and_flow_validation():
    a, b, forward, backward = pair()
    grid_a, grid_b = frame([96, 72], "a"), frame([96, 72], "b")
    block = InterpolateFrames()
    assert (
        block.execute(
            {
                "image_a": ValuePacket(a, {"spatial": grid_a}),
                "image_b": ValuePacket(b, {"spatial": grid_b}),
            }
        )
        is None
    )
    assert "Frame grids differ" in block.error
    for change, message in (
        ({"image_b": b[:, :-1]}, "identical"),
        ({"forward": np.zeros((72, 96, 2), np.int16)}, "float32/64"),
        ({"forward": forward * 100}, "maximum"),
    ):
        block = InterpolateFrames(parameters={"flow_mode": "provided"})
        assert (
            block.execute(
                {"image_a": a, "image_b": b, "forward": forward, "backward": backward, **change}
            )
            is None
        )
        assert message in block.error


def test_budget_and_native_errors_are_contained(monkeypatch):
    def broken(*args, **kwargs):
        raise cv2.error("flow backend unavailable")

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", broken)
    block = InterpolateFrames()
    large = np.zeros((1600, 1600), np.uint8)
    assert block.execute({"image_a": large, "image_b": large}) is None
    assert "2.1 MP" in block.error
    a, b, _, _ = pair()
    assert block.execute({"image_a": a, "image_b": b}) is None
    assert "flow backend unavailable" in block.error and not block.current_emissions


@pytest.mark.parametrize("graphics", ["on", "off"])
def test_public_example_intermediate_truth_and_saved_settings(graphics, tmp_path):
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    try:
        for block in graph.blocks.values():
            block.set_graphics_mode(graphics)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        result = graph.blocks["interpolate"].last_result
        truth = cv2.imread(str(EXAMPLE / "assets/middle.png"), cv2.IMREAD_UNCHANGED)
        selected = np.zeros(truth.shape[:2], bool)
        selected[24:-24, 32:-32] = True
        use = selected & (result.outputs["coverage"] != 0)
        assert use.sum() > selected.sum() * 0.95
        error = np.abs(result.outputs["image"].astype(float) - truth)
        assert error[use].mean() < 100
        assert result.outputs["timestamp_ms"] == 20
        saved = tmp_path / "roundtrip.json"
        graph.save(saved)
        raw = json.loads(saved.read_text(encoding="utf-8"))
        restored = Graph.from_dict(raw, builtin_registry())
        try:
            assert (
                restored.blocks["interpolate"].parameters == graph.blocks["interpolate"].parameters
            )
        finally:
            restored.close()
    finally:
        graph.close()
