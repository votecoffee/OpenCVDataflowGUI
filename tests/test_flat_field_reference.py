"""Independent capture statistics and held-out radiometric correction contracts."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame, metadata, packet_for
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE = "calibration.flat_field_reference"


def batch(images, payloads=None):
    payloads = payloads or [{} for _ in images]
    return FlowGroup(
        tuple(ValuePacket(image, payload) for image, payload in zip(images, payloads, strict=True)),
        tuple(f"capture-{i}" for i in range(len(images))),
    )


def build(bright, dark=None, parameters=None, **extra):
    block = builtin_registry().create(TYPE, parameters=parameters or {})
    inputs = {
        "bright": bright if isinstance(bright, (FlowGroup, ValuePacket)) else batch(bright),
        **extra,
    }
    if dark is not None:
        inputs["dark"] = dark if isinstance(dark, FlowGroup) else batch(dark)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize(
    "method,values,expected",
    (("mean", [10, 20, 90], 40), ("median", [10, 20, 90], 20), ("median", [10, 20, 30, 90], 25)),
)
def test_capture_estimates_use_raw_units_and_independent_order_statistics(method, values, expected):
    bright = [np.full((3, 4), value, np.uint16) for value in values]
    dark = [np.full((3, 4), 2, np.uint16)] * 3
    _, result = build(bright, dark, {"method": method, "minimum_response": 0})
    assert result.passed
    np.testing.assert_array_equal(result.outputs["flat"], expected)
    np.testing.assert_array_equal(result.outputs["dark"], 2)
    np.testing.assert_array_equal(result.outputs["response"], expected - 2)
    assert result.outputs["flat"].dtype == np.float64
    assert result.outputs["bright_noise"] is NO_VALUE


@pytest.mark.parametrize("method", ("mean", "median"))
def test_held_out_color_image_corrects_to_independent_reflectance_truth(method):
    y, x = np.mgrid[:8, :12]
    dark = np.stack([200 + 4 * x, 400 + 4 * y, 100 + 4 * x], axis=2).astype(np.uint16)
    response = np.stack([8000 + 32 * x, 12000 + 64 * y, 16000 + 64 * x], axis=2).astype(np.uint16)
    flat = dark + response
    captures = [flat - 4, flat, flat + 4]
    if method == "median":
        captures[2][2:5, 4:8] += 4000
    _, result = build(
        [ImageArray(a, "rgb") for a in captures],
        [ImageArray(dark - 4, "rgb"), ImageArray(dark, "rgb"), ImageArray(dark + 4, "rgb")],
        {"method": method},
    )
    observed = ImageArray(dark + response // 4, "rgb")
    tool = builtin_registry().create(
        "cv.flat_field",
        parameters={
            "reference_depth": "raw_sample_units",
            "target_mode": "fixed",
            "target_level": 0.5,
        },
    )
    corrected = tool.execute(
        {"image": observed, "flat": packet_for(result, "flat"), "dark": packet_for(result, "dark")}
    )
    assert corrected is not None and corrected.passed, tool.error
    np.testing.assert_array_equal(corrected.outputs["image"], np.full_like(observed, 8192))
    assert corrected.outputs["image"].color_format == "rgb"


def test_saturation_rejection_is_per_capture_pixel_across_color_channels():
    images = [np.full((2, 3, 3), 100, np.uint8) for _ in range(4)]
    images[0][0, 0, 1] = 255
    images[0][0, 1, 0] = images[1][0, 1, 2] = 255
    _, result = build(images, parameters={"minimum_valid_fraction": 0})
    assert result.passed
    assert result.outputs["bright_count"].tolist() == [[3, 2, 4], [4, 4, 4]]
    assert result.outputs["valid_mask"].tolist() == [[255, 0, 255], [255, 255, 255]]
    assert np.all(result.outputs["flat"][0, 0] == 100)
    assert not np.any(result.outputs["flat"][0, 1])


def test_noise_is_population_deviation_about_mean_even_for_median_estimates():
    images = [np.full((2, 3), value, np.uint8) for value in (30, 30, 90)]
    _, result = build(images, parameters={"method": "median", "retain_noise": True})
    np.testing.assert_array_equal(result.outputs["flat"], 30)
    np.testing.assert_allclose(result.outputs["bright_noise"], np.sqrt(800), rtol=1e-14)
    _, rejected = build(
        images, parameters={"method": "median", "check_noise": True, "maximum_noise": 0.1}
    )
    assert not rejected.passed
    assert rejected.outputs["bright_noise"] is NO_VALUE
    assert rejected.outputs["info"]["maximum_bright_noise"] == pytest.approx(np.sqrt(800))
    assert not np.any(rejected.outputs["flat"])


def test_selection_zeroes_unselected_references_without_inventing_invalid_pixels():
    image = np.full((6, 9), 200, np.uint16)
    mask = np.zeros_like(image, np.uint8)
    mask[1:5, 2:7] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [9, 6],
        "shape": "rectangle",
        "rectangle": [4, 0, 4, 6],
    }
    _, result = build([image] * 3, parameters={"minimum_response": 0}, mask=mask, region=region)
    expected = np.zeros_like(image, np.uint8)
    expected[1:5, 4:7] = 255
    np.testing.assert_array_equal(result.outputs["valid_mask"], expected)
    assert result.outputs["info"]["selected_pixels"] == 12
    assert not result.outputs["invalid_mask"].any()
    assert np.count_nonzero(result.outputs["flat"]) == 12


def test_absent_dark_declares_zero_offset_and_all_saturated_batches_fail_safely():
    image = np.full((3, 4), 80, np.uint8)
    _, result = build([image] * 3)
    assert result.outputs["info"]["dark_assumption"] == "zero_offset"
    assert not result.outputs["dark"].any() and not result.outputs["dark_count"].any()
    _, rejected = build([np.full_like(image, 255)] * 3, parameters={"method": "median"})
    assert not rejected.passed
    assert rejected.fail_reasons[0]["code"] == "flat_field_reference_insufficient_valid_pixels"
    assert not rejected.outputs["flat"].any() and not rejected.outputs["bright_count"].any()


def test_fraction_gate_and_published_integer_quantization_are_explicit():
    images = [np.full((2, 2), 100, np.uint8)] * 3
    mask = np.array([[255, 255], [0, 0]], np.uint8)
    dark = [np.array([[100, 0], [0, 0]], np.uint8)] * 3
    _, result = build(images, dark, {"minimum_valid_fraction": 0.5}, mask=mask)
    assert result.passed and result.outputs["info"]["valid_fraction"] == 0.5
    a, b = np.full((2, 2), 10, np.uint8), np.full((2, 2), 11, np.uint8)
    _, floating = build([a, b], [a, a], {"minimum_frames": 2, "minimum_response": 0})
    assert floating.passed and np.all(floating.outputs["response"] == 0.5)
    _, rounded = build(
        [a, b], [a, a], {"minimum_frames": 2, "minimum_response": 0, "output_depth": "same"}
    )
    assert not rounded.passed and not rounded.outputs["valid_mask"].any()


def test_hdr_and_tiny_signals_survive_per_pixel_normalization():
    image = np.array([[1e300, 1e-300]], np.float64)
    _, result = build([image] * 3, parameters={"reject_saturation": False, "minimum_response": 0})
    assert result.passed
    np.testing.assert_array_equal(result.outputs["flat"], image)
    varying = [np.array([[0.0, 0.0]]), image]
    _, noise = build(
        varying,
        parameters={
            "minimum_frames": 2,
            "reject_saturation": False,
            "minimum_response": 0,
            "retain_noise": True,
        },
    )
    np.testing.assert_allclose(noise.outputs["bright_noise"], image / 2, rtol=1e-14, atol=0)


def test_color_alpha_source_and_calibration_payload_ownership():
    clock = TicketClock()
    context = {
        "spatial": frame([4, 3], "camera"),
        "fixture": {"lot": "cal"},
        "pyramid": {
            "level": 1,
            "largest_size": [8, 6],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    images = [ImageArray(np.full((3, 4, 4), 1000, np.uint16), "rgba") for _ in range(3)]
    images[0][..., 3] = 2345
    contexts = [
        {**context, "ticket": clock.issue(str(i)), "alpha_representation": "straight"}
        for i in range(3)
    ]
    cohort = clock.issue("calibration-batch")
    _, result = build(ValuePacket(batch(images, contexts), {"ticket": cohort}))
    assert result.passed and result.outputs["flat"].shape == (3, 4, 3)
    assert result.outputs["flat"].color_format == "rgb"
    assert result.payloads["flat"]["ticket"] == cohort
    assert result.payloads["flat"]["spatial"] == context["spatial"]
    assert result.payloads["flat"]["pyramid"] == context["pyramid"]
    assert "alpha_representation" not in result.payloads["flat"]
    assert result.payloads["source"]["ticket"] == contexts[0]["ticket"]
    np.testing.assert_array_equal(result.outputs["source"], images[0])
    assert result.outputs["info"]["captures"]["bright"][2]["ticket"] == contexts[2]["ticket"]


def test_alignment_validates_capture_and_selection_grids():
    image = np.full((3, 4), 100, np.uint8)
    contexts = [{"spatial": frame([4, 3], str(i))} for i in range(3)]
    group = batch([image] * 3, contexts)
    block = builtin_registry().create(TYPE)
    assert block.execute({"bright": group}) is None
    assert block.error_kind == "invalid"
    _, result = build(group, parameters={"alignment": "pixel_grid"})
    assert result.passed
    mask = ImageArray(np.full_like(image, 255), "gray", {"spatial": frame([4, 3], "wrong")})
    block.parameters["alignment"] = "pixel_grid"
    assert block.execute({"bright": group, "mask": mask}) is None
    bad = batch([image] * 3, [{"spatial": frame([5, 3], "wrong")}] * 3)
    assert block.execute({"bright": bad}) is None


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros((2, 3), np.int16),
        np.full((2, 3), np.nan),
        ImageArray(np.zeros((2, 3, 3), np.uint8), "hsv"),
    ),
)
def test_unsupported_captures_fail_common_validation(bad):
    block = builtin_registry().create(TYPE)
    assert block.execute({"bright": batch([bad] * 3)}) is None
    assert block.error_kind == "invalid"


@pytest.mark.parametrize(
    "parameters",
    (
        {"minimum_frames": 4},
        {"anchor_index": 3},
        {"saturation_level": 0},
        {"minimum_response": 1.1},
        {"minimum_valid_fraction": 2},
    ),
)
def test_invalid_capture_configuration_is_rejected(parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    image = np.full((3, 4), 100, np.uint8)
    assert block.execute({"bright": batch([image] * 3)}) is None
    assert block.error_kind == "invalid"


def test_reference_precision_and_color_layout_guards():
    block = builtin_registry().create(
        TYPE, parameters={"output_depth": "float32", "reject_saturation": False}
    )
    assert block.execute({"bright": batch([np.full((2, 3), 1e300)] * 3)}) is None
    a = ImageArray(np.ones((2, 3, 3), np.uint8), "rgb")
    b = ImageArray(np.ones((2, 3, 3), np.uint8), "bgr")
    assert block.execute({"bright": batch([a, a, b])}) is None
    rgba = ImageArray(np.ones((2, 3, 4), np.uint16), "rgba")
    assert (
        block.execute(
            {"bright": batch([rgba] * 3, [{"alpha_representation": "premultiplied"}] * 3)}
        )
        is None
    )


def test_correction_accepts_raw_floating_rgb_references_for_rgba_without_changing_alpha():
    source = ImageArray(np.full((2, 3, 4), 1000, np.uint16), "rgba")
    flat = ImageArray(np.full((2, 3, 3), 10000.5), "rgb")
    tool = builtin_registry().create(
        "cv.flat_field", parameters={"reference_depth": "raw_sample_units", "target_mode": "fixed"}
    )
    result = tool.execute({"image": source, "flat": flat})
    assert result is not None and result.passed, tool.error
    np.testing.assert_array_equal(result.outputs["image"][..., 3], 1000)
    np.testing.assert_array_equal(result.outputs["image"][..., :3], 3277)
    strict = builtin_registry().create("cv.flat_field")
    assert strict.execute({"image": source, "flat": flat}) is None
    premultiplied = ImageArray(
        np.asarray(source), "rgba", {"alpha_representation": "premultiplied"}
    )
    assert tool.execute({"image": premultiplied, "flat": flat}) is None


def test_correction_float64_mean_avoids_overflow_and_can_check_frames():
    flat = np.full((2, 3), 3e307)
    source = flat / 2
    tool = builtin_registry().create("cv.flat_field")
    result = tool.execute({"image": source, "flat": flat})
    assert result is not None and result.passed, tool.error
    np.testing.assert_allclose(result.outputs["image"], source)
    tool.parameters["reference_alignment"] = "same_frame"
    a, b = [
        ImageArray(value, "gray", {"spatial": frame([3, 2], root)})
        for value, root in ((source, "a"), (flat, "b"))
    ]
    assert tool.execute({"image": a, "flat": b}) is None


@pytest.mark.parametrize("depth", (np.float32, np.float64))
def test_floating_correction_levels_are_raw_units_without_an_artificial_unit_interval(depth):
    source = np.full((2, 3), 100, depth)
    tool = builtin_registry().create(
        "cv.flat_field",
        parameters={
            "target_mode": "fixed",
            "target_level": 100.5,
            "min_response": 2.0,
        },
    )
    result = tool.execute({"image": source, "flat": source * 2})
    assert result is not None and result.passed, tool.error
    np.testing.assert_array_equal(result.outputs["image"], 50.25)


@pytest.mark.parametrize("field", ("target_level", "min_response"))
def test_unsigned_correction_levels_still_require_full_scale_fractions(field):
    source = np.full((2, 3), 100, np.uint8)
    tool = builtin_registry().create("cv.flat_field", parameters={field: 1.1})
    assert tool.execute({"image": source, "flat": source}) is None
    assert tool.error_kind == "invalid"


def test_graphics_disabled_skips_preview_conversion(monkeypatch):
    import opencv_dataflow.flat_field_reference as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected preview conversion")

    monkeypatch.setattr(implementation.np, "rint", unexpected)
    block = builtin_registry().create(TYPE)
    block.set_graphics_mode("off")
    image = np.full((3, 4), 0.5)
    result = block.execute({"bright": batch([image] * 3)})
    assert result is not None and result.passed, block.error
    assert result.outputs["preview"] is NO_VALUE and not result.graphics


def test_configuration_documentation_and_layout_round_trip_through_xml(tmp_path):
    registry, graph = builtin_registry(), Graph()
    try:
        graph.add(registry.create(TYPE, parameters={"method": "median", "retain_noise": True}))
        graph.add(
            registry.create("cv.flat_field", parameters={"reference_depth": "raw_sample_units"})
        )
        expected = graph.to_dict()
        path = tmp_path / "calibration.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


def test_sample_budget_is_checked_before_reduction(monkeypatch):
    import opencv_dataflow.flat_field_reference as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized batch reached reduction")

    monkeypatch.setattr(implementation, "estimate", unexpected)
    block = builtin_registry().create(TYPE)
    image = np.zeros((1024, 1024, 3), np.uint8)
    assert block.execute({"bright": batch([image] * 24)}) is None
    assert block.error_kind == "invalid" and "64 million" in block.error


def test_dark_capture_support_and_positive_response_are_independent_quality_gates():
    bright = [np.full((2, 3), 100, np.uint8)] * 3
    dark = [np.full((2, 3), 10, np.uint8) for _ in range(3)]
    dark[0][0, 0] = 255
    for image in dark:
        image[0, 1] = 110
    _, result = build(bright, dark, {"minimum_valid_fraction": 0})
    assert result.passed
    assert result.outputs["dark_count"][0, 0] == 2
    assert result.outputs["valid_mask"].tolist() == [[0, 0, 255], [255, 255, 255]]
    assert result.payloads["dark_count"]["flat_field_reference"]["units"] == "capture_count"


@pytest.mark.local_examples
@pytest.mark.parametrize("filename", ("project.json", "median.json"))
def test_public_project_corrects_only_a_held_out_capture_to_exact_truth(filename):
    folder = Path(__file__).parents[1] / "examples/custom tools/flat_field_reference"
    graph = Graph.load(folder / filename, builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        assert graph.blocks["verify"].last_result.outputs["value"] == 0
        assert graph.blocks["before"].last_result.outputs["value"] > 1000
        reference = graph.blocks["reference"].last_result.outputs
        assert reference["info"]["valid_fraction"] == 1
        assert reference["flat"].dtype == np.float64
        output = graph.blocks["correct"].last_result.outputs["image"]
        assert output.dtype == np.uint16 and output.color_format == "rgb"
        assert metadata(output)["ticket"]["value"] == "held_out.png"
    finally:
        graph.close()
