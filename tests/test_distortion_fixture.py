"""Independent donor equations, actual resampling and coordinate-map ownership."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.distortion_fixture import (
    COEFFICIENTS,
    DistortImageFixture,
    evaluate_distortion,
)
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark


def run(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def analytic(points, size, coefficients, center=(0.5, 0.5), radius_scale=0.5):
    """Homogeneous projective multiply followed by explicit polynomial terms."""
    pts = np.asarray(points, float)
    midpoint = (np.array(size) - 1) * center
    radius = max(size) * radius_scale
    values = np.c_[(pts - midpoint) / radius, np.ones(len(pts))]
    c = {**dict.fromkeys(COEFFICIENTS, 0.0), **coefficients}
    matrix = [
        [1, c["shear_x"], 0],
        [c["shear_y"], 1, 0],
        [c["perspective_x"], c["perspective_y"], 1],
    ]
    projected = values @ np.asarray(matrix).T
    x, y = (projected[:, :2] / projected[:, 2:]).T
    squared = x * x + y * y
    radial = 1 + c["k1"] * squared + c["k2"] * squared**2
    out = np.c_[
        x * radial + 2 * c["p1"] * x * y + c["p2"] * (3 * x * x + y * y),
        y * radial + c["p1"] * (x * x + 3 * y * y) + 2 * c["p2"] * x * y,
    ]
    return out * radius + midpoint


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"shear_x": 0.17, "shear_y": -0.09},
        {"perspective_x": 0.06, "perspective_y": -0.05},
        {"k1": 0.08, "k2": 0.015, "p1": 0.002, "p2": -0.003},
        {"shear_x": 0.12, "perspective_y": 0.04, "k1": 0.05, "p2": 0.003},
    ],
)
def test_maps_and_continuous_probe_pairs_match_independent_equations(coefficients):
    yy, xx = np.indices((60, 80), np.float64)
    probes = [[7.25, 13.5], [39.5, 29.5], [65.75, 48.125]]
    result = run(
        DistortImageFixture(parameters={**coefficients, "probe_points": probes}), image=xx + 2 * yy
    )
    expected = analytic(np.c_[xx.ravel(), yy.ravel()], [80, 60], coefficients)
    np.testing.assert_array_equal(
        result.outputs["map_x"], expected[:, 0].reshape(60, 80).astype(np.float32)
    )
    np.testing.assert_array_equal(
        result.outputs["map_y"], expected[:, 1].reshape(60, 80).astype(np.float32)
    )
    np.testing.assert_allclose(
        [p["input"] for p in result.outputs["point_pairs"]],
        analytic(probes, [80, 60], coefficients),
        atol=1e-12,
    )
    assert result.outputs["mapping"]["direction"] == "output_to_input"
    assert result.outputs["mapping"]["global_invertibility"] == "not_established"


def test_analytic_jacobian_matches_independent_numerical_derivatives():
    c = dict(zip(COEFFICIENTS, (0.13, -0.07, 0.04, -0.06, 0.04, 0.01, 0.002, -0.003), strict=True))
    points = np.array([[12, 17], [39.5, 29.5], [65.75, 48.125]])
    h = 1e-4
    dx = (analytic(points + [h, 0], [80, 60], c) - analytic(points - [h, 0], [80, 60], c)) / (2 * h)
    dy = (analytic(points + [0, h], [80, 60], c) - analytic(points - [0, h], [80, 60], c)) / (2 * h)
    expected = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    _, _, jacobian = evaluate_distortion(points[:, 0], points[:, 1], c, [39.5, 29.5], 40)
    np.testing.assert_allclose(jacobian, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("depth", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_identity_preserves_rgb_alpha_samples_and_explicit_interpolation_support(depth):
    maximum = -1000 if depth == np.int16 else 200 if np.dtype(depth).kind == "u" else 3.0
    image = ImageArray(
        np.full((18, 24, 4), (maximum, maximum / 2, maximum / 4, maximum), depth), "rgba"
    )
    for method, halo in [("nearest", 0), ("linear", 1), ("cubic", 2), ("lanczos4", 4)]:
        if depth == np.float64 and method == "cubic":
            block = DistortImageFixture(parameters={"interpolation": method})
            assert block.execute({"image": image}) is None
            assert "Float64 cubic" in block.error
            continue
        result = run(DistortImageFixture(parameters={"interpolation": method}), image=image)
        np.testing.assert_array_equal(result.outputs["image"], image)
        assert (
            result.outputs["image"].dtype == depth
            and result.outputs["image"].color_format == "rgba"
        )
        support = np.zeros((18, 24), np.uint8)
        support[halo : 18 - halo if halo else 18, halo : 24 - halo if halo else 24] = 255
        np.testing.assert_array_equal(result.outputs["valid_mask"], support)


def test_shear_resamples_an_integer_ramp_with_expected_quantization_and_border():
    yy, xx = np.indices((32, 40), np.float64)
    image = xx + 100 * yy
    original = image.copy()
    result = run(
        DistortImageFixture(parameters={"shear_x": 0.25, "center_y": 0, "border_value": -9}),
        image=image,
    )
    # Every fourth row has an integer donor shift. No native remap defines this reference.
    for row in range(0, 32, 4):
        offset = row // 4
        np.testing.assert_array_equal(
            result.outputs["image"][row, : 40 - offset], image[row, offset:]
        )
        if offset:
            np.testing.assert_array_equal(result.outputs["image"][row, 40 - offset :], -9)
    np.testing.assert_array_equal(image, original)


def test_seeded_bounds_repeat_wired_seed_wins_and_global_rng_is_untouched():
    parameters = {
        "coefficient_mode": "uniform_bounds",
        "shear_x": 0.1,
        "perspective_y": 0.02,
        "k1": 0.03,
        "seed": 7,
    }
    block = DistortImageFixture(parameters=parameters)
    image = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    before = np.random.get_state()
    first = run(block, image=image, seed=19)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    second = run(DistortImageFixture(parameters=parameters), image=image, seed=19)
    np.testing.assert_array_equal(first.outputs["image"], second.outputs["image"])
    expected_rng = np.random.Generator(np.random.PCG64(19))
    expected = {
        name: float(expected_rng.uniform(-parameters.get(name, 0), parameters.get(name, 0)))
        for name in COEFFICIENTS
    }
    assert first.outputs["mapping"]["coefficients"] == expected
    assert first.outputs["seed"] == 19
    assert not np.array_equal(
        first.outputs["map_x"], run(block, image=image, seed=20).outputs["map_x"]
    )


def test_selection_limits_support_without_blending_or_clipping_warped_pixels():
    image = np.full((32, 40), 200, np.uint8)
    mask = np.zeros_like(image)
    mask[4:28, 5:35] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [40, 32],
        "shape": "rectangle",
        "rectangle": [10, 8, 20, 16],
    }
    result = run(
        DistortImageFixture(parameters={"interpolation": "linear"}),
        image=image,
        mask=mask,
        region=region,
    )
    expected = np.zeros_like(image)
    expected[9:23, 11:29] = 255
    np.testing.assert_array_equal(result.outputs["valid_mask"], expected)
    np.testing.assert_array_equal(result.outputs["image"], image)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_typed_maps_reuse_through_existing_remappers_with_source_timing_and_output_frame(
    packed, interpolation
):
    registry = builtin_registry()
    image = ImageArray(np.arange(48 * 64 * 4, dtype=np.uint16).reshape(48, 64, 4), "rgba")
    timing = advance_timing(None, mark(), mark(), source=True)
    context = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([64, 48], "camera-grid"),
        "pyramid": {
            "level": 1,
            "largest_size": [128, 96],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "fixture": {"lot": "A"},
        "processing_time": timing,
        "alpha_representation": "straight",
    }
    source = ValuePacket(image, context)
    result = run(
        registry.create(
            "cv.distortion_fixture",
            parameters={"shear_x": 0.12, "k1": 0.04, "interpolation": interpolation},
        ),
        image=source,
    )
    inputs = {
        "image": source,
        "map_x": packet_for(result, "map_x"),
        "map_y": packet_for(result, "map_y"),
    }
    if packed:
        pack = registry.create(
            "cv.pack_remap_map",
            parameters={"sampling": "nearest" if interpolation == "nearest" else "subpixel"},
        )
        converted = run(pack, map_x=inputs["map_x"], map_y=inputs["map_y"])
        inputs = {"image": source, "packed_map": packet_for(converted, "packed_map")}
    remapped = run(
        registry.create(
            "cv.remap_packed" if packed else "cv.remap", parameters={"interpolation": interpolation}
        ),
        **inputs,
    )
    if packed and interpolation == "linear":
        native_maps = converted.outputs["packed_map"].native_maps()
        expected = cv2.remap(np.asarray(image), *native_maps, cv2.INTER_LINEAR)
        np.testing.assert_array_equal(remapped.outputs["image"], expected)
        error = np.abs(np.asarray(expected, float) - result.outputs["image"])
        # Each coordinate rounds by <=1/64; bilinear per-axis variation is <=range.
        assert error.max() <= float(image.max()) / 32 + 2
    else:
        np.testing.assert_array_equal(remapped.outputs["image"], result.outputs["image"])
    assert remapped.payloads["image"]["spatial"] == result.payloads["image"]["spatial"]
    assert remapped.payloads["image"]["spatial"] != context["spatial"]
    assert "pyramid" not in remapped.payloads["image"]
    for port in ("image", "map_x", "map_y", "valid_mask", "point_pairs"):
        payload = result.payloads[port]
        assert payload["ticket"] == context["ticket"] and payload["fixture"] == context["fixture"]
        assert payload["processing_time"]["started_at"] == timing["started_at"]
    assert result.outputs["mapping"]["source_context"]["pyramid"] == context["pyramid"]
    assert "alpha_representation" not in result.payloads["valid_mask"]


def test_swapped_mixed_cropped_and_wrong_source_maps_fail_without_outputs():
    registry = builtin_registry()
    image = ImageArray(np.zeros((32, 40), np.uint8), "gray", {"spatial": frame([40, 32], "camera")})
    first = run(DistortImageFixture(), image=image)
    second = run(DistortImageFixture(parameters={"k1": 0.1}), image=image)
    x, y = packet_for(first, "map_x"), packet_for(first, "map_y")
    cases = [
        (image, y, x),
        (image, x, packet_for(second, "map_y")),
        (image, x, first.outputs["map_y"].view(np.ndarray)),
        (
            ImageArray(np.zeros((32, 40), np.uint8), "gray", {"spatial": frame([40, 32], "other")}),
            x,
            y,
        ),
    ]
    crop = run(registry.create("cv.crop", parameters={"x": 1}), image=x)
    crop_y = run(registry.create("cv.crop", parameters={"x": 1}), image=y)
    cases.append((image, packet_for(crop, "image"), packet_for(crop_y, "image")))
    for source, mx, my in cases:
        block = registry.create("cv.remap")
        assert block.execute({"image": source, "map_x": mx, "map_y": my}) is None
        assert block.error and not block.current_emissions


@pytest.mark.parametrize(
    "parameters,extra",
    [
        ({"coefficient_mode": "uniform_bounds", "k1": -0.1}, {}),
        ({"k1": -1}, {}),
        ({"perspective_x": -0.5, "center_x": 0, "radius_scale": 0.1}, {}),
        ({"shear_x": float("nan")}, {}),
        ({}, {"seed": 0.5}),
        ({}, {"points": [[200, 2]]}),
        ({}, {"points": [[1, 2, 3]]}),
    ],
)
def test_invalid_models_or_controls_are_failsafe(parameters, extra):
    block = DistortImageFixture(parameters=parameters)
    assert block.execute({"image": np.zeros((48, 64), np.uint8), **extra}) is None
    assert block.error and not block.current_emissions


def test_pixel_budget_is_checked_before_allocating_grid(monkeypatch):
    import opencv_dataflow.distortion_fixture as module

    monkeypatch.setattr(module, "MAX_FIXTURE_PIXELS", 10)
    block = DistortImageFixture()
    assert block.execute({"image": np.zeros((32, 40), np.uint8)}) is None
    assert "limited to" in block.error


@pytest.mark.parametrize("packed", [False, True])
def test_existing_remappers_reject_the_native_float64_cubic_boundary_bug(packed):
    registry = builtin_registry()
    y, x = np.indices((8, 10), np.float32)
    inputs = {"image": np.full((8, 10), 1.5, np.float64), "map_x": x, "map_y": y}
    if packed:
        converted = run(registry.create("cv.pack_remap_map"), map_x=x, map_y=y)
        inputs = {"image": inputs["image"], "packed_map": converted.outputs["packed_map"]}
    block = registry.create(
        "cv.remap_packed" if packed else "cv.remap", parameters={"interpolation": "cubic"}
    )
    assert block.execute(inputs) is None
    assert "Float64 cubic" in block.error and not block.current_emissions


def test_wired_empty_probes_graphics_off_and_xml_roundtrip(tmp_path, monkeypatch):
    import opencv_dataflow.distortion_fixture as module

    def unexpected(*args, **kwargs):
        raise AssertionError("Graphics disabled")

    monkeypatch.setattr(module, "graphic_object", unexpected)
    graph = Graph()
    block = graph.add(DistortImageFixture(parameters={"k1": 0.05}))
    block.set_graphics_mode("off")
    preview = run(block, image=np.zeros((32, 40), np.uint8))
    assert len(preview.outputs["point_pairs"]) == 9 and not preview.drawings
    result = run(block, image=np.zeros((32, 40), np.uint8), points=[])
    assert result.outputs["point_pairs"] == [] and not result.drawings
    try:
        data = graph.to_dict()
        export_project_xml(data, tmp_path / "fixture.xml")
        restored = Graph.from_dict(read_project_xml(tmp_path / "fixture.xml"), builtin_registry())
        try:
            assert restored.to_dict() == data
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_fixture_branches_match_golden_images_maps_and_parallel_execution():
    folder = Path(__file__).parents[1] / "examples/custom tools/distortion_fixture"
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(folder / "project.json", builtin_registry())
    try:
        for workers in (1, 2):
            graph.set_branch_workers(workers)
            graph.run()
            for name, entry in truth["cases"].items():
                block = graph.blocks[name]
                result = block.last_result
                assert result is not None and result.passed, block.error
                assert graph.blocks["verify_" + name].last_result.outputs["value"] == 0
                np.testing.assert_allclose(
                    [pair["input"] for pair in result.outputs["point_pairs"]],
                    entry["input_probes"],
                    atol=1e-11,
                )
                assert (
                    graph.blocks["reuse_" + name].last_result.payloads["image"]["spatial"]
                    == result.payloads["image"]["spatial"]
                )
                assert result.drawings["output sampling probes"].from_output
    finally:
        graph.close()
