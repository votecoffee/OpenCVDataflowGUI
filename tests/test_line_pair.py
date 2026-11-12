"""Focused fixtures for line corners, support gaps, frames and measurement failures."""

import math
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.objects import collection, line_item
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata
from opencv_dataflow.tickets import TicketClock

TYPE = "inspect.line_pair"


def _line(a, b):
    return {"shape": "line", "endpoints": [a, b]}


def _fixture(**parameters):
    space = frame([200, 150], "camera")
    payload = {"spatial": space, "ticket": TicketClock().issue("camera")}
    image = np.full((150, 200, 3), 33, np.uint16)
    block = builtin_registry().create(TYPE, parameters=parameters)
    inputs = {
        "image": ValuePacket(image, payload),
        "line_a": ValuePacket(_line([20, 40], [140, 40]), payload),
        "line_b": ValuePacket(_line([55, 70], [55, 120]), payload),
    }
    return block, inputs, payload


def test_native_fits_measure_subpixel_corner_and_emit_drawable_graphics():
    block, inputs, payload = _fixture(check_angle=True, angle_tolerance=0.5)
    registry = builtin_registry()
    rng = np.random.default_rng(174)
    for port, points in (
        ("line_a", [[x, 40 + rng.normal(0, 0.01)] for x in range(20, 141, 3)]),
        ("line_b", [[55 + rng.normal(0, 0.01), y] for y in range(70, 121, 2)]),
    ):
        fit = registry.create("inspect.fit_shape")
        fitted = fit.execute({"points": ValuePacket(points, payload)})
        assert fitted is not None and fitted.passed, fit.error
        inputs[port] = ValuePacket(fitted.outputs["fit"], fitted.payloads["fit"])
    block.exposed_outputs.append("result_graphics")
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    np.testing.assert_allclose(result.outputs["vertex"], [55, 40], atol=0.02)
    assert result.outputs["angle"] == pytest.approx(90, abs=0.02)
    assert result.outputs["gap"] is NO_VALUE
    assert result.outputs["image"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["image"], inputs["image"].value)
    assert len(result.outputs["result_graphics"].layers) == 2
    assert result.payloads["vertex"]["spatial"] == payload["spatial"]
    assert result.payloads["vertex"]["ticket"] == payload["ticket"]


def test_aligns_cropped_pyramid_lines_to_the_current_image_grid():
    block, inputs, payload = _fixture()
    crop = frame([60, 40], "camera")
    crop["root_size"] = [200, 150]
    crop["to_root"] = [[2.0, 0.0, 20.0], [0.0, 2.0, 10.0], [0.0, 0.0, 1.0]]
    inputs["line_a"] = ValuePacket(_line([0, 10], [40, 10]), {**payload, "spatial": crop})
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    np.testing.assert_allclose(result.outputs["vertex"], [55, 30], atol=1e-12)
    assert result.outputs["measurements"]["line_a_pixels"] == [[20.0, 30.0], [100.0, 30.0]]


@pytest.mark.parametrize("angle_mode,expected", [("acute", 60), ("obtuse", 120)])
def test_angle_choice_is_independent_of_endpoint_order(angle_mode, expected):
    block, inputs, payload = _fixture(
        angle_mode=angle_mode, check_angle=True, target_angle=expected, angle_tolerance=0.01
    )
    vertex = np.array([55.0, 40.0])
    direction = np.array([0.5, math.sqrt(3) / 2])
    endpoints = vertex + np.array([50.0, 10.0])[:, None] * direction
    inputs["line_b"] = ValuePacket(_line(*endpoints.tolist()), payload)
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    assert result.outputs["angle"] == pytest.approx(expected)
    np.testing.assert_allclose(result.outputs["vertex"], vertex, atol=1e-12)


@pytest.mark.parametrize("tilt", [0.0, 1e-7])
def test_parallel_and_poorly_conditioned_corners_are_unmeasurable(tilt):
    block, inputs, payload = _fixture()
    inputs["line_b"] = ValuePacket(_line([20, 50], [140, 50 + tilt]), payload)
    result = block.execute(inputs)
    assert result is not None and not result.passed, block.error
    assert block.state == "fail"
    assert result.outputs["measurements"]["status"] == "unmeasurable"
    assert result.outputs["vertex"] is result.outputs["angle"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "line_pair_parallel"


def test_gap_measures_common_support_and_checks_the_full_taper_range():
    block, inputs, payload = _fixture(
        mode="parallel_gap",
        maximum_parallel_angle=5,
        check_gap=True,
        minimum_gap=5,
        maximum_gap=9,
    )
    inputs["line_a"] = ValuePacket(_line([20, 40], [120, 40]), payload)
    inputs["line_b"] = ValuePacket(_line([20, 44], [120, 50]), payload)
    result = block.execute(inputs)
    assert result is not None and not result.passed, block.error
    report = result.outputs["measurements"]
    assert report["status"] == "measured"
    assert report["overlap"] == pytest.approx(100)
    assert [report[k] for k in ("gap", "gap_min", "gap_max")] == pytest.approx([7, 4, 10])
    assert result.outputs["gap"] == pytest.approx(7)
    assert result.outputs["vertex"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "line_pair_gap_out_of_range"


def test_angle_only_allows_parallel_lines_without_a_vertex():
    block, inputs, payload = _fixture(mode="angle", check_angle=True, target_angle=0)
    inputs["line_b"] = ValuePacket(_line([150, 50], [180, 50]), payload)
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    assert result.outputs["angle"] == 0
    assert result.outputs["vertex"] is result.outputs["gap"] is NO_VALUE


def test_gap_rejects_nonparallel_or_disjoint_support():
    block, inputs, payload = _fixture(mode="parallel_gap")
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "line_pair_not_parallel"
    inputs["line_b"] = ValuePacket(_line([150, 50], [180, 50]), payload)
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "line_pair_no_overlap"
    assert result.outputs["gap"] is NO_VALUE


@pytest.mark.parametrize("mode", ["corner", "parallel_gap"])
def test_workplane_scale_converts_lengths_and_vertex_using_its_origin(mode):
    block, inputs, payload = _fixture(units="mm", mode=mode)
    inputs["scale"] = ValuePacket(
        {
            "model": "rectified_workplane_v1",
            "spatial": payload["spatial"],
            "units": "mm",
            "units_per_pixel": 0.25,
            "origin": [-2, 1],
        },
        payload,
    )
    inputs["valid_mask"] = ValuePacket(np.full((150, 200), 255, np.uint8), payload)
    if mode == "parallel_gap":
        inputs["line_b"] = ValuePacket(_line([50, 50], [160, 50]), payload)
        block.parameters.update(check_gap=True, minimum_gap=2.5, maximum_gap=2.5)
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    if mode == "corner":
        assert result.outputs["vertex"] == pytest.approx([11.75, 11])
        assert result.outputs["vertex_pixels"] == [55, 40]
        assert "spatial" not in result.payloads["vertex"]
        assert result.payloads["vertex"]["coordinates"]["units"] == "mm"
        assert result.outputs["measurements"]["extension"] == pytest.approx([0, 7.5])
    else:
        assert result.outputs["gap"] == pytest.approx(2.5)
        assert result.outputs["measurements"]["overlap"] == pytest.approx(22.5)
    assert result.outputs["measurements"]["units"] == "mm"


def test_coverage_holes_withhold_measurements_and_limits_keep_rejected_values():
    block, inputs, payload = _fixture(check_angle=True, target_angle=80, angle_tolerance=1)
    measured = block.execute(inputs)
    assert measured is not None and not measured.passed
    assert measured.outputs["angle"] == 90
    assert measured.fail_reasons[0]["code"] == "line_pair_angle_out_of_range"
    mask = np.full((150, 200), 255, np.uint8)
    mask[40, 55] = 0
    inputs["valid_mask"] = ValuePacket(mask, payload)
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.outputs["vertex"] is result.outputs["angle"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "line_pair_invalid_coverage"


def test_extrapolation_limit_qualifies_the_intersection():
    block, inputs, _ = _fixture(maximum_extension=20)
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.outputs["vertex"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "line_pair_extrapolation"


@pytest.mark.parametrize("problem", ["ticket", "frame", "mask_ticket", "degenerate"])
def test_wrong_ticket_frame_and_bad_geometry_are_invalid(problem):
    block, inputs, payload = _fixture()
    if problem == "ticket":
        inputs["line_a"] = ValuePacket(
            inputs["line_a"].value,
            {
                **payload,
                "ticket": TicketClock().issue("other"),
            },
        )
    elif problem == "frame":
        inputs["line_a"] = ValuePacket(
            inputs["line_a"].value,
            {
                **payload,
                "spatial": frame([200, 150], "other"),
            },
        )
    elif problem == "mask_ticket":
        inputs["valid_mask"] = ValuePacket(
            np.full((150, 200), 255, np.uint8),
            {
                **payload,
                "ticket": TicketClock().issue("other"),
            },
        )
    else:
        inputs["line_a"] = ValuePacket(_line([20, 40], [20, 40]), payload)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid"


def test_collections_select_lines_and_empty_detection_is_a_failed_measurement():
    block, inputs, payload = _fixture(index_a=1)
    data = collection(
        "lines",
        [
            line_item([[10, 15], [90, 15]], 0),
            line_item([[20, 40], [140, 40]], 1),
        ],
        inputs["image"].value,
        payload,
    )
    inputs["line_a"] = ValuePacket(data, payload)
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    assert result.outputs["vertex"] == [55, 40]
    assert result.outputs["measurements"]["source_ids"][0] == 1
    inputs["line_a"] = ValuePacket(collection("lines", [], inputs["image"].value, payload), payload)
    missing = block.execute(inputs)
    assert missing is not None and not missing.passed
    assert block.state == "fail" and missing.outputs["vertex"] is NO_VALUE
    assert missing.fail_reasons[0]["code"] == "line_pair_missing_line"


@pytest.mark.local_examples
@pytest.mark.parametrize("project,expected", [("corner", [90.0, 80.0]), ("gap", [20.0, 30.0])])
def test_saved_examples_detect_real_lines_and_replay_pass_and_fail(project, expected):
    folder = Path(__file__).parents[1] / "examples/custom tools/line_pair_metrology"
    graph = Graph.load(folder / f"{project}.json", builtin_registry())
    try:
        for index, state in enumerate(("pass", "fail")):
            graph.run()
            measurement = graph.blocks["measure"]
            assert measurement.state == state, measurement.error
            result = measurement.last_result
            port = "angle" if project == "corner" else "gap"
            assert result.outputs[port] == pytest.approx(expected[index], abs=0.4)
            if project == "corner":
                np.testing.assert_allclose(result.outputs["vertex"], [60, 60], atol=2)
            assert metadata(result.outputs["image"])["ticket"]["kind"] == "string"
    finally:
        graph.close()
