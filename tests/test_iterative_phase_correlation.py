"""Regression coverage for subpixel registration and unusable native results."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, metadata

TYPE_ID = "inspect.phase_correlation_iterative"


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_subpixel_shift_direction_and_inverse_in_both_precisions(precision):
    texture = np.random.default_rng(547).random((160, 192), dtype=np.float32)
    a = cv2.GaussianBlur(texture, (0, 0), 1.3)
    expected = (4.3125, -5.6875)
    matrix = np.float32([[1, 0, expected[0]], [0, 1, expected[1]]])
    b = cv2.warpAffine(a, matrix, a.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    # The float32 setting must scale the input before narrowing these large samples.
    block = builtin_registry().create(TYPE_ID, parameters={"precision": precision})
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"image_a": a.astype(np.float64) * 1e200, "image_b": b})
    assert result and result.passed, block.error or (result and result.message)
    shift = result.outputs["shift"]
    assert (shift["x"], shift["y"]) == pytest.approx(expected, abs=0.06)
    assert shift["direction"] == "image_a to image_b"
    transform = np.asarray(result.outputs["alignment_transform"]["source_to_target"])
    np.testing.assert_allclose(transform @ [shift["x"], shift["y"], 1], [0, 0, 1])
    assert result.outputs["status"] == "ok"
    assert result.outputs["result_graphics"].layers[0].objects[0].kind == "text"


def test_texture_only_on_window_border_does_not_emit_motion():
    image = np.zeros((24, 32), np.uint8)
    image[0, 1:4] = 255
    block = builtin_registry().create(TYPE_ID)
    result = block.execute({"image_a": image, "image_b": image})
    assert result and not result.passed
    assert block.state == "fail"
    assert result.outputs["status"] == result.fail_reasons[0]["code"] == "phase_flat_image"
    assert result.outputs["shift"] is result.outputs["alignment_transform"] is NO_VALUE


@pytest.mark.parametrize("native_shift", ((-16.0, 0.0), (0.0, float("nan"))))
def test_unusable_native_shift_is_withheld(monkeypatch, native_shift):
    calls = []

    def phase_shift(a, b, neighborhood, iterations):
        calls.append((a.shape, a.dtype, b.dtype, neighborhood, iterations))
        return native_shift

    monkeypatch.setattr(cv2, "phaseCorrelateIterative", phase_shift)
    image = np.random.default_rng(8).random((24, 32))
    block = builtin_registry().create(
        TYPE_ID, parameters={"neighborhood_size": 9, "max_iterations": 3}
    )
    result = block.execute({"image_a": image, "image_b": image})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "phase_ambiguous_shift"
    assert result.outputs["shift"] is result.outputs["alignment_transform"] is NO_VALUE
    assert calls == [((24, 32), np.dtype("float64"), np.dtype("float64"), 9, 3)]


@pytest.mark.parametrize(
    ("shape", "parameters", "error"),
    (
        ((19, 20), {}, "even dimensions"),
        ((2, 20), {}, "at least 4"),
        ((20, 20), {"neighborhood_size": 6}, "positive odd integer"),
        ((20, 20), {"max_iterations": 0}, "between 1 and 1000"),
    ),
)
def test_geometry_and_refinement_limits(shape, parameters, error):
    image = np.ones(shape, np.uint8)
    block = builtin_registry().create(TYPE_ID, parameters=parameters)
    assert block.execute({"image_a": image, "image_b": image}) is None
    assert error in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_applies_measured_correction_to_color_image(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "iterative_phase_correlation"
        / "iterative_phase_correlation.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    reference = graph.blocks["reference"].last_result.outputs["image"]
    moving = graph.blocks["moving"].last_result.outputs["image"]
    aligned = graph.blocks["align"].last_result.outputs["image"]
    outputs = graph.blocks["correlate"].last_result.outputs
    assert (outputs["shift"]["x"], outputs["shift"]["y"]) == pytest.approx(
        (4.3125, -5.6875), abs=0.08
    )
    transform = outputs["alignment_transform"]
    assert transform["source_spatial"] == metadata(moving)["spatial"]
    assert transform["target_spatial"] == metadata(reference)["spatial"]
    assert aligned.shape == moving.shape
    ref = np.asarray(reference, dtype=np.int16)
    assert np.abs(ref - aligned).mean() < np.abs(ref - moving).mean() * 0.25
