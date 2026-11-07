"""Check scalar reductions, selected byte distances, and reference alignment."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame


@pytest.mark.parametrize(
    ("metric", "magnitude", "difference", "relative"),
    [
        ("infinity", 4, 4, 2),
        ("l1", 7, 6, 2),
        ("l2", 5, np.sqrt(20), 2),
        ("l2_squared", 25, 20, 4),
    ],
)
def test_numeric_norm_modes(metric, magnitude, difference, relative):
    block = builtin_registry().create("cv.image_norm", parameters={"metric": metric})
    image = np.array([[-3, 4]], dtype=np.int16)
    reference = np.array([[1, 2]], dtype=np.int16)
    assert block.execute({"image": image}).outputs["value"] == pytest.approx(magnitude)
    inputs = {"image": image, "reference": reference}
    assert block.execute(inputs).outputs["value"] == pytest.approx(difference)
    block.parameters["mode"] = "relative"
    assert block.execute(inputs).outputs["value"] == pytest.approx(relative)


@pytest.mark.parametrize(("metric", "expected"), [("hamming", 6), ("hamming2", 3)])
def test_hamming_norms_use_a_pixel_mask_and_exclude_alpha_by_default(metric, expected):
    block = builtin_registry().create("cv.image_norm", parameters={"metric": metric})
    image = np.array([[[3, 12, 48, 255], [255, 255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[1, 0]], dtype=np.uint8)
    reference = np.zeros_like(image)
    inputs = {"image": image, "reference": reference, "mask": mask}
    result = block.execute(inputs)
    assert result.outputs["value"] == expected
    assert result.outputs["measurements"]["selected_count"] == 1
    block.parameters["include_alpha"] = True
    assert block.execute(inputs).outputs["value"] == expected + (8 if metric == "hamming" else 4)
    assert block.execute({"image": image.astype(np.float32), "mask": mask}) is None
    assert block.error_kind == "invalid" and "uint8" in block.error


def test_absolute_norm_preserves_a_small_difference_next_to_a_large_equal_value():
    block = builtin_registry().create("cv.image_norm")
    inputs = {"image": np.array([[1e300, 3.0]]), "reference": np.array([[1e300, 0.0]])}
    assert block.execute(inputs).outputs["value"] == 3.0
    assert block.execute({"image": np.array([[1e-200]])}).outputs["value"] == pytest.approx(
        1e-200, abs=0
    )


def test_reference_frame_and_relative_denominator_are_validated():
    block = builtin_registry().create("cv.image_norm", parameters={"mode": "relative"})
    image = ImageArray(np.array([[3, 4]], np.uint8), "gray", {"spatial": frame([2, 1], "sample")})
    ref = ImageArray(np.array([[1, 2]], np.uint8), "gray", {"spatial": frame([2, 1], "ref")})
    assert block.execute({"image": image}) is None
    assert "Connect a reference" in block.error
    assert block.execute({"image": image, "reference": ref}) is None
    assert block.error_kind == "invalid" and "coordinate frame" in block.error
    block.parameters["alignment"] = "pixel_grid"
    assert block.execute({"image": image, "reference": ref}).outputs["value"] == pytest.approx(
        np.sqrt(8 / 5)
    )
    assert block.execute({"image": image, "reference": np.zeros_like(ref)}) is None
    assert "nonzero norm" in block.error
    assert (
        block.execute({"image": image, "reference": ref, "mask": np.zeros((1, 2), np.uint8)})
        is None
    )
    assert "at least one selected pixel" in block.error


def test_saved_example_reports_selected_norms_and_inclusive_limit():
    path = Path(__file__).parents[1] / "examples/opencv tools/image_norm/image_norm.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.set_run_profile("lean")
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        assert graph.blocks["absolute"].last_result.outputs["value"] == 4200
        relative = graph.blocks["relative"]
        assert relative.last_result.outputs["value"] == 0.25
        relative.parameters["maximum"] = 0.24
        graph.run()
        assert relative.state == "fail"
        assert relative.last_result.fail_reasons[0]["code"] == "image_norm_above_limit"
    finally:
        graph.close()
