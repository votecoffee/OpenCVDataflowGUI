"""Check Pei–Lin normalization against native and independently drawn shapes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.peilin_normalization import PeiLinNormalization
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/peilin_normalization"


@pytest.fixture
def silhouette():
    return cv2.imread(str(EXAMPLE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_example_aligns_independently_rasterized_shapes(profile, workers):
    truth = json.loads((EXAMPLE / "truth.json").read_text("utf-8"))
    graph = Graph.load(EXAMPLE / "peilin_normalization.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["normalize_reference"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        before = graph.blocks["before"].last_result
        after = graph.blocks["after"].last_result
        assert before and before.passed and after and after.passed
        assert before.outputs["measurements"]["iou"] == pytest.approx(0.38860, abs=1e-5)
        assert after.outputs["measurements"]["iou"] == pytest.approx(0.97711, abs=1e-5)
        reference = block.last_result.outputs["transform"]
        distorted = graph.blocks["normalize_distorted"].last_result.outputs["transform"]
        polygon = np.asarray(truth["reference_vertices"], np.float64)
        affine = np.asarray(truth["reference_to_distorted"])
        source_vertices = polygon @ affine[:, :2].T + affine[:, 2]
        a = cv2.transform(polygon[None], np.array(reference["source_to_target"]))[0]
        b = cv2.transform(source_vertices[None], np.array(distorted["source_to_target"]))[0]
        assert np.linalg.norm(a - b, axis=1).max() < 2
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_source_to_target_matches_native_and_wired_affine_warp(silhouette):
    block = PeiLinNormalization(parameters={"interpolation": "nearest"})
    context = frame([240, 220], "part_camera")
    packet = ValuePacket(silhouette, {"spatial": context})
    result = block.execute({"image": packet})
    assert result and result.passed, block.error
    native = cv2.ximgproc.PeiLinNormalization(silhouette)
    np.testing.assert_array_equal(result.outputs["transform"]["source_to_target"], native)
    forward = np.vstack((native, [0, 0, 1]))
    inverse = result.outputs["transform"]["target_to_source"]
    np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(result.payloads["image"]["spatial"]["to_root"], inverse)
    assert result.payloads["image"]["spatial"]["root_id"] == "part_camera"
    warp = builtin_registry().create("cv.warp_affine", parameters={"interpolation": "nearest"})
    wired = warp.execute({"image": packet, "transform": result.outputs["transform"]})
    assert wired and wired.passed, warp.error
    np.testing.assert_array_equal(result.outputs["image"], wired.outputs["image"])


def test_silhouette_selection_preserves_color_encoding_depth_and_canvas(silhouette):
    colors = np.zeros((*silhouette.shape, 4), np.uint16)
    colors[silhouette != 0] = (34021, 12545, 29223, 50000)
    source = ImageArray(colors, "rgba")
    block = PeiLinNormalization(
        parameters={
            "weight_source": "selection",
            "width": 180,
            "height": 160,
            "interpolation": "nearest",
        }
    )
    result = block.execute({"image": source, "mask": silhouette})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    report = result.outputs["report"]
    assert output.shape == (160, 180, 4) and output.color_format == "rgba"
    assert output.dtype == np.uint16
    assert report["support_pixels"] == np.count_nonzero(silhouette)
    assert report["target_spread"] == 20
    matrix = np.asarray(result.outputs["transform"]["source_to_target"])
    expected = cv2.warpAffine(colors, matrix, (180, 160), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_allclose(matrix @ [*report["source_center"], 1.0], [90, 80], atol=1e-10)


def test_dark_threshold_estimates_the_same_shape_and_warps_original_pixels(silhouette):
    dark = 255 - silhouette
    parameters = {"weight_source": "threshold", "polarity": "dark", "interpolation": "nearest"}
    result = PeiLinNormalization(parameters=parameters).execute({"image": dark})
    assert result and result.passed
    np.testing.assert_allclose(
        result.outputs["transform"]["source_to_target"],
        cv2.ximgproc.PeiLinNormalization(silhouette),
        atol=1e-10,
    )
    assert result.outputs["report"]["support_pixels"] == np.count_nonzero(silhouette)


def test_stable_axis_math_when_native_cannot_resolve_eigenvector(monkeypatch, silhouette):
    expected = cv2.ximgproc.PeiLinNormalization(silhouette)
    monkeypatch.setattr(cv2.ximgproc, "PeiLinNormalization", lambda _image: np.full((2, 3), np.nan))
    tool = PeiLinNormalization()
    result = tool.execute({"image": silhouette})
    assert result and result.passed, tool.error
    assert result.outputs["report"]["estimator"] == "stable_moments"
    np.testing.assert_allclose(result.outputs["transform"]["source_to_target"], expected, atol=1e-8)


def test_horizontal_symmetric_rectangle_and_degenerate_input():
    shape = np.zeros((100, 150), np.uint8)
    shape[35:65, 20:106] = 255
    tool = PeiLinNormalization()
    result = tool.execute({"image": shape})
    assert result and result.passed, tool.error
    assert result.outputs["report"]["orientation_resolved"] is False
    assert np.isfinite(result.outputs["image"]).all()
    empty = np.zeros(shape.shape, dtype=np.uint8)
    assert tool.execute({"image": empty}) is None and "nonzero" in tool.error
    empty[40, 20:106] = 255
    assert tool.execute({"image": empty}) is None and "two-dimensional" in tool.error
