"""Verify chart sampling, native CCM fitting, and portable model application."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray


def reference_colors():
    detector = cv2.mcc.CCheckerDetector.create()
    detector.setColorChartType(cv2.mcc.MCC24)
    return detector.getRefColors().reshape(-1, 3)[:, ::-1] / 255.0


def test_detected_chart_patch_order_and_frame_coordinates():
    registry = builtin_registry()
    detector = registry.create("calibration.detect_color_chart")
    rgb = reference_colors()
    image = np.full((360, 480, 3), 32, np.uint8)
    for index, color in enumerate(np.rint(rgb[:, ::-1] * 255).astype(np.uint8)):
        x, y = 50 + (index % 6) * 62, 50 + (index // 6) * 62
        image[y : y + 52, x : x + 52] = color
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[18:330, 20:456] = 255
    result = detector.execute({"image": image, "mask": mask})
    assert result is not None, detector.error
    assert result.passed and result.outputs["count"] == 1
    chart = result.outputs["charts"]["items"][0]
    np.testing.assert_allclose(chart["rgb"], rgb, atol=1 / 255)
    np.testing.assert_allclose(np.mean(chart["corners"], axis=0), [232, 168], atol=8)
    assert len(chart["patch_corners"]) == 24
    missing = detector.execute({"image": image, "mask": np.zeros_like(mask)})
    assert not missing.passed and missing.outputs["charts"]["items"] == []


def test_portable_ccm_corrects_held_out_rgb_bgr_and_preserves_alpha_and_selection():
    registry = builtin_registry()
    reference = reference_colors()
    gains = np.array([0.82, 0.91, 0.75])
    fitter = registry.create("calibration.fit_ccm", parameters={"weights": [0] + [1] * 23})
    fit = fitter.execute({"patches": (reference * gains).tolist(), "reference": reference.tolist()})
    assert fit is not None, fitter.error
    assert fit.outputs["statistics"]["used_patches"] == 23
    portable = json.loads(json.dumps(fit.outputs["model"], allow_nan=False))
    block = registry.create("calibration.apply_ccm", parameters={"model": portable})
    target = np.array([0.31, 0.48, 0.64])
    sample = np.float32(target * gains)
    image = ImageArray(np.array([[[*sample, 0.4], [*sample, 0.7]]], np.float32), "rgba")
    selection = np.array([[255, 0]], np.uint8)
    corrected = block.execute({"image": image, "mask": selection})
    assert corrected is not None, block.error
    np.testing.assert_allclose(corrected.outputs["image"][0, 0, :3], target, atol=0.01)
    np.testing.assert_array_equal(corrected.outputs["image"][0, 1], image[0, 1])
    np.testing.assert_array_equal(corrected.outputs["image"][..., 3], image[..., 3])
    bgr = np.float32([[sample[::-1]]])
    result = block.execute({"image": bgr, "model": portable})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["image"][0, 0], target[::-1], atol=0.01)


@pytest.mark.parametrize("linearization", ["identity", "gamma", "color_polynomial"])
def test_affine_ccm_restores_each_exposed_linearization(linearization):
    registry = builtin_registry()
    reference = reference_colors()
    fit = registry.create(
        "calibration.fit_ccm",
        parameters={
            "matrix_type": "affine",
            "linearization": linearization,
        },
    )
    fitted = fit.execute({"patches": (reference * 0.92).tolist(), "reference": reference.tolist()})
    assert fitted is not None, fit.error
    assert np.asarray(fitted.outputs["matrix"]).shape == (4, 3)
    block = registry.create("calibration.apply_ccm")
    result = block.execute(
        {
            "image": np.full((4, 4, 3), 110, np.uint8),
            "model": fitted.outputs["model"],
        }
    )
    assert result is not None, block.error
    assert result.outputs["image"].dtype == np.uint8


def test_color_chart_example_executes_headlessly():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/calibration/color_chart_calibration/color_chart_calibration.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        assert graph.blocks["fit"].last_result.outputs["statistics"]["used_patches"] >= 20
    finally:
        graph.close()
