"""Compare Durand variants with a global baseline on known HDR texture."""

from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.core import Graph
from opencv_dataflow.durand_tonemap import ToneMapDurand
from opencv_dataflow.images import ImageArray
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/durand_tonemap"


@pytest.fixture(scope="module")
def bgr():
    return run_path(str(EXAMPLE / "build_example.py"))["fixture"]()


def texture_steps(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    steps = []
    for start in (12, 60, 108):
        band = gray[:, start : start + 25]
        steps.append(float(band[2::4].mean() - band[0::4].mean()))
    return np.array(steps)


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_graph_recovers_shadow_texture(bgr, profile, workers):
    from opencv_dataflow.blocks import builtin_registry

    graph = Graph.load(EXAMPLE / "durand_tonemap.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["local"]
        assert block_entry(block).path[:2] == (
            "OpenCV", "Extended Computational Photography (xphoto)",
        )
        source = graph.blocks["source"].last_result
        local = block.last_result
        baseline = graph.blocks["linear"].last_result
        assert local and local.passed, block.error
        assert baseline and baseline.passed
        np.testing.assert_array_equal(source.outputs["image"], bgr)
        np.testing.assert_allclose(texture_steps(baseline.outputs["image"]), [
            0.0011638, 0.0143072, 0.1729507,
        ], atol=1e-6)
        np.testing.assert_allclose(texture_steps(local.outputs["image"]), [
            0.0105955, 0.0417339, 0.1628754,
        ], atol=1e-5)
        assert local.outputs["info"] == {"backend": "portable", "output_depth": "float32"}
        for key in ("ticket", "spatial"):
            assert local.payloads["image"][key] == source.payloads["image"][key]
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_portable_black_rgb_order_gamma_and_display_depth(bgr):
    bgr = bgr.copy()
    bgr[:2, :5] = 0
    bgr[2, 0] = (1, 0, 0)
    settings = {"output_depth": "float32"}
    block = ToneMapDurand(parameters=settings)
    result = block.execute({"image": bgr})
    assert result and result.passed, block.error
    image = result.outputs["image"]
    assert image.color_format == "bgr" and image.dtype == np.float32
    assert image.min() == 0 and image.max() == 1
    assert np.count_nonzero(image[:2, :5]) == 0
    assert image[2, 0, 0] > 0 and not image[2, 0, 1:].any()

    for encoding in ("rgb", "bgr"):
        samples = bgr if encoding == "bgr" else bgr[..., ::-1]
        previous = samples.copy()
        tool = ToneMapDurand(parameters={"output_depth": "uint8", "gamma": 2.0})
        output = tool.execute({"image": ImageArray(samples, encoding)}).outputs["image"]
        expected = np.rint(np.sqrt(image) * 255).astype(np.uint8)
        if encoding == "rgb":
            expected = expected[..., ::-1]
        assert output.color_format == encoding
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(samples, previous)


@pytest.mark.parametrize("sample", [0, np.finfo(np.float32).smallest_subnormal, 200])
def test_uniform_gray_is_finite_and_has_no_artificial_detail(sample):
    tool = ToneMapDurand(parameters={"output_depth": "float32"})
    result = tool.execute({"image": np.full((6, 8), sample, np.float32)})
    assert result and result.passed, tool.error
    assert result.outputs["image"].color_format == "gray"
    np.testing.assert_array_equal(result.outputs["image"], 1 if sample else 0)


@pytest.mark.parametrize(("contrast", "saturation"), [(2, 1), (4, 0.25)])
def test_portable_adjustments_change_contrast_and_saturation(bgr, contrast, saturation):
    tool = ToneMapDurand(parameters={
        "contrast": contrast, "saturation": saturation, "output_depth": "float32",
    })
    result = tool.execute({"image": bgr})
    assert result and result.passed, tool.error
    output = result.outputs["image"]
    if contrast == 2:
        assert texture_steps(output)[0] > 0.03
    else:
        blue, green, red = output[20, 16]
        assert blue / green == pytest.approx(1.1**0.25, abs=1e-6)
        assert red / green == pytest.approx(0.9**0.25, abs=1e-6)


@pytest.mark.parametrize(("settings", "image", "error"), [
    ({}, np.zeros((4, 4), np.uint16), "Unsupported image depth"),
    ({}, np.full((4, 4), -1, np.float32), "nonnegative"),
    ({}, np.full((4, 4), np.nan, np.float32), "finite"),
    ({"sigma_space": 17}, np.ones((4, 4), np.float32), "between 0.1 and 16"),
    ({"sigma_space": 16}, np.ones((700, 700), np.float32), "Reduce the spatial sigma"),
])
def test_invalid_image_parameter_and_workload(settings, image, error):
    tool = ToneMapDurand(parameters=settings)
    assert tool.execute({"image": image}) is None
    assert error in tool.error


def test_missing_native_factory_reports_backend_error(monkeypatch, bgr):
    monkeypatch.delattr(cv2.xphoto, "createTonemapDurand", raising=False)
    tool = ToneMapDurand(parameters={"backend": "opencv"})
    assert tool.execute({"image": bgr}) is None
    assert "OPENCV_ENABLE_NONFREE" in tool.error


def test_native_enabled_matches_direct_call_and_retains_texture(bgr):
    try:
        mapper = cv2.xphoto.createTonemapDurand(1.0, 4.0, 1.0, 2.0, 2.0)
    except cv2.error as exc:
        if exc.code != -213:
            raise
        tool = ToneMapDurand(parameters={"backend": "opencv"})
        assert tool.execute({"image": bgr}) is None
        assert "OPENCV_ENABLE_NONFREE" in tool.error
        return

    tool = ToneMapDurand(parameters={"backend": "opencv", "output_depth": "float32"})
    previous = bgr.copy()
    output = tool.execute({"image": bgr})
    assert output and output.passed, tool.error
    actual = output.outputs["image"]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    direct = mapper.process(rgb)
    np.testing.assert_array_equal(actual, np.clip(direct[..., ::-1], 0, 1))
    np.testing.assert_array_equal(bgr, previous)
    np.testing.assert_allclose(texture_steps(actual), [0.0266051, 0.0482728, 0.1619586], atol=1e-5)
    assert output.outputs["info"]["backend"] == "opencv"

    black = bgr.copy()
    black[:2, :5] = 0
    output = tool.execute({"image": black})
    assert output and output.passed, tool.error
    assert not output.outputs["image"][:2, :5].any()
