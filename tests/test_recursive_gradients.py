"""Qualify signed recursive gradients with independent local references."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.ndimage import correlate1d

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/recursive_gradients"


def tool(family, **parameters):
    return builtin_registry().create(f"cv.{family}_gradient", parameters=parameters)


def test_deriche_matches_native_on_replicated_input():
    signal = np.random.default_rng(213).integers(2000, 63000, (31, 47, 3), dtype=np.uint16)
    alpha, omega = 1.4, 0.8
    block = tool("deriche", alpha=alpha, omega=omega)
    source = ImageArray(signal[..., ::-1], "rgb")
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    halo = math.ceil(16 / alpha)
    padded = cv2.copyMakeBorder(np.asarray(source), halo, halo, halo, halo, cv2.BORDER_REPLICATE)
    for key, fn in (("dx", cv2.ximgproc.GradientDericheX), ("dy", cv2.ximgproc.GradientDericheY)):
        native = fn(padded, alpha, omega)
        expected = native[halo:-halo, halo:-halo]
        output = result.outputs[key]
        assert output.dtype == np.float32 and output.color_format == "rgb"
        np.testing.assert_array_equal(output, expected)


def test_paillou_matches_truncated_analytic_impulse_response():
    source = np.random.default_rng(520).integers(6, 240, (12, 19), dtype=np.uint8)
    alpha, omega = 1.15, 0.53
    block = tool("paillou", alpha=alpha, omega=omega)
    result = block.execute({"image": source})
    assert result and result.passed, block.error

    lags = np.arange(-96, 97)
    magnitude = np.abs(lags)
    pole1, pole2 = np.exp(-alpha + omega), np.exp(-alpha - omega)
    lowpass = ((1 - pole1) * (1 - pole2)) * (
        (pole1 ** (magnitude + 1) - pole2 ** (magnitude + 1)) / (pole1 - pole2)
    )
    derivative = np.sign(lags) * lowpass
    smooth = np.exp(-alpha * magnitude) * (
        alpha * np.sinh(omega * magnitude) + omega * np.cosh(omega * magnitude)
    )
    smooth /= smooth.sum()
    for key, axis in (("dx", 1), ("dy", 0)):
        reference = correlate1d(source.astype(float), derivative, axis=axis, mode="nearest")
        reference = correlate1d(reference, smooth, axis=1 - axis, mode="nearest")
        np.testing.assert_allclose(result.outputs[key], reference, atol=5e-6, rtol=0)


@pytest.mark.parametrize("family", ["deriche", "paillou"])
def test_typed_selection_drops_alpha_and_retains_ticket_and_geometry(family):
    signal = np.random.default_rng(229).integers(15, 215, (24, 30, 4), dtype=np.uint8)
    source = ImageArray(signal, "rgba")
    source[..., 3] = 255
    block = tool(family)
    standard = block.execute({"image": source})
    assert standard and standard.passed
    spatial = frame([30, 24], "camera")
    ticket = TicketClock("gradients").issue("camera", "fixed", "part-2")
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[4:19, 8:26] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [30, 24],
        "rectangle": [12, 6, 12, 15],
    }
    source[..., 3] = 0
    selected = block.execute(
        {
            "image": ValuePacket(source, {"ticket": ticket, "spatial": spatial}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert selected and selected.passed, block.error
    area = np.zeros(source.shape[:2], bool)
    area[6:19, 12:24] = True
    for key in ("dx", "dy", "magnitude", "preview"):
        result = selected.outputs[key]
        assert result.shape == (24, 30, 3) and result.color_format == "rgb"
        assert not result[~area].any()
        np.testing.assert_array_equal(result[area], standard.outputs[key][area])
        assert selected.payloads[key]["ticket"] == ticket
        assert selected.payloads[key]["spatial"] == spatial


@pytest.mark.parametrize("family", ["deriche", "paillou"])
def test_flat_color_field_has_zero_response_and_blank_preview(family):
    signal = np.full((22, 46, 3), [98, 201, 160], dtype=np.uint8)
    result = tool(family).execute({"image": signal})
    assert result and result.passed
    for key in ("dx", "dy", "magnitude", "preview"):
        assert not result.outputs[key].any()
    invalid = tool(family, alpha=0)
    assert invalid.execute({"image": signal}) is None and "Alpha" in invalid.error
    if family == "paillou":
        unstable = tool(family, alpha=1, omega=1)
        assert unstable.execute({"image": signal}) is None and "below alpha" in unstable.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_localizes_both_polarities_on_both_axes(profile):
    graph = Graph.load(EXAMPLE / "recursive_gradients.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        deriche = graph.blocks["deriche"]
        paillou = graph.blocks["paillou"]
        assert block_entry(deriche).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        assert block_entry(paillou).path == ("Custom", "Image processing", "Edges")
        for key in ("deriche", "paillou"):
            result = graph.blocks[key].last_result
            assert result and result.passed, graph.blocks[key].error
            dx, dy = (result.outputs[name] for name in ("dx", "dy"))
            probes = (
                (dx[39], dy[39], 36, 1),
                (dx[39], dy[39], 76, -1),
                (dy[:, 115], dx[:, 115], 79, 1),
                (dy[:, 115], dx[:, 115], 100, -1),
            )
            for along, cross, first_pixel, sign in probes:
                nearby = slice(first_pixel - 3, first_pixel + 4)
                peak = int(np.argmax(abs(along[nearby]))) + nearby.start
                assert peak in (first_pixel - 1, first_pixel)
                assert along[peak] * sign > 70
                assert np.max(abs(cross[nearby])) < 0.01
            np.testing.assert_allclose(result.outputs["magnitude"], np.hypot(dx, dy))
        result = graph.blocks["selected"].last_result
        area = np.zeros((116, 154), bool)
        area[19:58, 21:91] = True
        for key in ("dx", "dy", "magnitude", "preview"):
            selected = result.outputs[key]
            np.testing.assert_array_equal(selected[area], deriche.last_result.outputs[key][area])
            assert not selected[~area].any()
    finally:
        graph.close()
