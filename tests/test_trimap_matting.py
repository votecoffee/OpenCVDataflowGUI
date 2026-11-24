"""Verify the matting system, hard trimap constraints, and saved gasket example."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/custom tools/trimap_matting"
TYPE = "cv.trimap_matting"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def dense_reference(guide, trimap, radius, epsilon):
    """Assemble the full covariance-form Laplacian for a tiny independent check."""
    height, width = trimap.shape
    side = 2 * radius + 1
    count = side**2
    g = np.arange(height * width).reshape(height, width)
    laplacian = np.zeros((g.size, g.size))
    for y in range(height - side + 1):
        for x in range(width - side + 1):
            points = g[y : y + side, x : x + side].ravel()
            colors = guide[y : y + side, x : x + side].reshape(count, -1)
            offsets = colors - colors.mean(axis=0)
            covariance = np.cov(colors, rowvar=False, bias=True)
            inverse = np.linalg.inv(covariance + np.eye(guide.shape[2]) * epsilon / count)
            local = np.eye(count) - (1 + offsets @ inverse @ offsets.T) / count
            laplacian[np.ix_(points, points)] += local
    unknown = np.flatnonzero(trimap.ravel() == 128)
    foreground = np.flatnonzero(trimap.ravel() == 255)
    alpha = np.zeros(g.size)
    alpha[foreground] = 1
    alpha[unknown] = np.linalg.solve(
        laplacian[np.ix_(unknown, unknown)], -laplacian[np.ix_(unknown, foreground)].sum(axis=1)
    )
    return np.clip(alpha.reshape(trimap.shape), 0, 1)


@pytest.mark.parametrize("radius", (1, 2))
def test_matches_independent_covariance_form(registry, radius):
    rng = np.random.default_rng(45809)
    guide = rng.uniform(0.05, 0.95, (6, 8, 3))
    trimap = np.full(guide.shape[:2], 128, np.uint8)
    trimap[:2, :3] = 0
    trimap[-2:, -3:] = 255
    epsilon = 2e-5
    tool = registry.create(
        TYPE,
        parameters={
            "window_radius": radius,
            "color_regularization": epsilon,
            "solver_tolerance": 1e-10,
        },
    )
    result = tool.execute({"image": ImageArray(guide, "rgb"), "trimap": trimap})
    assert result and result.passed, tool.error
    reference = dense_reference(guide, trimap, radius, epsilon)
    np.testing.assert_allclose(result.outputs["alpha"], reference, rtol=0, atol=4e-8)
    assert result.outputs["report"]["relative_residual"] < 1e-10


def test_binary_shortcut_selected_alpha_and_payload(registry):
    color = ImageArray(np.full((2, 3, 4), (4000, 8000, 12000, 32768), np.uint16), "rgba")
    trimap = np.array([[0, 255, 0], [255, 255, 255]], np.uint8)
    mask = np.array([[255, 255, 255], [255, 0, 255]], np.uint8)
    spatial = frame([3, 2], "camera")
    ticket = TicketClock("trimap").issue("camera", "fixed", "image")
    meta = {"spatial": spatial, "ticket": ticket, "alpha_representation": "premultiplied"}
    tool = registry.create(TYPE, parameters={"trimap_alignment": "same_frame"})
    inputs = {
        "image": ValuePacket(color, meta),
        "trimap": ValuePacket(trimap, {"spatial": spatial}),
        "mask": ValuePacket(mask, {"spatial": spatial}),
    }
    result = tool.execute(inputs)
    assert result and result.passed, tool.error
    expected = (trimap == 255) & (mask != 0)
    np.testing.assert_array_equal(result.outputs["alpha"], expected)
    cutout = result.outputs["cutout"]
    assert cutout.dtype == np.uint16 and image_format(cutout) == "rgba"
    np.testing.assert_array_equal(cutout[..., 3], expected.astype(np.uint16) * 65535)
    np.testing.assert_array_equal(cutout[0, 1, :3], [8000, 16000, 24000])
    assert result.payloads["cutout"]["alpha_representation"] == "straight"
    alpha_payload = result.payloads["alpha"]
    assert alpha_payload["ticket"] == ticket and alpha_payload["spatial"] == spatial
    assert "alpha_representation" not in alpha_payload
    assert result.outputs["report"]["iterations"] == 0
    inputs["trimap"] = ValuePacket(trimap, {"spatial": frame([3, 2], "other")})
    assert tool.execute(inputs) is None
    assert "trimap coordinate frame" in tool.error


@pytest.mark.parametrize(
    "case,message",
    [
        ("depth", "uint8 grayscale"),
        ("missing_seed", "known foreground and background"),
        ("window", "dimensions must fit"),
        ("work", "work limit"),
        ("iteration_budget", "did not converge"),
    ],
)
def test_invalid_trimap_and_solve_limits(registry, monkeypatch, case, message):
    image = cv2.imread(str(EXAMPLE / "assets/composite.png"))
    trimap = cv2.imread(str(EXAMPLE / "assets/trimap.png"), cv2.IMREAD_GRAYSCALE)
    settings = {}
    if case == "depth":
        trimap = trimap.astype(np.uint16)
    if case == "missing_seed":
        trimap[trimap == 255] = 128
    if case == "window":
        image = image[:2, :3]
        trimap = np.array([[0, 12, 255], [0, 128, 255]], np.uint8)
    if case == "work":
        monkeypatch.setattr("opencv_dataflow.trimap_matting._MAX_WINDOW_ENTRIES", 100)
    if case == "iteration_budget":
        settings["max_iterations"] = 1
    tool = registry.create(TYPE, parameters=settings)
    assert tool.execute({"image": image, "trimap": trimap}) is None
    assert message in tool.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_gasket_recovers_analytic_opacity(registry, profile):
    graph = Graph.load(EXAMPLE / "trimap_matting.json", registry)
    graph.set_run_profile(profile)
    trimap = cv2.imread(str(EXAMPLE / "assets/trimap.png"), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(EXAMPLE / "assets/truth.png"), cv2.IMREAD_UNCHANGED) / 65535
    try:
        graph.run()
        node = graph.blocks["matte"]
        assert block_entry(node).path == ("Custom", "Image processing", "Segmentation")
        result = node.last_result
        assert result and result.passed, node.error
        alpha = result.outputs["alpha"]
        report = result.outputs["report"]
        assert alpha.dtype == np.float32 and alpha.shape == trimap.shape
        assert report["unknown_pixels"] == 3908
        assert report["relative_residual"] <= 1.01e-7
        assert np.all(alpha[trimap == 0] == 0) and np.all(alpha[trimap == 255] == 1)
        error = np.abs(alpha[trimap == 128] - reference[trimap == 128])
        assert error.mean() < 0.001 and np.quantile(error, 0.95) < 0.004
        assert result.payloads["alpha"]["spatial"]["root_id"] == "image:image"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()
