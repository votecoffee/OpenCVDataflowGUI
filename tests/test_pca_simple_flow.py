"""Check native parity, image limits, and the saved translation replay."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/pca_simple_flow"
INTERIOR = np.s_[16:-16, 16:-16]


@pytest.fixture(autouse=True)
def native_thread_pool():
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(before)


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture(scope="module")
def fixture_pair():
    return tuple(
        cv2.imread(str(EXAMPLE / "assets" / name), cv2.IMREAD_GRAYSCALE)
        for name in ("00_reference.png", "01_shifted.png")
    )


@pytest.mark.parametrize("which", ("pcaflow", "simpleflow"))
def test_native_defaults_on_tagged_uint16_with_roi(registry, fixture_pair, which):
    a, b = fixture_pair
    if which == "pcaflow":
        native = cv2.optflow.createOptFlow_PCAFlow().calc(a, b, None)
    else:
        inputs = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (a, b)]
        native = cv2.optflow.createOptFlow_SimpleFlow().calc(*inputs, None)
    tool = registry.create(f"sequence.{which}", parameters={"max_gap_ms": 0})
    spatial = frame([128, 96], "test-camera")
    clock = TicketClock(which)
    first, second = (clock.issue("test-camera", "fixed", key) for key in ("first", "second"))

    def packet(image, ticket):
        rgb = np.repeat(image[:, :, None].astype(np.uint16) * 257, 3, axis=2)
        return ValuePacket(ImageArray(rgb, "rgb"), {"spatial": spatial, "ticket": ticket})

    warmup = tool.execute({"image": packet(a, first)})
    assert warmup and warmup.outputs["flow"] is NO_VALUE
    mask = np.zeros_like(a)
    mask[INTERIOR] = 255
    result = tool.execute(
        {
            "image": packet(b, second),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.outputs["ready"], tool.error
    field = result.outputs["flow"]
    assert field.vectors.dtype == np.float32
    np.testing.assert_allclose(field.vectors[INTERIOR], native[INTERIOR], rtol=0, atol=1e-6)
    assert not field.vectors[mask == 0].any()
    assert field.metadata["spatial"] == spatial and field.metadata["ticket"] == second
    assert result.outputs["status"]["previous_ticket"] == first
    assert tool.execute({"image": packet(b, second), "reset": True}).outputs["flow"] is NO_VALUE


def test_simpleflow_full_controls_match_configured_native_call(registry, fixture_pair):
    # Three distinct channels exercise color matching after tagged RGB conversion.
    def color(image):
        return cv2.merge((image, np.roll(image, 3, axis=0), np.roll(image, 6, axis=1)))

    a, b = map(color, fixture_pair)
    settings = {
        "layers": 2,
        "averaging_radius": 1,
        "search_radius": 5,
        "spatial_sigma": 2.5,
        "color_sigma": 29.0,
        "postprocess_radius": 7,
        "postprocess_spatial_sigma": 9.0,
        "postprocess_color_sigma": 33.0,
        "occlusion_threshold": 1.0,
        "upscale_radius": 5,
        "upscale_spatial_sigma": 8.0,
        "upscale_color_sigma": 31.0,
        "speed_up_threshold": 0.0,
        "max_gap_ms": 0,
    }
    expected = cv2.optflow.calcOpticalFlowSF(
        a, b, 2, 1, 5, 2.5, 29.0, 7, 9.0, 33.0, 1.0, 5, 8.0, 31.0, 0.0
    )
    tool = registry.create("sequence.simpleflow", parameters=settings)
    assert tool.execute({"image": ImageArray(a[..., ::-1], "rgb")})
    result = tool.execute({"image": ImageArray(b[..., ::-1], "rgb")})
    assert result and result.outputs["ready"], tool.error
    np.testing.assert_allclose(result.outputs["flow"].vectors, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "which,shape,params,message",
    [
        ("pcaflow", (56, 71), {}, "requires at least"),
        ("simpleflow", (5, 8), {"layers": 6}, "too many levels"),
        ("simpleflow", (96, 128), {"search_radius": 32, "averaging_radius": 32}, "budget"),
        ("simpleflow", (96, 128), {"postprocess_color_sigma": 0}, "Postprocess color sigma"),
    ],
)
def test_limits_are_reported_before_a_native_solve(registry, which, shape, params, message):
    tool = registry.create(f"sequence.{which}", parameters=params)
    assert tool.execute({"image": np.zeros(shape, np.uint8)}) is None
    assert message in tool.error
    assert not tool.memory["streams"]


def test_pcaflow_textureless_error_restarts_state(registry, fixture_pair):
    tool = registry.create("sequence.pcaflow", parameters={"max_gap_ms": 0})
    flat = np.full((96, 128), 100, np.uint8)
    assert tool.execute({"image": flat})
    assert tool.execute({"image": flat}) is None
    assert "trackable texture" in tool.error
    clean = tool.execute({"image": fixture_pair[0]})
    assert clean and not clean.outputs["ready"]


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_replay_has_known_displacement(registry, profile):
    graph = Graph.load(EXAMPLE / "pca_simple_flow.json", registry)
    graph.set_run_profile(profile)
    try:
        graph.run()
        for key in ("pcaflow", "simpleflow"):
            assert graph.blocks[key].last_result.outputs["flow"] is NO_VALUE
        graph.run()
        assert graph.sources_exhausted()
        for key, tolerance in (("pcaflow", 0.06), ("simpleflow", 0.005)):
            node = graph.blocks[key]
            assert block_entry(node).path[:2] == ("OpenCV", "Optical Flow Algorithms (optflow)")
            result = node.last_result
            assert result and result.outputs["ready"], node.error
            status = result.outputs["status"]
            assert status["previous_ticket"]["value"] == "00_reference.png"
            assert status["current_ticket"]["value"] == "01_shifted.png"
            vectors = result.outputs["flow"].vectors
            assert np.linalg.norm(vectors[INTERIOR] - [3, -2], axis=2).mean() < tolerance
            assert np.count_nonzero(result.outputs["consistency_mask"][INTERIOR]) > 5_000
    finally:
        graph.close()
