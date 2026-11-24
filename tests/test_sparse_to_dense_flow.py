"""Check native parity, sparse support validation, and known motion boundaries."""

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

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/sparse_to_dense_flow"
TYPE = "sequence.sparse_to_dense"


@pytest.fixture(autouse=True)
def local_native_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


@pytest.mark.parametrize("color", [False, True])
def test_native_parity_and_ticketed_selection(color):
    shape = (80, 96, 3) if color else (80, 96)
    base = np.random.default_rng(451).integers(8, 240, shape, np.uint8)
    first = cv2.GaussianBlur(base, (0, 0), 1.1)
    next_frame = np.roll(first, (2, 3), axis=(0, 1))
    settings = {
        "grid_step": 6,
        "neighbors": 32,
        "sigma": 0.08,
        "post_smoothing": color,
        "fgs_lambda": 230.0,
        "fgs_sigma": 2.4,
        "consistency_tolerance": 0.5,
        "max_gap_ms": 0,
    }
    expected = cv2.optflow.calcOpticalFlowSparseToDense(
        first,
        next_frame,
        None,
        6,
        32,
        0.08,
        color,
        230.0,
        2.4,
    )
    registry = builtin_registry()
    tool = registry.create(TYPE, parameters=settings)
    spatial = frame([96, 80], "test-camera")
    clock = TicketClock("sparse-to-dense")
    a, b = (clock.issue("test-camera", "fixed", name) for name in ("first", "second"))

    def packet(image, ticket):
        if color:
            image = ImageArray(image[..., ::-1].astype(np.uint16) * 257, "rgb")
        else:
            image = image.astype(np.float32) / 255
        return ValuePacket(image, {"spatial": spatial, "ticket": ticket})

    start = tool.execute({"image": packet(first, a)})
    assert start and not start.outputs["ready"] and start.outputs["flow"] is NO_VALUE
    mask = np.zeros((80, 96), np.uint8)
    mask[12:61, 19:70] = 255
    result = tool.execute(
        {
            "image": packet(next_frame, b),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.outputs["ready"], tool.error
    selected = mask != 0
    field = result.outputs["flow"]
    np.testing.assert_allclose(field.vectors[selected], expected[selected], rtol=0, atol=2e-6)
    assert not field.vectors[~selected].any()
    assert np.isnan(result.outputs["consistency_error"][~selected]).all()
    assert field.metadata["ticket"] == b and field.metadata["spatial"] == spatial
    assert result.outputs["status"]["previous_ticket"] == a
    reset = tool.execute({"image": packet(next_frame, b), "reset": True})
    assert reset and reset.outputs["flow"] is NO_VALUE


def test_untrackable_image_is_rejected_before_interpolation(monkeypatch):
    block = builtin_registry().create(TYPE, parameters={"neighbors": 16})
    image = np.full((64, 64), 109, np.uint8)
    first = block.execute({"image": image})
    assert first and not first.outputs["ready"]

    def unexpected_solver():
        pytest.fail("The native interpolator was created with insufficient tracks")

    monkeypatch.setattr(cv2.ximgproc, "createEdgeAwareInterpolator", unexpected_solver)
    assert block.execute({"image": image}) is None
    assert "Only 0 valid sparse matches for 16 neighbors" in block.error
    assert not block.memory["streams"]


@pytest.mark.parametrize(
    "shape,settings,message",
    [
        ((1, 64), {}, "pixels per axis"),
        ((32, 32), {}, "enough candidates"),
        ((384, 384), {"grid_step": 2}, "fewer than 32,767"),
        ((64, 64), {"sigma": 0.0001}, "Sigma"),
    ],
)
def test_grid_and_controls_fail_before_native_tracking(shape, settings, message):
    block = builtin_registry().create(TYPE, parameters=settings)
    assert block.execute({"image": np.zeros(shape, np.uint8)}) is None
    assert message in block.error
    assert not block.memory["streams"]


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_replay_on_opposing_motion(profile):
    graph = Graph.load(EXAMPLE / "sparse_to_dense_flow.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        for key in ("default", "coarse"):
            assert graph.blocks[key].last_result.outputs["flow"] is NO_VALUE
        graph.run()
        assert graph.sources_exhausted()
        for key in ("default", "coarse"):
            node = graph.blocks[key]
            assert block_entry(node).path[:2] == ("OpenCV", "Optical Flow Algorithms (optflow)")
            result = node.last_result
            assert result and result.outputs["ready"], node.error
            assert result.outputs["status"]["previous_ticket"]["value"] == "00_reference.png"
            assert result.outputs["status"]["current_ticket"]["value"] == "01_shifted.png"
            vectors = result.outputs["flow"].vectors
            left_error = np.linalg.norm(vectors - [0, 4], axis=2)
            right_error = np.linalg.norm(vectors - [0, -3], axis=2)
            assert left_error[18:-18, 16:66].mean() < 0.04
            assert right_error[18:-18, 110:-16].mean() < 0.02
            assert right_error[18:-18, 88:96].mean() < 0.15
            dark_strip_error = left_error[18:-18, 80:88].mean()
            if key == "default":
                assert dark_strip_error < 0.15
            else:
                assert dark_strip_error > 5
    finally:
        graph.close()
