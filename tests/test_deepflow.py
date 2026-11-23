"""Qualify the native DeepFlow adapter and known visibility regions."""

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

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/deepflow"
TYPE = "sequence.deepflow"
VISIBLE = np.s_[36:65, 41:55]
HIDDEN = np.s_[33:70, 63:93]


@pytest.fixture(autouse=True)
def small_native_thread_pool():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


def test_native_parity_with_uint16_rgb_and_ticketed_output_selection():
    registry = builtin_registry()
    block = registry.create(TYPE, parameters={"max_gap_ms": 0, "consistency_tolerance": 0.25})
    source = np.random.default_rng(3529).integers(2000, 54000, (58, 72), dtype=np.uint16)
    target = np.roll(source, (2, -1), axis=(0, 1))

    def to_rgb(image):
        return ImageArray(np.repeat(image[..., None], 3, axis=2), "rgb")

    native_inputs = [
        (i.astype(np.float64) / 65535).astype(np.float32) * np.float32(255)
        for i in (source, target)
    ]
    native = cv2.optflow.createOptFlow_DeepFlow()
    expected = native.calc(*native_inputs, None)
    spatial = frame([72, 58], "camera")
    clock = TicketClock("deepflow")
    a, b = [clock.issue("camera", "fixed", str(i)) for i in range(2)]
    first = block.execute({"image": ValuePacket(to_rgb(source), {"spatial": spatial, "ticket": a})})
    assert first and not first.outputs["ready"] and first.outputs["flow"] is NO_VALUE
    mask = np.zeros(source.shape, np.uint8)
    mask[11:47, 15:54] = 255
    result = block.execute(
        {
            "image": ValuePacket(to_rgb(target), {"spatial": spatial, "ticket": b}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.outputs["ready"], block.error
    flow = result.outputs["flow"]
    selected = mask > 0
    np.testing.assert_allclose(flow.vectors[selected], expected[selected], rtol=0, atol=1e-6)
    assert not flow.vectors[~selected].any()
    assert np.isnan(result.outputs["consistency_error"][~selected]).all()
    assert flow.metadata["spatial"] == spatial and flow.metadata["ticket"] == b
    assert result.outputs["status"]["previous_ticket"] == a
    reset = block.execute({"image": to_rgb(target), "reset": True})
    assert reset and not reset.outputs["ready"] and reset.outputs["flow"] is NO_VALUE


def test_runtime_validation_before_native_solve(monkeypatch):
    block = builtin_registry().create(TYPE)
    assert block.execute({"image": np.zeros((2, 32), np.uint8)}) is None
    assert "dimensions between" in block.error
    assert block.execute({"image": np.full((32, 32), -0.1, np.float32)}) is None
    assert "within [0, 1]" in block.error
    monkeypatch.setattr("opencv_dataflow.deepflow._MAX_NATIVE_BYTES", 8)
    assert block.execute({"image": np.zeros((32, 32), np.uint8)}) is None
    assert "workspace exceeds" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_sequence_distinguishes_visible_and_hidden_points(profile):
    graph = Graph.load(EXAMPLE / "deepflow.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        node = graph.blocks["flow"]
        assert block_entry(node).path[:2] == ("OpenCV", "Optical Flow Algorithms (optflow)")
        assert not node.last_result.outputs["ready"]
        assert node.last_result.outputs["consistency_mask"] is NO_VALUE
        graph.run()
        before = node.last_result
        assert before and before.outputs["ready"], node.error
        assert (
            np.linalg.norm(before.outputs["flow"].vectors[VISIBLE] - [5, -2], axis=2).mean() < 0.05
        )
        assert (before.outputs["consistency_mask"][VISIBLE] == 255).all()
        graph.run()
        assert graph.sources_exhausted()
        after = node.last_result
        assert after and after.outputs["ready"], node.error
        status = after.outputs["status"]
        assert status["previous_ticket"]["value"] == "01_shifted.png"
        assert status["current_ticket"]["value"] == "02_occluded.png"
        flow = after.outputs["flow"].vectors
        assert np.linalg.norm(flow[VISIBLE] - [5, -2], axis=2).mean() < 0.065
        mask = after.outputs["consistency_mask"]
        error = after.outputs["consistency_error"]
        assert (mask[VISIBLE] == 255).all()
        hidden = mask[HIDDEN]
        assert hidden.size == 1110 and 0.35 < (hidden == 0).mean() < 0.75
        np.testing.assert_array_equal(hidden == 0, error[HIDDEN] > 0.05)
    finally:
        graph.close()
