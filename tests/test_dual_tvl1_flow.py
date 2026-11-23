"""Qualify native settings, temporal initialization, and a known moving patch."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/dual_tvl1_flow"
TYPE = "sequence.dual_tvl1"


@pytest.fixture(autouse=True)
def small_native_thread_pool():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


def test_native_parity_with_normalized_float_and_previous_flow_seed():
    registry = builtin_registry()
    settings = {
        "tau": 0.21,
        "lambda_weight": 0.19,
        "theta": 0.23,
        "scales": 2,
        "warps": 2,
        "epsilon": 0.02,
        "inner_iterations": 7,
        "outer_iterations": 3,
        "scale_step": 0.61,
        "gamma": 0.02,
        "median_filter": "3",
        "initial_flow": "previous",
        "max_gap_ms": 0,
        "consistency_tolerance": 0.7,
    }
    node = registry.create(TYPE, parameters=settings)
    native = cv2.optflow.DualTVL1OpticalFlow_create(
        0.21, 0.19, 0.23, 2, 2, 0.02, 7, 3, 0.61, 0.02, 3, False
    )
    base = np.random.default_rng(721).random((52, 62), dtype=np.float32)
    images = [np.roll(base, (i, i * 2), axis=(0, 1)).copy() for i in range(3)]
    mask = np.zeros(base.shape, np.uint8)
    mask[12:40, 15:46] = 255
    selected = mask != 0
    context = frame([62, 52], "source")
    clock = TicketClock("tvl1")
    tickets = [clock.issue("camera", "fixed", str(i)) for i in range(3)]
    first = node.execute(
        {"image": ValuePacket(images[0], {"spatial": context, "ticket": tickets[0]})}
    )
    assert first and not first.outputs["ready"] and first.outputs["flow"] is NO_VALUE
    initial = None
    for step, image in enumerate(images[1:], 1):
        native.setUseInitialFlow(initial is not None)
        expected = native.calc(images[step - 1], image, initial)
        result = node.execute(
            {
                "image": ValuePacket(image, {"spatial": context, "ticket": tickets[step]}),
                "mask": ValuePacket(mask, {"spatial": context}),
            }
        )
        assert result and result.outputs["ready"], node.error
        flow = result.outputs["flow"]
        assert flow.vectors.dtype == np.float32
        np.testing.assert_array_equal(flow.vectors[selected], expected[selected])
        assert not flow.vectors[~selected].any()
        assert np.isnan(result.outputs["consistency_error"][~selected]).all()
        assert result.outputs["status"]["previous_ticket"] == tickets[step - 1]
        assert flow.metadata["ticket"] == tickets[step] and flow.metadata["spatial"] == context
        initial = expected.copy()
    reset = node.execute({"image": images[2], "reset": True})
    assert reset and not reset.outputs["ready"] and reset.outputs["flow"] is NO_VALUE


def test_validation_and_workspace_limits(monkeypatch):
    registry = builtin_registry()
    for key, value in (("scale_step", 1.0), ("median_filter", "7"), ("theta", 0)):
        node = registry.create(TYPE, parameters={key: value})
        with pytest.raises(ValidationError) as error:
            node.validate_parameters()
        assert error.value.fields == (key,)
    node = registry.create(TYPE)
    assert node.execute({"image": np.zeros((2, 20), np.uint8)}) is None
    assert "dimensions between" in node.error
    assert node.execute({"image": np.full((20, 20), 2.0, np.float32)}) is None
    assert "within [0, 1]" in node.error
    monkeypatch.setattr("opencv_dataflow.dual_tvl1_flow._MAX_NATIVE_BYTES", 8)
    assert node.execute({"image": np.zeros((20, 20), np.uint8)}) is None
    assert "workspace exceeds" in node.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_two_frame_project_compares_known_patch_motion(profile):
    graph = Graph.load(EXAMPLE / "dual_tvl1_flow.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        for key in ("dual_tvl1", "farneback"):
            result = graph.blocks[key].last_result
            assert not result.outputs["ready"] and result.outputs["flow"] is NO_VALUE
        assert not graph.sources_exhausted()
        graph.run()
        assert graph.sources_exhausted()
        for key in ("dual_tvl1", "farneback"):
            node = graph.blocks[key]
            result = node.last_result
            assert result and result.outputs["ready"], node.error
            flow = result.outputs["flow"].vectors
            interior = np.s_[34:64, 43:76]
            assert np.linalg.norm(flow[interior] - [5, -3], axis=2).mean() < 0.02
            assert (result.outputs["consistency_mask"][interior] == 255).all()
            assert np.nanmax(result.outputs["consistency_error"]) < 0.08
            assert result.outputs["status"]["consistency"]["selected"] == 990
            assert result.outputs["status"]["previous_ticket"]["value"] == "previous.png"
            assert result.outputs["status"]["current_ticket"]["value"] == "current.png"
        assert block_entry(graph.blocks["dual_tvl1"]).path[:2] == (
            "OpenCV",
            "Optical Flow Algorithms (optflow)",
        )
    finally:
        graph.close()
