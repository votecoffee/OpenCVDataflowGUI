"""Dense RLOF native dispatch, seed bounds and translated texture."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, convert_depth
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/dense_rlof/dense_rlof.json"
TYPE = "sequence.dense_rlof"


@pytest.fixture(autouse=True)
def deterministic_native_pool():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


@pytest.mark.parametrize("mode", ["epic", "ric", "direct"])
def test_native_parity_with_rgb16_tickets_and_selection(mode, monkeypatch):
    optflow = cv2.optflow
    p = {
        "support": "fixed",
        "solver": "standard",
        "interpolation": "epic" if mode == "direct" else mode,
        "global_motion_prior": False,
        "illumination_model": False,
        "max_level": 2,
        "iterations": 18,
        "m_estimator": True,
        "norm_sigma0": 4.0,
        "norm_sigma1": 8.0,
        "post_smoothing": False,
        "epic_neighbors": 32,
        "epic_sigma": 0.03,
        "epic_lambda": 41.0,
        "ric_superpixel_mode": "SLICO",
        "ric_superpixel_size": 12,
    }
    grid = (1, 1) if mode == "direct" else (6, 6)
    block = builtin_registry().create(
        TYPE, parameters={**p, "grid_x": grid[0], "grid_y": grid[1], "max_gap_ms": 0}
    )
    source = np.random.default_rng(555).integers(1200, 64300, (64, 80, 3), dtype=np.uint16)
    target = np.roll(source, (2, 1), axis=(0, 1))
    native_images = [convert_depth(img[..., ::-1], "uint8") for img in (source, target)]

    local = optflow.RLOFOpticalFlowParameter_create()
    local.setSupportRegionType(optflow.SR_FIXED)
    local.setSolverType(optflow.ST_STANDART)
    local.setUseGlobalMotionPrior(False)
    local.setUseIlluminationModel(False)
    local.setMaxLevel(2)
    local.setMaxIteration(18)
    local.setUseMEstimator(True)
    local.setNormSigma0(4.0)
    local.setNormSigma1(8.0)
    ric = mode == "ric"
    native = optflow.calcOpticalFlowDenseRLOF
    expected_args = (
        0.0,
        grid,
        optflow.INTERP_RIC if ric else optflow.INTERP_EPIC,
        16 if ric else 32,
        0.03,
        41.0,
        12,
        cv2.ximgproc.SLICO,
        False,
        500.0,
        1.5,
        False,
    )
    observed = []

    def solve(previous, current, flow, settings, *args):
        np.testing.assert_array_equal(previous, native_images[0])
        np.testing.assert_array_equal(current, native_images[1])
        assert flow is None and args == expected_args
        assert settings.getSolverType() == local.getSolverType()
        assert settings.getSupportRegionType() == local.getSupportRegionType()
        assert settings.getNormSigma0() == local.getNormSigma0()
        assert settings.getNormSigma1() == local.getNormSigma1()
        assert not settings.getUseGlobalMotionPrior()
        result = native(previous, current, flow, settings, *args)
        observed.append(result.copy())
        return result

    monkeypatch.setattr(optflow, "calcOpticalFlowDenseRLOF", solve)
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[12:40, 20:56] = 255
    spatial = frame([80, 64], "camera")
    clock = TicketClock("rlof")
    tickets = [clock.issue("camera", "fixed", str(index)) for index in range(2)]
    a, b = (
        ValuePacket(ImageArray(img, "rgb"), {"ticket": ticket, "spatial": spatial})
        for img, ticket in zip((source, target), tickets, strict=True)
    )

    warmup = block.execute({"image": a})
    assert warmup and warmup.outputs["flow"] is NO_VALUE
    result = block.execute({"image": b, "mask": ValuePacket(mask, {"spatial": spatial})})
    assert result and result.outputs["ready"], block.error
    vectors = result.outputs["flow"]
    selected = mask != 0
    assert len(observed) == 1
    np.testing.assert_array_equal(vectors.vectors[selected], observed[0][selected])
    expected_motion = np.linalg.norm(vectors.vectors[selected] - [1, 2], axis=1)
    assert expected_motion.mean() < 0.4
    assert not vectors.vectors[~selected].any()
    assert result.outputs["consistency_mask"] is NO_VALUE
    assert vectors.metadata["ticket"] == tickets[1] and vectors.metadata["spatial"] == spatial
    assert result.outputs["status"]["previous_ticket"] == tickets[0]
    assert block.execute({"image": b, "reset": True}).outputs["flow"] is NO_VALUE


@pytest.mark.parametrize(
    ("params", "shape", "error"),
    [
        ({}, (23, 60), "at least 40 × 30"),
        ({"small_window": 25}, (96, 120), "no larger"),
        ({"seed_consistency_threshold": 1.0}, (64, 80), "requires geodesic"),
        ({"m_estimator": True, "norm_sigma0": 8}, (64, 80), "first Hampel cutoff"),
        ({"grid_x": 100}, (40, 50), "seed grid is empty"),
        ({"grid_x": 10}, (64, 64), "neighbors exceed"),
        ({"interpolation": "ric", "ric_superpixel_size": 36}, (64, 64), "room for"),
    ],
)
def test_rlof_rejects_unsafe_or_incompatible_inputs(params, shape, error, monkeypatch):
    def unexpected(*args):
        raise AssertionError("Unsafe settings reached OpenCV")

    monkeypatch.setattr(cv2.optflow, "RLOFOpticalFlowParameter_create", unexpected)
    block = builtin_registry().create(TYPE, parameters=params)
    assert block.execute({"image": np.zeros(shape, np.uint8)}) is None
    assert error in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_translated_texture_fixed_and_cross_agreement(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    region = np.zeros((96, 120), dtype=bool)
    region[24:72, 24:96] = True
    try:
        graph.run()
        for mode in ("fixed", "cross"):
            assert graph.blocks[mode].last_result.outputs["flow"] is NO_VALUE
        graph.run()
        assert graph.sources_exhausted()
        fields = []
        for mode in ("fixed", "cross"):
            block = graph.blocks[mode]
            assert block_entry(block).path[:2] == ("OpenCV", "Optical Flow Algorithms (optflow)")
            result = block.last_result
            assert result and result.outputs["ready"], block.error
            out = result.outputs
            flow = out["flow"].vectors
            assert np.linalg.norm(flow[region] - [3, 2], axis=1).mean() < 0.004
            assert not flow[~region].any()
            np.testing.assert_array_equal(out["consistency_mask"] != 0, region)
            assert np.isnan(out["consistency_error"][~region]).all()
            assert out["visualization"].shape == (96, 120, 3)
            status = out["status"]
            assert status["previous_ticket"]["value"] == "00_reference.png"
            assert status["current_ticket"]["value"] == "01_translated.png"
            assert status["consistency"]["consistent"] == 3456
            fields.append(flow)
        assert not np.array_equal(*fields)
    finally:
        graph.close()
