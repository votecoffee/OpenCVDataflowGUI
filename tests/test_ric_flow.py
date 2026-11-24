"""Controlled geometry, parameter handling and process isolation for native RIC."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import opencv_dataflow.ric_flow as ric
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/ric_correspondences"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def load_graph(registry):
    return Graph.load(EXAMPLE / "ric_correspondences.json", registry)


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_preserves_the_analytic_seam_and_withholds_unsupported_warp(registry, profile):
    graph = load_graph(registry)
    graph.set_run_profile(profile)
    try:
        graph.run()
        node = graph.blocks["ric"]
        assert block_entry(node).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        assert node.state == "pass", node.error
        result = node.last_result
        flow = result.outputs["flow"].vectors
        mask = result.outputs["support"] != 0
        yy, xx = np.indices(mask.shape)
        truth = np.where(xx < 88, 7, -6) + 0.012 * (yy - 56)
        seam = mask & (yy >= 26) & (yy < 88) & (xx >= 80) & (xx < 96)
        assert seam.sum() == 992
        error = np.hypot(flow[..., 0], flow[..., 1] - truth)
        assert error[seam].max() < 0.001
        assert error[mask].max() < 0.001
        assert not mask[8:30, 157:172].any()
        report = result.outputs["report"]
        assert report["method"] == "ric"
        assert report["seeds"] == 206 and report["control_max_error_pixels"] > 10
        assert report["supported_pixels"] == mask.sum() == 16603
        assert bool(result.drawings) == (profile == "debug")
        warp = graph.blocks["warp"]
        assert warp.state == "pass", warp.error
        np.testing.assert_array_equal(warp.last_result.outputs["coverage"] != 0, mask)
        assert not warp.last_result.outputs["image"][~mask].any()
    finally:
        graph.close()


@pytest.mark.parametrize("mode", ("slic", "slico", "mslic"))
def test_grayscale_and_cost_map_can_use_both_native_refinement_stages(registry, mode):
    graph = load_graph(registry)
    try:
        settings = graph.blocks["ric"].parameters
        query, train = (
            cv2.imread(str(EXAMPLE / "assets" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            for name in ("query", "train")
        )
        size = list(query.shape[::-1])
        source_context = {"spatial": frame(size, "query")}
        target_context = {"spatial": frame(size, "train")}
        record = {
            "query_points": settings["source_points"],
            "train_points": settings["target_points"],
            "query_context": source_context,
            "train_context": target_context,
        }
        block = registry.create(
            "cv.ric_flow",
            parameters={
                "superpixel_mode": mode, "superpixel_neighbors": 48, "post_smoothing": True,
                "variational_refinement": True,
            },
        )
        selection = np.zeros(query.shape, np.uint8)
        selection[28:72, 20:74] = 255
        cost = np.full(query.shape, 0.02, np.float32)
        cost[:, 87:89] = 1
        inputs = {
            "image": ValuePacket(query, source_context), "target": ValuePacket(train, target_context),
            "matches": record, "cost_map": ValuePacket(cost, source_context),
            "mask": ValuePacket(selection, source_context),
        }
        threads = cv2.getNumThreads()
        result = block.execute(inputs)
        assert result and result.passed, block.error
        assert cv2.getNumThreads() == threads
        assert result.outputs["flow"].metadata["flow_matches"] == {
            "source": source_context, "target": target_context,
        }
        np.testing.assert_array_equal(result.outputs["support"], selection)
        vector = result.outputs["flow"].vectors
        assert np.median(vector[selection != 0, 1]) == pytest.approx(6.93, abs=0.2)
        assert not vector[selection == 0].any()
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("case", "message"),
    [("same_pixel", "share a source seed"), ("too_few", "Only 4 distinct"),
     ("region_size", "at least two superpixels"), ("float32_edge", "outside the float32")],
)
def test_invalid_controls_do_not_start_the_worker(registry, monkeypatch, case, message):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid input reached the native worker")

    monkeypatch.setattr(ric, "ManagedProcess", unexpected)
    parameters = {"neighbors": 4, "superpixel_neighbors": 4}
    points = [[8, 8], [16, 8], [8, 16], [16, 16]]
    target = [[8, 9], [16, 9], [8, 17], [16, 17]]
    if case == "same_pixel":
        points[-1] = [8.8, 8.8]
    if case == "too_few":
        parameters["superpixel_neighbors"] = 5
    if case == "region_size":
        parameters["superpixel_size"] = 33
    if case == "float32_edge":
        points[-1] = [64 - 1e-8, 16]
    a = np.zeros((64, 64), np.uint8)
    block = registry.create("cv.ric_flow", parameters=parameters)
    assert block.execute({"image": a, "target": a, "source_points": points, "target_points": target}) is None
    assert message in block.error


@pytest.mark.parametrize(
    ("worker_body", "message", "timeout"),
    [("raise SystemExit(19)", "RIC worker failed", 10), ("import time; time.sleep(10)", "timed out", 0.2)],
)
def test_worker_failure_and_timeout_stay_inside_the_block(
    registry, tmp_path, monkeypatch, worker_body, message, timeout
):
    path = tmp_path / "failed_worker.py"
    path.write_text(worker_body, encoding="utf-8")
    monkeypatch.setattr(ric, "WORKER_SCRIPT", path)
    graph = load_graph(registry)
    try:
        block = graph.blocks["ric"]
        block.parameters["timeout_seconds"] = timeout
        graph.run()
        assert block.state == "error"
        assert message in block.error
        assert not block.current_emissions
    finally:
        graph.close()
