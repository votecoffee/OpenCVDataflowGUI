"""Qualify native RGB-D template training, registered geometry, and pose refusals."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/linemod_pose"


@pytest.fixture
def graph():
    graph = Graph.load(EXAMPLE / "linemod_pose.json", builtin_registry())
    try:
        yield graph
    finally:
        graph.close()


def _search(graph, *, configured=False):
    graph.run()
    blocks = graph.blocks
    assert blocks["match"].last_result and blocks["match"].last_result.passed, blocks["match"].error
    model = blocks["train_quarter_turn"].last_result.outputs["model"]
    image = blocks["search_image"].last_result
    depth = blocks["search_depth"].last_result
    camera = blocks["camera"].last_result.outputs["camera"]
    inputs = {
        "image": ValuePacket(image.outputs["image"], image.payloads["image"]),
        "depth": ValuePacket(depth.outputs["depth"], depth.payloads["depth"]),
        "camera": camera,
    }
    params = {"alignment": "pixel_grid"}
    if configured:
        params["trained_model"] = json.loads(json.dumps(model, allow_nan=False))
    else:
        inputs["model"] = model
    return builtin_registry().create("3d.match_linemod_pose", parameters=params), inputs


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_selects_held_out_view_and_refines_metric_translation(graph, profile):
    graph.set_run_profile(profile)
    graph.run()
    tracker = graph.blocks["match"]
    result = tracker.last_result
    assert result and result.passed, tracker.error
    truth = json.loads((EXAMPLE / "assets/ground_truth.json").read_text(encoding="utf-8"))
    pose = result.outputs["pose"]
    assert (pose["source_frame"], pose["target_frame"], pose["units"]) == (
        "marker_plate",
        "registered_optical",
        "m",
    )
    found, expected = np.asarray(pose["source_to_target"]), np.asarray(truth["search"])
    assert np.linalg.norm(found[:3, 3] - expected[:3, 3]) < 0.0015
    np.testing.assert_allclose(found[:3, :3], expected[:3, :3], atol=1e-12)
    report = result.outputs["report"]
    assert report["views"] == 2 and report["rotation_source"] == "trained_view"
    hit = report["selected"]
    assert hit["view"] == "quarter_turn"
    assert hit["similarity_percent"] >= 95
    assert hit["patch_correlation"] >= 0.90
    assert hit["anchor_inlier_fraction"] >= 0.9 and hit["reprojection_rms_px"] < 0.8
    assert result.payloads["pose"]["ticket"]["value"] == "search"
    assert bool(result.drawings) == (profile == "debug")


@pytest.mark.local_examples
def test_persisted_template_bank_and_confidence_failures(graph, tmp_path):
    block, inputs = _search(graph, configured=True)
    try:
        assert block.execute(inputs).passed
        saved = Graph()
        saved.add(block)
        path = tmp_path / "trained_pose.json"
        saved.save(path)
        persisted = Graph.load(path, builtin_registry())
        try:
            loaded = persisted.blocks[block.id]
            assert loaded.execute(inputs).passed
            loaded.parameters["maximum_reprojection_rms_px"] = 0.1
            result = loaded.execute(inputs)
            assert result and not result.passed and result.outputs["pose"] is NO_VALUE
            assert [hit["status"] for hit in result.outputs["candidates"]] == ["reprojection_limit"]
            assert result.fail_reasons[0]["code"] == "linemod_unverified"
        finally:
            persisted.close()

        block.parameters["minimum_patch_correlation"] = 0.995
        result = block.execute(inputs)
        assert not result.passed and result.outputs["pose"] is NO_VALUE
        assert result.outputs["candidates"][0]["status"] == "appearance_mismatch"

        block.parameters["minimum_patch_correlation"] = 0.65
        old = inputs["depth"].value
        missing = DepthArray(np.full(old.shape, np.nan, np.float32), "m", 0, 5)
        result = block.execute({**inputs, "depth": missing})
        assert result is not None and not result.passed and result.outputs["pose"] is NO_VALUE
    finally:
        block.close()


@pytest.mark.local_examples
def test_registration_calibration_and_ticket_checks(graph):
    block, inputs = _search(graph)
    image, depth = inputs["image"], inputs["depth"]
    try:
        block.parameters["alignment"] = "same_frame"
        assert block.execute(inputs) is None
        assert "Register depth" in block.error
        registered = ValuePacket(
            depth.value, {**image.payload, "optical_frame": inputs["camera"]["frame"]}
        )
        assert block.execute({**inputs, "depth": registered}).passed
        wrong_frame = ValuePacket(
            depth.value, {**registered.payload, "optical_frame": "unregistered"}
        )
        assert block.execute({**inputs, "depth": wrong_frame}) is None
        assert "full optical grid" in block.error
        ticket = TicketClock().issue("different_capture")
        wrong_ticket = ValuePacket(depth.value, {**registered.payload, "ticket": ticket})
        assert block.execute({**inputs, "depth": wrong_ticket}) is None
        assert "share a ticket" in block.error
        altered = copy.deepcopy(inputs["camera"])
        altered["camera_matrix"][0][0] += 10
        assert block.execute({**inputs, "depth": registered, "camera": altered}) is None
        assert "share calibrated intrinsics" in block.error
    finally:
        block.close()


@pytest.mark.local_examples
def test_invalid_serialized_features_are_rejected_before_native_matching(graph):
    block, inputs = _search(graph)
    try:
        bad = copy.deepcopy(inputs["model"])
        bad["views"][0]["templates"][0]["features"][0][2] = 8
        assert block.execute({**inputs, "model": bad}) is None
        assert "feature location or orientation" in block.error
    finally:
        block.close()
