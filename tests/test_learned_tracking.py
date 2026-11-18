"""Native score/box evidence, guarded state and shared tracker lifecycle contracts."""

import copy
import gc
import hashlib
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.learned_tracking import VitSequence
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.sequence import share_native_sequence_runtime
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_tracking"
MODEL = BASE / "assets/vit.onnx"


def tool(**parameters):
    return VitSequence(
        parameters={
            "model_path": str(MODEL),
            "initial_bounds": [92, 133, 43, 43],
            "warmup": 3,
            **parameters,
        }
    )


def image(index=0, encoding="bgr"):
    array = cv2.imread(str(BASE / "assets" / f"frame_{index:02d}.png"))
    if encoding in ("rgb", "rgba"):
        array = array[..., ::-1]
    if encoding in ("rgba", "bgra"):
        array = np.dstack((array, np.full(array.shape[:2], 123, np.uint8)))
    if encoding == "gray":
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return ImageArray(array, encoding)


def execute(block, value=None, **inputs):
    result = block.execute({"image": image() if value is None else value, **inputs})
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "gray"])
@pytest.mark.parametrize("engine", ["classic", "new"])
def test_native_localization_score_current_ticket_pose_graphics_and_warmup(encoding, engine):
    block = tool(engine=engine)
    clock = TicketClock("camera")
    try:
        for index in range(7):
            source = image(index, encoding)
            before = source.copy()
            ticket = clock.issue("camera")
            spatial = frame([360, 270], "camera")
            payload = {
                "ticket": ticket,
                "spatial": spatial,
                "aux": {"lot": "A"},
                "pyramid": {
                    "level": 1,
                    "largest_size": [720, 540],
                    "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
                },
            }
            result = execute(block, ValuePacket(source, payload))
            np.testing.assert_array_equal(source, before)
            np.testing.assert_array_equal(result.outputs["image"], before)
            assert result.outputs["image"].color_format == encoding
            assert result.outputs["ready"] == (index >= 2)
            assert result.passed and result.outputs["receipt"]["configuration_verified"] is False
            assert result.outputs["receipt"]["requested_availability_scope"] == "requested_cpu_only"
            if index == 0:
                assert result.outputs["tracking_score"] is NO_VALUE and result.score is None
            else:
                assert (
                    0.7 < result.outputs["tracking_score"] <= 1
                    and result.score == result.outputs["tracking_score"]
                )
            if index >= 2:
                bounds = result.outputs["bounds"]
                center = np.array(bounds[:2]) + (np.array(bounds[2:]) - 1) / 2
                expected = np.array([92 + 2 * index, 133 + index]) + 21
                assert np.linalg.norm(center - expected) < 3
                assert result.outputs["pose"]["track_id"] == result.outputs["status"]["track_id"]
                assert result.outputs["tracked_region"]["coordinate_context"] == spatial
                for key in payload:
                    assert result.payloads["bounds"][key] == payload[key]
            else:
                assert result.outputs["pose"] is NO_VALUE and result.outputs["bounds"] is NO_VALUE
            assert result.drawings and result.outputs["overlay"] is NO_VALUE
    finally:
        block.close()


def test_native_score_loss_expiry_and_explicit_reseed_never_emit_stale_boxes():
    block = tool(max_misses=2, minimum_contrast=0.0)
    blank = np.zeros((270, 360, 3), np.uint8)
    try:
        first = execute(block)
        identity = first.outputs["status"]["track_id"]
        for i in (1, 2):
            execute(block, image(i))
        for i in range(2):
            result = execute(block, blank)
            assert not result.passed and not result.outputs["ready"]
            assert result.outputs["status"]["reason"] == "tracking_score_below_threshold"
            assert result.outputs["bounds"] is NO_VALUE and result.outputs["pose"] is NO_VALUE
            assert 0 < result.outputs["tracking_score"] < 0.2
        result = execute(block)
        assert not result.passed and result.outputs["status"]["phase"] == "seed_required"
        assert result.outputs["tracking_score"] is NO_VALUE and result.score is None
        assert result.outputs["receipt"]["model_active"] is False
        assert result.outputs["receipt"]["native_updates"] == 0
        result = execute(block, seed=[92, 133, 43, 43])
        assert result.outputs["status"]["phase"] == "initialized" and result.passed
        assert result.outputs["status"]["track_id"] != identity and not result.outputs["ready"]
        execute(block)
        assert execute(block).outputs["ready"]
    finally:
        block.close()


def test_reset_stream_grid_recipe_and_runtime_copy_contracts():
    graph = Graph()
    block = graph.add(tool())
    registry = builtin_registry()
    try:
        a = execute(block, stream="A")
        execute(block, stream="A")
        shared = Graph.from_dict(graph.to_dict(), registry)
        share_native_sequence_runtime(shared, graph)
        assert shared.blocks[block.id].memory is block.memory
        independent = registry.create(block.type_id, parameters=block.parameters)
        independent.copy_runtime_from(block)
        assert not independent.memory["streams"]
        snapshot = block.snapshot_runtime()
        assert all("native" not in s for s in snapshot["streams"].values())
        current = execute(block, stream="A", reset=True)
        assert current.outputs["status"]["generation"] != a.outputs["status"]["generation"]
        assert current.outputs["status"]["phase"] == "initialized"
        other = execute(block, stream="B")
        assert other.outputs["status"]["stream"] == "B"
        block.parameters["tracking_score_threshold"] = 0.3
        changed = execute(block, stream="A")
        assert changed.outputs["status"]["phase"] == "initialized"
        ctx = {
            "spatial": frame([360, 270], "new-grid"),
            "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        }
        changed = execute(block, ValuePacket(image(), ctx), stream="A")
        assert changed.outputs["status"]["phase"] == "initialized"
        assert len(block.memory["streams"]) == 2
        independent.close()
        shared.close()
    finally:
        graph.close()


def test_native_state_has_no_reference_cycle_and_limits_are_enforced():
    block = tool(max_streams=1, overflow="evict_oldest")
    try:
        execute(block, stream="A")
        state = next(iter(block.memory["streams"].values()))
        ref = weakref.ref(state["native"])
        execute(block, stream="B")
        del state
        gc.collect()
        assert ref() is None and len(block.memory["streams"]) == 1
        assert block.memory["evicted"] == 1
        block.close()
        assert not block.memory["streams"]
    finally:
        block.close()


def test_cpu_never_enumerates_gpu_targets_and_raw_model_changes_cannot_reuse_old_native(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cv2.dnn, "getAvailableTargets", lambda *a: pytest.fail("CPU must not enumerate GPU")
    )
    raw = MODEL.read_bytes()
    path = tmp_path / "model.onnx"
    path.write_bytes(raw)
    block = tool(model_path=str(path))
    try:
        first = execute(block)
        old = first.outputs["status"]["generation"]
        path.write_bytes(b"invalid changed ONNX")
        assert block.execute({"image": image(1)}) is None and not block.current_emissions
        assert not block.memory["streams"] and block._snapshot is None
        path.write_bytes(raw)
        current = execute(block)
        assert current.outputs["status"]["generation"] != old
        assert current.outputs["receipt"]["artifact"]["sha256"] == hashlib.sha256(raw).hexdigest()
    finally:
        block.close()


def test_lean_and_disabled_observation_keep_tracking_state_and_current_image():
    block = tool()
    try:
        identity = execute(block).outputs["status"]["track_id"]
        block.run_profile = PROFILES["lean"]
        result = execute(block, image(1))
        assert result.outputs["status"]["track_id"] == identity
        result = execute(block, image(2))
        assert result.outputs["ready"]
        assert result.outputs["overlay"] is NO_VALUE and not result.drawings and not result.graphics
        block.parameters["receipt_mode"] = "off"
        result = execute(block)
        assert result.outputs["receipt"] is NO_VALUE
    finally:
        block.close()


class BrokenScore:
    def __init__(self, score):
        self.score = score

    def init(self, *args):
        return None

    def update(self, *args):
        return True, (92, 133, 43, 43)

    def getTrackingScore(self):
        return self.score


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0, 1.5])
def test_invalid_native_scores_are_contained_and_never_clipped(score, monkeypatch):
    monkeypatch.setattr(cv2.TrackerVit, "create", lambda *a: BrokenScore(score))
    block = tool()
    try:
        execute(block)
        assert block.execute({"image": image(1)}) is None and "score" in block.error
        assert not block.memory["streams"] and not block.current_emissions
    finally:
        block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"mean": [0]},
        {"std": [1, 0, 1]},
        {"tracking_score_threshold": 2.0},
        {"initial_bounds": [92, 133, 2, 43]},
        {"initial_bounds": [0, 0, 8.5, 8]},
        {"max_misses": True},
        {"warmup": 0},
        {"fusion": 1},
        {"draw_overlay": 1},
        {"engine": "new", "target": "opencl"},
        {"receipt_mode": "detailed"},
        {"maximum_source_pixels": 1},
        {"maximum_patch_side": 64},
        {"max_bytes_mb": 0.01},
    ],
)
def test_parameter_resource_failures_have_no_output_and_recover(parameters):
    block = tool(**parameters)
    try:
        assert block.execute({"image": image()}) is None and block.error
        assert not block.current_emissions and not block.memory["streams"]
        good = tool()
        block.parameters = copy.deepcopy(good.parameters)
        good.close()
        assert execute(block).outputs["status"]["phase"] == "initialized"
    finally:
        block.close()
