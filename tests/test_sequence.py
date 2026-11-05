"""Temporal behavior, portable references and worker/display memory isolation."""

import copy
import json
import time

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QInputDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import sequence_graph
from opencv_dataflow.image_assets import decode_image, encode_image, seal
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parallel_inspection import ParallelInspect
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, describe_value
from opencv_dataflow.poses import local_offset, make_pose
from opencv_dataflow.queued_execution import QueuedExecution, check_upstream
from opencv_dataflow.recipe_editor import RecipeManager
from opencv_dataflow.recipes import RecipeCatalog, reference
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolblocks import Binding, ToolBlock


def make(kind="image_memory", **parameters):
    return builtin_registry().create(f"sequence.{kind}", parameters=parameters)


def frame(block, value=10, **inputs):
    image = np.full((48, 64), value, np.uint8) if np.isscalar(value) else value
    block.execute({"image": image, **inputs})
    assert not block.error, block.error
    return block.last_result.outputs


def texture():
    return np.random.default_rng(5).integers(0, 256, (96, 128), dtype=np.uint8)


def test_warmup_suppresses_primary_events_and_reports_progress():
    block = make(warmup=3)
    for index in range(1, 4):
        out = frame(block, index * 10)
        assert out["activations"] == index
        assert out["ready"] == (index == 3)
        assert bool(block.current_emissions["reference"]) == (index == 3)
        assert "status" in block.current_emissions
        assert block.state == ("pass" if index == 3 else "waiting")
        if index < 3:
            assert out["reference"] is NO_VALUE
    np.testing.assert_array_equal(out["reference"], 20)
    out = frame(block, 100)
    np.testing.assert_array_equal(out["reference"], 20)
    np.testing.assert_array_equal(out["difference"], 80)
    assert out["status"]["learning_activations"] == 3


@pytest.mark.parametrize("mode, expected", [("first", 10), ("latest", 30), ("mean", 20)])
def test_reference_modes_and_learning_gate(mode, expected):
    block = make(mode=mode, warmup=3)
    assert not frame(block, 80, learn=False)["ready"]
    for value in (10, 20, 30):
        out = frame(block, value)
    np.testing.assert_array_equal(out["reference"], expected)
    assert out["status"]["activations"] == 4
    assert out["status"]["learning_activations"] == 3


def test_ticket_progress_and_explicit_reset():
    block, clock = make(warmup=2), TicketClock()
    for index in range(3):
        ticket = clock.issue("camera")
        out = frame(block, ValuePacket(np.zeros((48, 64), np.uint8), {"ticket": ticket}))
        assert out["activations"] == index + 1
        assert out["status"]["current_ticket"] == ticket
        if index:
            assert out["status"]["previous_ticket"]["sequence"] == index
        assert block.last_result.payloads["status"]["ticket"] == ticket
    assert frame(block, reset=True)["activations"] == 1  # default is another stream
    assert frame(block, reset=True)["activations"] == 1


def test_context_changes_and_separate_context_cache():
    block = make(warmup=2, context_change="separate", max_streams=2)
    a = frame(block)["status"]["generation"]
    frame(block, np.zeros((24, 32), np.uint8))
    assert frame(block)["activations"] == 2
    block = make(warmup=2)
    frame(block)
    out = frame(block, np.zeros((24, 32), np.uint8))
    assert out["activations"] == 1 and out["status"]["generation"] != a
    block.parameters["mode"] = "first"
    assert frame(block, np.zeros((24, 32), np.uint8))["activations"] == 1


def test_recipe_revision_is_a_temporal_boundary():
    block = make(warmup=2)
    catalog = RecipeCatalog()
    a, b = catalog.put("A", {}), catalog.put("B", {})
    for recipe in (a, a, b):
        packet = ValuePacket(np.zeros((48, 64), np.uint8), {"recipe_ref": reference(recipe)})
        out = frame(block, packet)
    assert out["activations"] == 1 and not out["ready"]


@pytest.mark.parametrize("overflow", ["error", "evict_oldest"])
def test_stream_bounds(overflow):
    block = make(max_streams=1, overflow=overflow)
    frame(block, stream="left")
    block.execute({"image": np.zeros((48, 64), np.uint8), "stream": "right"})
    if overflow == "error":
        assert block.state == "error" and "stream limit" in block.error
        assert not block.memory["streams"]
    else:
        assert block.last_result.outputs["status"]["evicted"] == 1
        assert frame(block, stream="left")["activations"] == 1


def test_memory_budget_and_failure_reset():
    block = make(max_bytes_mb=0.001)
    block.execute({"image": np.zeros((48, 64), np.uint8)})
    assert block.state == "error" and "byte limit" in block.error
    assert not block.memory["streams"]
    block = make()
    frame(block)
    block.execute({"image": np.zeros((48, 64), np.uint8), "learn": "yes"})
    assert block.state == "error" and not block.memory["streams"]
    assert frame(block)["activations"] == 1


def test_gap_and_runtime_copies_are_private():
    block = make(max_gap_ms=100)
    block.now = lambda: 1.0
    frame(block)
    clone = make(max_gap_ms=100)
    clone.copy_runtime_from(block)
    clone.now = lambda: 1.05
    assert frame(clone)["activations"] == 2
    assert next(iter(block.memory["streams"].values()))["activations"] == 1
    clone.now = lambda: 2.0
    assert frame(clone)["activations"] == 1
    clone.close()
    assert block.memory["streams"]


@pytest.mark.parametrize("method", ["absolute", "variance"])
def test_background_foreground_mask_and_freeze(method):
    block = make("background", warmup=2, method=method)
    frame(block, 0)
    assert frame(block, 0)["ready"]
    image = np.zeros((48, 64), np.uint8)
    image[10:20, 20:30] = 255
    mask = np.zeros_like(image)
    mask[10:15, 20:30] = 255
    out = frame(block, image, mask=mask, learn=False)
    assert np.count_nonzero(out["mask"]) == 50
    assert not np.any(out["background"])
    assert out["status"]["foreground_fraction"] == 1


def test_background_optional_stability_gate():
    block = make("background", warmup=1, stable_activations=3)
    assert not frame(block)["ready"]
    assert not frame(block)["ready"]
    assert frame(block)["ready"]


@pytest.mark.parametrize("initial, gaussian", [("none", False), ("previous", True)])
def test_dense_flow_direction_metadata_and_selection(initial, gaussian):
    block = make("farneback", initial_flow=initial, gaussian_window=gaussian, max_gap_ms=0)
    image = texture()
    moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (128, 96))
    assert not frame(block, image)["ready"]
    out = frame(block, moved)
    assert out["ready"]
    np.testing.assert_allclose(
        np.median(out["flow"].vectors[20:-20, 20:-20], axis=(0, 1)), [2, 1], atol=0.25
    )
    assert out["flow"].metadata["sequence"]["activations"] == 2
    json.dumps(describe_value(out["flow"]), allow_nan=False)
    mask = np.zeros_like(image)
    mask[20:40, 20:40] = 255
    out = frame(block, moved, mask=mask)
    assert not np.any(out["flow"].vectors[mask == 0])


@pytest.mark.parametrize("metric", ["patch", "min_eigenvalue"])
def test_sparse_tracking_ids_and_age(metric):
    block = make("track_points", error_metric=metric, max_gap_ms=0)
    image = texture()
    assert not frame(block, image)["ready"]
    moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (128, 96))
    out = frame(block, moved)
    assert out["ready"]
    tracks = out["tracks"]["items"]
    assert len(tracks) >= 3 and all(t["age"] == 2 for t in tracks)
    np.testing.assert_allclose(
        np.median([t["displacement"] for t in tracks], axis=0), [2, 1], atol=0.25
    )
    old_ids = {t["id"] for t in tracks}
    again = frame(block, moved)["tracks"]["items"]
    assert old_ids & {t["id"] for t in again}


def test_featureless_tracking_does_not_become_ready_from_count_alone():
    block = make("track_points")
    for _ in range(4):
        out = frame(block, 0)
    assert not out["ready"] and out["tracks"] is NO_VALUE
    assert "Not ready" in block.last_result.message


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_trained_image_assets_roundtrip_and_integrity(dtype):
    image = ImageArray(np.full((8, 10, 3), 3, dtype), "rgb", {"custom": "discarded"})
    asset = encode_image(image)
    assert asset["context"] == {}
    restored = decode_image(asset)
    np.testing.assert_array_equal(restored, image)
    assert restored.color_format == "rgb" and restored.dtype == dtype
    bad = copy.deepcopy(asset)
    bad["shape"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        decode_image(bad)
    bad["id"] = seal(bad)
    with pytest.raises(ValueError, match="shape"):
        decode_image(bad)


def test_trained_seed_persists_but_live_state_does_not(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(make(warmup=8))
    block.train_reference(np.full((48, 64), 12, np.uint8))
    assert frame(block, 15)["ready"]
    path = tmp_path / "trained.json"
    graph.save(path)
    loaded = Graph.load(path, registry).blocks[block.id]
    assert not loaded.memory["streams"]
    out = frame(loaded, 15)
    assert out["status"]["seeded"] and out["status"]["required_activations"] == 0
    np.testing.assert_array_equal(out["difference"], 3)


def test_recipe_image_asset_swap_and_strict_context():
    catalog = RecipeCatalog()
    a = catalog.put("A", {})
    asset = encode_image(
        ImageArray(np.full((48, 64), 12, np.uint8), "gray", {"recipe_ref": reference(a)})
    )
    b = catalog.put("B", {}, {"reference": asset})
    block = make()
    image = ValuePacket(np.full((48, 64), 15, np.uint8), {"recipe_ref": reference(b)})
    block.execute({"image": image, "reference": asset})
    assert "grid/recipe" in block.error
    # An explicit recipe-field assignment binds this immutable asset to revision B.
    out = frame(block, image, reference=ValuePacket(asset, {"recipe_ref": reference(b)}))
    assert out["ready"]
    np.testing.assert_array_equal(out["difference"], 3)
    block = make(reference_context="shape_only")
    assert frame(block, image, reference=asset)["ready"]


def test_sequence_cannot_run_in_acquisition_prefix():
    with pytest.raises(ValueError, match="after the acquisition queue"):
        check_upstream(make())


def test_rejected_failed_image_does_not_advance_or_replace_history():
    block, clock = make(), TicketClock()
    image = np.full((48, 64), 10, np.uint8)
    frame(block, ValuePacket(image, {"ticket": clock.issue("camera")}))
    before = block.snapshot_runtime()
    block.input_failure_policies["image"] = "reject_failed"
    packet = ValuePacket(
        image + 10,
        {
            "ticket": clock.issue("camera"),
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": False,
                "chain_passed": False,
                "source": {"id": "upstream"},
                "score": None,
                "data": {},
            },
        },
    )
    block.execute({"image": packet})
    assert block.input_rejected and not block.current_emissions
    assert block.last_result.outputs["activations"] == 1
    np.testing.assert_array_equal(block.last_inputs["image"], image)
    assert (
        next(iter(before["streams"].values()))["activations"]
        == next(iter(block.memory["streams"].values()))["activations"]
    )


def test_nested_sequence_runtime_and_parallel_rejection():
    registry, child = builtin_registry(), Graph()
    memory = child.add(make(warmup=2))
    pose = child.add(registry.create("pose.offset"))
    tool = ToolBlock(
        child,
        {"image": Binding(memory.id, "image"), "pose": Binding(pose.id, "pose")},
        {"image": Binding(memory.id, "image")},
    )
    image = np.zeros((48, 64), np.uint8)
    inputs = {"image": image, "pose": make_pose(local_offset(0, 0, 0), [64, 48], {})}
    tool.execute(inputs)
    assert not tool.error and not memory.last_result.outputs["ready"]
    tool.execute(inputs)
    assert memory.last_result.outputs["ready"]
    parallel = ParallelInspect(inspection=tool, registry=registry)
    with pytest.raises(ValueError, match="stable stream"):
        parallel.validate_inspection()


def test_background_queue_only_advances_memory_for_dequeued_frames():
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.sequence_demo"))
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": 3}
        )
    )
    memory = graph.add(make(warmup=3))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", memory.id, "image")
    queued = QueuedExecution(graph, registry)
    try:
        queued.start()
        for index in range(1, 5):
            assert queued.run_one()
            assert memory.last_result.outputs["activations"] == index
        assert memory.last_result.outputs["ready"]
    finally:
        queued.close()
        graph.close()


def test_first_reference_keeps_original_training_ticket():
    block, clock = make(mode="first", warmup=2), TicketClock()
    for _ in range(2):
        out = frame(
            block, ValuePacket(np.zeros((48, 64), np.uint8), {"ticket": clock.issue("camera")})
        )
    assert out["status"]["reference_ticket"]["sequence"] == 1
    assert out["status"]["current_ticket"]["sequence"] == 2


def pump(app, predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("Worker did not finish")


def test_editor_f5_retains_memory_training_and_runtime_reset(app):
    window = MainWindow()
    window.replace_graph(sequence_graph(window.registry))
    block = window.graph.blocks["sequence_image_memory"]
    window.inspector.select(block)
    for index in range(1, 4):
        window.start_run(False)
        pump(app, lambda: window.runner is None)
        assert block.last_result.outputs["activations"] == index
    assert window.inspector.reference_capture.isEnabled()
    window.inspector.capture_reference()
    assert block.trained_image and window.dirty
    window.dirty = False
    window.start_run(False)
    pump(app, lambda: window.runner is None)
    assert block.last_result.outputs["ready"]
    window.inspector.reset_sequence()
    assert not block.memory["streams"] and block.trained_image
    assert not window.dirty
    window.close()


def test_synthetic_example_matches_saved_and_runs():
    registry = builtin_registry()
    built = sequence_graph(registry)
    saved = Graph.load("examples/sequence_memory/sequence_memory.json", registry)
    assert saved.to_dict() == built.to_dict()
    for _ in range(8):
        saved.run()
        assert all(b.state in ("pass", "waiting") for b in saved.blocks.values())
    assert all(b.state == "pass" for b in saved.blocks.values())
    json.dumps(saved.to_dict(), allow_nan=False)


def test_editor_stop_resume_and_layout_keep_sequence_history(app):
    window = MainWindow()
    window.replace_graph(sequence_graph(window.registry))
    block = window.graph.blocks["sequence_image_memory"]
    window.start_run(True)
    pump(
        app, lambda: block.last_result is not None and block.last_result.outputs["activations"] >= 4
    )
    window.stop_run()
    pump(app, lambda: window.runner is None)
    count = block.last_result.outputs["activations"]
    generation = block.last_result.outputs["status"]["generation"]
    node = window.scene.nodes[block.id]
    node.setPos(node.pos().x() + 20, node.pos().y())
    assert block.last_result.outputs["activations"] == count
    window.start_run(False)
    pump(app, lambda: window.runner is None)
    assert block.last_result.outputs["activations"] == count + 1
    assert block.last_result.outputs["status"]["generation"] == generation
    window.dirty = False
    window.close()


def test_recipe_manager_copies_and_describes_reference_image(app, monkeypatch):
    window = MainWindow()
    block = window.graph.add(make())
    block.train_reference(np.zeros((48, 64), np.uint8))
    window.inspector.select(block)
    manager = RecipeManager(window)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("reference", True))
    manager.copy_model()
    assert manager.models["reference"] == block.trained_image
    assert manager.assets.topLevelItem(0).text(1) == "trained_image_v1"
    assert "48" in manager.assets.topLevelItem(0).text(2)
    manager.close()
    window.dirty = False
    window.close()
