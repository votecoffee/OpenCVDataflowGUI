"""Verify native background masks, stream lifecycle, and sequence budgets."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.sequence import share_native_sequence_runtime


@pytest.mark.parametrize("kind", ["sequence.mog2", "sequence.knn_background"])
def test_native_background_learns_then_reports_foreground_and_known_mask(kind):
    block = builtin_registry().create(kind, parameters={"warmup": 20, "detect_shadows": False})
    baseline = np.full((48, 64), 100, np.uint8)
    first = block.execute({"image": baseline})
    assert first is not None, block.error
    assert first.outputs["mask"] is NO_VALUE
    assert not first.outputs["ready"]
    for _ in range(24):
        block.execute({"image": baseline})
    frame = baseline.copy()
    frame[10:25, 15:30] = 230
    known = np.zeros_like(frame)
    known[4, 6] = 255
    result = block.execute({"image": frame, "known_foreground": known})
    assert result is not None, block.error
    assert result.outputs["ready"]
    assert result.outputs["mask"][16, 20] == 255
    assert result.outputs["mask"][4, 6] == 255
    assert result.outputs["mask"][0, 0] == 0
    assert result.outputs["foreground"][16, 20] == 230
    assert result.outputs["background"][16, 20] == 100


def test_mog2_freeze_and_reset_do_not_advance_learning():
    block = builtin_registry().create("sequence.mog2", parameters={"detect_shadows": False})
    baseline = np.full((30, 40), 60, np.uint8)
    paused = block.execute({"image": baseline, "learn": False})
    assert paused.outputs["status"]["learning_activations"] == 0
    assert paused.outputs["mask"] is NO_VALUE
    for _ in range(8):
        block.execute({"image": baseline})
    for _ in range(40):
        result = block.execute({"image": 255 - baseline, "learn": False})
    assert result.outputs["status"]["learning_activations"] == 8
    assert np.all(result.outputs["mask"] == 255)
    assert np.all(result.outputs["background"] == baseline)
    restarted = block.execute({"image": baseline, "reset": True})
    assert restarted.outputs["mask"] is NO_VALUE
    assert restarted.outputs["activations"] == 1


def test_editor_copies_share_native_model_but_independent_copies_do_not():
    registry = builtin_registry()
    root = Graph()
    block = root.add(registry.create("sequence.mog2"))
    image = np.full((30, 40), 60, np.uint8)
    for count in range(1, 8):
        worker = Graph.from_dict(root.to_dict(), registry)
        worker.copy_runtime_from(root)
        share_native_sequence_runtime(worker, root)
        worker.run(injected={(block.id, "image"): image})
        executed = worker.blocks[block.id]
        assert executed.last_result.outputs["activations"] == count
        root.copy_runtime_from(worker)
        share_native_sequence_runtime(root, worker)
        worker.close()
    separate = Graph.from_dict(root.to_dict(), registry)
    separate.copy_runtime_from(root)
    separate.run(injected={(block.id, "image"): image})
    assert separate.blocks[block.id].last_result.outputs["activations"] == 1
    root.run(injected={(block.id, "image"): image})
    assert root.blocks[block.id].last_result.outputs["activations"] == 8
    separate.close()
    root.close()


def test_background_memory_eviction_happens_before_model_allocation():
    block = builtin_registry().create(
        "sequence.mog2", parameters={"max_bytes_mb": 0.25, "overflow": "evict_oldest"}
    )
    image = np.full((48, 64), 60, np.uint8)
    block.execute({"image": image, "stream": "a"})
    second = block.execute({"image": image, "stream": "b"})
    assert second is not None, block.error
    assert second.outputs["status"]["streams"] == 1
    assert second.outputs["status"]["evicted"] == 1


def test_native_background_example_reports_moving_foreground():
    path = Path(__file__).resolve().parents[1] / "examples/native_background/native_background.json"
    graph = Graph.load(path, builtin_registry())
    try:
        for _ in range(25):
            graph.run()
        for key in ("mog2", "knn"):
            result = graph.blocks[key].last_result
            assert result.outputs["ready"]
            assert result.outputs["status"]["foreground_fraction"] > 0
    finally:
        graph.close()
