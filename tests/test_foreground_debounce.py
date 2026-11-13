"""Independent pixel references, temporal boundaries and bounded component state."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.foreground_debounce import ForegroundDebounce
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.recipes import RecipeCatalog, reference
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark
from opencv_dataflow.toolbox import block_entry


def target(x=4, *, y=16, width=4, height=5, shape=(40, 80), value=255, dtype=np.uint8):
    image = np.zeros(shape, dtype)
    image[y : y + height, x : x + width] = value
    return image


def run(block, image, **inputs):
    result = block.execute({"image": image, **inputs})
    assert result is not None, block.error
    return result


def test_disjoint_moving_part_confirms_without_emitting_single_frame_spikes():
    block = ForegroundDebounce()
    for index in range(6):
        truth = target(4 + 6 * index)
        image = truth.copy()
        if index % 2:
            image[2:5, 65:69] = 255  # Same glint, but never in consecutive frames.
        result = run(block, image)
        assert result.outputs["ready"] == (index >= 2)
        assert bool(block.current_emissions.get("mask")) == (index >= 2)
        assert "status" in block.current_emissions
        if index < 2:
            assert result.outputs["mask"] is NO_VALUE
            assert not result.drawings
        else:
            np.testing.assert_array_equal(result.outputs["mask"], truth)
            np.testing.assert_array_equal(result.outputs["rejected_mask"], image - truth)
            assert result.outputs["count"] == 1
            collection = result.outputs["components"]
            validate_collection(collection)
            assert collection["items"][0]["metrics"]["area"] == 20
            assert collection["items"][0]["metrics"]["center_x"] == 5.5 + 6 * index
            assert collection["items"][0]["temporal"]["consecutive_hits"] == 3
            assert result.drawings["foreground confirmation"].source == "image"
            colors = {obj.color for obj in result.drawings["foreground confirmation"].objects}
            assert (40, 210, 40) in colors
            assert ((0, 190, 255) in colors) == bool(index % 2)
        np.testing.assert_array_equal(image[16:21, 4 + 6 * index : 8 + 6 * index], 255)


def test_disappearance_emits_empty_mask_and_reappearance_reconfirms_without_ghosts():
    block = ForegroundDebounce(parameters={"warmup": 1, "consecutive_hits": 2})
    image = target()
    assert run(block, image).outputs["count"] == 0
    found = run(block, image)
    key = found.outputs["observations"][0]["track_id"]
    assert found.outputs["count"] == 1
    empty = run(block, np.zeros_like(image))
    assert empty.outputs["ready"] and empty.outputs["count"] == 0
    assert not empty.outputs["mask"].any() and not empty.outputs["observations"]
    again = run(block, image)
    assert again.outputs["count"] == 0
    assert again.outputs["observations"][0]["track_id"] != key
    assert run(block, image).outputs["count"] == 1


@pytest.mark.parametrize(
    "parameters,next_image",
    [
        ({"maximum_displacement": 2}, target(8)),
        ({"minimum_iou": 0.5}, target(8)),
        ({"minimum_area_ratio": 0.5}, target(width=8, height=10)),
    ],
)
def test_motion_overlap_and_area_gates_restart_confirmation(parameters, next_image):
    block = ForegroundDebounce(parameters={"warmup": 1, **parameters})
    run(block, target())
    result = run(block, next_image)
    assert result.outputs["observations"][0]["hits"] == 1
    assert not result.outputs["mask"].any()


def test_merge_and_split_do_not_arbitrarily_transfer_a_confirmed_identity():
    block = ForegroundDebounce(
        parameters={
            "warmup": 1,
            "consecutive_hits": 2,
            "minimum_area_ratio": 0,
            "maximum_displacement": 8,
        }
    )
    separate = target(12) | target(24)
    assert run(block, separate).outputs["count"] == 0
    assert run(block, separate).outputs["count"] == 2
    merged = target(12, width=16)
    result = run(block, merged)
    assert result.outputs["count"] == 0
    assert result.outputs["status"]["ambiguous_components"] == 1
    result = run(block, separate)
    assert result.outputs["count"] == 0
    assert result.outputs["status"]["ambiguous_components"] == 2
    assert run(block, separate).outputs["count"] == 2


@pytest.mark.parametrize(
    "dtype,value,threshold",
    [
        (np.uint8, 255, 127),
        (np.uint16, 50000, 32767),
        (np.float32, 0.75, 0.5),
        (np.float64, 4.0, 3.0),
    ],
)
def test_raw_threshold_preserves_input_samples_and_always_emits_binary_uint8(
    dtype, value, threshold
):
    image = target(dtype=dtype, value=value)
    image[2:7, 60:64] = threshold  # Strictly greater excludes shadow/equality labels.
    original = image.copy()
    block = ForegroundDebounce(
        parameters={"warmup": 1, "consecutive_hits": 1, "foreground_threshold": threshold}
    )
    result = run(block, ImageArray(image, "gray"))
    expected = target()
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    assert result.outputs["mask"].dtype == np.uint8
    np.testing.assert_array_equal(image, original)


def test_area_filter_and_connectivity_are_explicit():
    image = np.zeros((10, 10), np.uint8)
    image[2, 2] = image[3, 3] = 255
    for connectivity, count in (("4", 2), ("8", 1)):
        block = ForegroundDebounce(
            parameters={
                "warmup": 1,
                "consecutive_hits": 1,
                "minimum_area": 1,
                "connectivity": connectivity,
            }
        )
        assert run(block, image).outputs["count"] == count
    block = ForegroundDebounce(
        parameters={"warmup": 1, "consecutive_hits": 1, "minimum_area": 2, "maximum_area": 2}
    )
    assert run(block, image).outputs["count"] == 1
    block.parameters["minimum_area"] = 3
    block.parameters["maximum_area"] = 0
    result = run(block, image)
    assert result.outputs["count"] == 0
    np.testing.assert_array_equal(result.outputs["rejected_mask"], image)


@pytest.mark.parametrize("kind", ["rectangle", "circle", "polygon", "mask"])
def test_selection_clips_before_confirmation_and_changes_reset_history(kind):
    image = np.full((40, 80), 255, np.uint8)
    region = {"model": "image_region_v1", "image_size": [80, 40], "shape": kind}
    if kind == "rectangle":
        region["rectangle"] = [4, 8, 8, 8]
    elif kind == "circle":
        region.update(center=[10, 16], radius=5)
    elif kind == "polygon":
        region["vertices"] = [[4, 8], [12, 8], [8, 18]]
    block = ForegroundDebounce(parameters={"warmup": 1, "consecutive_hits": 2})
    controls = {"mask": target()} if kind == "mask" else {"region": region}
    first = run(block, image, **controls)
    assert first.outputs["count"] == 0
    result = run(block, image, **controls)
    assert result.outputs["count"] == 1
    generation = result.outputs["status"]["generation"]
    changed = run(block, image, mask=target(8))
    assert changed.outputs["status"]["generation"] != generation
    assert changed.outputs["activations"] == 1 and changed.outputs["count"] == 0


def test_current_ticket_timing_grid_pyramid_and_arbitrary_context_are_retained():
    block, clock = ForegroundDebounce(), TicketClock()
    grid = frame([80, 40], "inspection-level-1")
    pyramid = {
        "level": 1,
        "largest_size": [160, 80],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    for index in range(4):
        timing = advance_timing(None, mark(), mark(), source=True)
        context = {
            "ticket": clock.issue("camera"),
            "spatial": grid,
            "pyramid": pyramid,
            "capture": {"encoder": index * 17},
            "processing_time": timing,
        }
        result = run(block, ValuePacket(target(4 + index * 4), context))
        if index >= 2:
            for name in ("mask", "rejected_mask", "components", "count", "observations"):
                payload = result.payloads[name]
                for field in ("ticket", "spatial", "pyramid", "capture"):
                    assert payload[field] == context[field]
                assert payload["processing_time"]["started_at"] == timing["started_at"]
            assert result.outputs["components"]["context"] == {"spatial": grid, "pyramid": pyramid}
            assert (
                result.outputs["components"]["items"][0]["metrics"]["center_x"] == 5.5 + 4 * index
            )
    assert result.outputs["status"]["previous_ticket"]["sequence"] == 3


def test_runtime_copy_is_private_and_frame_recipe_parameter_gap_and_explicit_resets():
    block = ForegroundDebounce(parameters={"warmup": 1, "max_gap_ms": 100})
    block.now = lambda: 1.0
    result = run(block, target())
    result.outputs["observations"][0]["hits"] = 999  # Emitted data cannot alter retained memory.
    clone = ForegroundDebounce(parameters=block.parameters)
    clone.copy_runtime_from(block)
    clone.now = lambda: 1.05
    assert run(clone, target()).outputs["observations"][0]["hits"] == 2
    assert next(iter(block.memory["streams"].values()))["observations"][0]["hits"] == 1
    clone.now = lambda: 2.0
    assert run(clone, target()).outputs["activations"] == 1
    assert run(clone, target(), reset=True).outputs["activations"] == 1
    clone.parameters["minimum_area"] = 5
    assert run(clone, target()).outputs["activations"] == 1
    assert run(clone, target(shape=(50, 80))).outputs["activations"] == 1
    catalog = RecipeCatalog()
    first, second = catalog.put("A", {}), catalog.put("A", {"threshold": 1})
    a = ValuePacket(target(), {"recipe_ref": reference(first)})
    b = ValuePacket(target(), {"recipe_ref": reference(second)})
    run(clone, a)
    assert run(clone, b).outputs["activations"] == 1
    clone.close()
    assert block.memory["streams"]


def test_streams_do_not_share_evidence_and_component_byte_limits_fail_safely():
    block = ForegroundDebounce(parameters={"warmup": 1, "max_streams": 2, "maximum_components": 2})
    for stream in ("left", "right"):
        assert run(block, target(), stream=stream).outputs["observations"][0]["hits"] == 1
    assert run(block, target(), stream="left").outputs["observations"][0]["hits"] == 2
    assert block.execute({"image": target() | target(24) | target(44), "stream": "left"}) is None
    assert "component limit" in block.error and not block.memory["streams"]
    block.parameters["max_bytes_mb"] = 0.001
    assert block.execute({"image": target()}) is None
    assert "byte limit" in block.error and not block.memory["streams"]


def test_large_observation_diagnostics_do_not_overflow_execution_events():
    image = np.zeros((80, 100), np.uint8)
    image[::10, ::10] = 255
    block = ForegroundDebounce(
        parameters={
            "warmup": 1,
            "consecutive_hits": 1,
            "minimum_area": 1,
            "maximum_components": 100,
        }
    )
    result = run(block, image)
    assert result.outputs["count"] == 80
    assert len(json.dumps(result.outputs["observations"])) > 8192
    assert len(json.dumps(result.event_data).encode()) < 8192


def test_long_sequence_keeps_only_current_components_and_caps_streak_counters():
    block = ForegroundDebounce(parameters={"warmup": 1})
    for index in range(120):
        result = run(block, target(4 + index % 2))
    state = next(iter(block.memory["streams"].values()))
    assert len(state["observations"]) == 1 and state["next_id"] == 1
    assert state["observations"][0]["hits"] == 3
    assert result.outputs["activations"] == 120
    assert len(json.dumps(state)) < 2048  # No images or unbounded frame history.
    packet = ValuePacket(target(), {"spatial": frame([80, 40], "new-grid")})
    assert run(block, packet).outputs["activations"] == 1


def test_rejected_failed_inputs_preserve_history_and_hits_count_eligible_activations():
    block = ForegroundDebounce(parameters={"warmup": 1, "consecutive_hits": 2})
    block.input_failure_policies["image"] = "reject_failed"
    clock = TicketClock()
    run(block, ValuePacket(target(), {"ticket": clock.issue("camera")}))
    before = block.snapshot_runtime()
    failed = ValuePacket(
        target(40),
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
    block.execute({"image": failed})
    assert block.input_rejected and not block.current_emissions
    assert block.snapshot_runtime() == before
    np.testing.assert_array_equal(block.last_inputs["image"], target())
    result = run(block, ValuePacket(target(), {"ticket": clock.issue("camera")}))
    assert result.outputs["count"] == 1 and result.outputs["activations"] == 2
    assert result.payloads["mask"]["ticket"]["sequence"] == 3


@pytest.mark.parametrize(
    "parameters,image",
    [
        ({"consecutive_hits": 0}, target()),
        ({"minimum_iou": 2}, target()),
        ({"maximum_area": 2}, target()),
        ({"maximum_components": 513}, target()),
        ({}, np.zeros((40, 80, 3), np.uint8)),
        ({}, np.full((40, 80), np.nan)),
    ],
)
def test_bad_parameters_or_images_clear_state_and_emit_nothing(parameters, image):
    block = ForegroundDebounce()
    run(block, target())
    block.parameters.update(parameters)
    assert block.execute({"image": image}) is None
    assert block.error and not block.memory["streams"] and not block.current_emissions


def test_graphics_off_and_project_xml_preserve_configuration(tmp_path, monkeypatch):
    import opencv_dataflow.foreground_debounce as module

    def unexpected(*args, **kwargs):
        raise AssertionError("Graphics were disabled")

    monkeypatch.setattr(module, "graphic_box", unexpected)
    block = builtin_registry().create("sequence.foreground_debounce", parameters={"warmup": 1})
    block.set_graphics_mode("off")
    assert not run(block, target()).drawings
    assert block_entry(block).path == ("Custom", "Sequence", "Foreground masks")
    assert ForegroundDebounce.requires_serial_execution
    graph = Graph()
    graph.add(block)
    try:
        data = graph.to_dict()
        export_project_xml(data, tmp_path / "debounce.xml")
        restored = Graph.from_dict(read_project_xml(tmp_path / "debounce.xml"), builtin_registry())
        try:
            assert restored.to_dict() == data
            assert not restored.blocks[block.id].memory["streams"]
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_replay_matches_independent_truth_and_reset_repeats_it():
    folder = Path(__file__).parents[1] / "examples/custom tools/foreground_debounce"
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(folder / "project.json", builtin_registry())
    try:
        for repetition in range(2):
            for index, count in enumerate(truth["expected_counts"]):
                graph.run()
                block = graph.blocks["debounce"]
                assert block.last_result is not None, block.error
                result = block.last_result
                assert result.outputs["activations"] == index + 1
                if count is None:
                    assert result.outputs["mask"] is NO_VALUE
                else:
                    assert result.outputs["count"] == count
                    expected = cv2.imread(
                        str(folder / f"assets/expected/frame-{index:02d}.png"), cv2.IMREAD_GRAYSCALE
                    )
                    np.testing.assert_array_equal(result.outputs["mask"], expected)
            graph.invalidate()
    finally:
        graph.close()
