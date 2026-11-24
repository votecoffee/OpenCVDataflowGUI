"""Verify the native replay source and its saved ground-truth comparison."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.tickets import ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "opencv tools" / "synthetic_background_sequence"
TYPE = "input.synthetic_background_sequence"


def make_source(**parameters):
    source = builtin_registry().create(
        TYPE,
        block_id="source",
        parameters={
            "background_path": "assets/background.png",
            "object_path": "assets/target.png",
            **parameters,
        },
    )
    source.project_dir = EXAMPLE
    return source


def execute(source):
    result = source.execute({})
    assert result is not None, source.error
    return result


def test_matches_native_sequence_across_direction_changes():
    source = make_source(frame_count=24, object_speed=20.0, fps=10.0)
    background = cv2.imread(str(EXAMPLE / "assets" / "background.png"), cv2.IMREAD_UNCHANGED)
    target = cv2.imread(str(EXAMPLE / "assets" / "target.png"), cv2.IMREAD_UNCHANGED)
    native = cv2.bgsegm.createSyntheticSequenceGenerator(background, target, 2.0, 20.0, 0.2, 20.0)
    tickets, positions = set(), set()
    try:
        for index in range(24):
            expected_image, expected_mask = native.getNextFrame()
            result = execute(source)
            frame, mask = (result.outputs[name] for name in ("image", "foreground_mask"))
            np.testing.assert_array_equal(frame, expected_image)
            np.testing.assert_array_equal(mask, expected_mask)
            info = result.outputs["info"]
            assert info["sequence_index"] == info["frame_index"] == index
            assert info["timestamp_ms"] == index * 100.0
            x, y, width, height = info["foreground_bounds_xywh"]
            assert (width, height) == (30, 24)
            assert np.count_nonzero(mask) == info["foreground_pixels"] == 720
            assert np.all(mask[y : y + height, x : x + width] == 255)
            assert result.payloads["image"] == result.payloads["foreground_mask"]
            assert result.payloads["image"]["spatial"]["image_size"] == [176, 128]
            assert result.payloads["image"]["video_frame"] == info
            ticket = ticket_key(result.outputs["ticket"])
            assert ticket_key(result.payloads["foreground_mask"]["ticket"]) == ticket
            assert ticket_key(result.payloads["info"]["ticket"]) == ticket
            tickets.add(ticket)
            positions.add((x, y))
        assert len(tickets) == 24
        assert len(positions) > 15
        assert source.exhausted
        assert all(value is NO_VALUE for value in execute(source).outputs.values())
    finally:
        source.close()


def test_resumes_from_cursor_after_direction_changes():
    source = make_source(frame_count=24, object_speed=20.0)
    resume = make_source(frame_count=24, object_speed=20.0)
    try:
        for _ in range(13):
            execute(source)
        resume.copy_runtime_from(source)
        assert resume.execute({}, cancelled=lambda: True).outputs["image"] is NO_VALUE
        expected, restored = execute(source), execute(resume)
        assert restored.outputs["info"]["frame_index"] == 13
        assert restored.outputs["info"] == expected.outputs["info"]
        np.testing.assert_array_equal(restored.outputs["image"], expected.outputs["image"])
        np.testing.assert_array_equal(
            restored.outputs["foreground_mask"], expected.outputs["foreground_mask"]
        )
    finally:
        source.close()
        resume.close()


def test_loops_and_rewinds_repeat_native_frames_with_new_tickets():
    source = make_source(frame_count=3, fps=20.0)
    source.configure_replay("loop")
    try:
        results = [execute(source) for _ in range(7)]
        assert not source.finite_source
        for index, result in enumerate(results):
            info = result.outputs["info"]
            assert index % 3 == info["frame_index"]
            assert index // 3 == info["loop_index"]
            assert index * 50.0 == info["timestamp_ms"]
            np.testing.assert_array_equal(result.outputs["image"], results[index % 3].outputs["image"])
        source.configure_replay("stop", rewind=True)
        first = execute(source)
        assert first.outputs["info"] == results[0].outputs["info"]
        np.testing.assert_array_equal(first.outputs["image"], results[0].outputs["image"])
        assert len({ticket_key(result.outputs["ticket"]) for result in [*results, first]}) == 8
    finally:
        source.close()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"object_speed": 53}, "Object speed must be at most 52"),
        ({"object_path": "assets/background.png"}, "Foreground must be smaller"),
        ({"object_path": "missing.png"}, "FileNotFoundError"),
    ],
)
def test_rejects_invalid_assets_or_motion_before_native_creation(parameters, message, monkeypatch):
    source = make_source(**parameters)
    native_calls = []
    monkeypatch.setattr(cv2.bgsegm, "createSyntheticSequenceGenerator", lambda *args: native_calls.append(args))
    try:
        assert source.execute({}) is None
        assert message in source.error
        assert source.exhausted
        assert not native_calls
    finally:
        source.close()


def test_resume_refuses_changed_input_file(tmp_path):
    source = make_source()
    path = tmp_path / "target.png"
    original = EXAMPLE / "assets" / path.name
    path.write_bytes(original.read_bytes())
    source.parameters["object_path"] = str(path)
    try:
        execute(source)
        source.close()
        assert cv2.imwrite(str(path), np.full((24, 30), 220, np.uint8))
        assert source.execute({}) is None
        assert "Source images changed" in source.error
        source.configure_replay("stop", rewind=True)
        assert execute(source).outputs["info"]["frame_index"] == 0
    finally:
        source.close()


def test_save_as_preserves_relative_image_paths(tmp_path):
    registry = builtin_registry()
    original = EXAMPLE / "synthetic_background_sequence.json"
    destination = tmp_path / "relocated.json"
    graph = Graph.load(original, registry)
    try:
        graph.save(destination)
    finally:
        graph.close()
    relocated = Graph.load(destination, registry)
    try:
        relocated.run()
        assert relocated.blocks["compare"].state == "pass", relocated.blocks["source"].error
    finally:
        relocated.close()


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_matches_ground_truth_for_every_ticket(profile):
    graph = Graph.load(EXAMPLE / "synthetic_background_sequence.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        for index in range(24):
            assert not graph.sources_exhausted()
            graph.run()
            comparison = graph.blocks["compare"]
            assert comparison.state == "pass", comparison.error
            assert comparison.last_result.score == 1.0
            source = graph.blocks["source"]
            for key in ("image", "foreground_mask"):
                assert source.current_emissions[key][0].payload["video_frame"]["frame_index"] == index
            if profile == "debug":
                result = comparison.last_result
                metrics = result.outputs["measurements"]
                assert metrics["iou"] == 1.0
                assert metrics["extra_pixels"] == metrics["missing_pixels"] == 0
                assert result.outputs["distance"] == 0.0
        assert graph.sources_exhausted()
    finally:
        graph.close()
