"""Exercise assignment, ticketed encoder displacement, and track life cycles."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.conveyor_tracking import associate_positions
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.poses import local_offset, make_pose
from opencv_dataflow.tickets import TicketClock


def test_assignment_maximizes_count_before_distance():
    predicted = np.array([[0.0, 0.0], [3.0, 0.0]])
    detections = np.array([[1.0, 0.0], [-2.0, 0.0]])
    assert associate_positions(predicted, detections, 2.0) == {0: 1, 1: 0}
    rng = np.random.default_rng(15)
    for _ in range(10):
        predicted = rng.integers(0, 8, size=(3, 2))
        detections = rng.integers(0, 8, size=(4, 2))
        distances = np.linalg.norm(predicted[:, None, :] - detections, axis=2)
        gate = 3
        matches = associate_positions(predicted, detections, gate)
        actual = -len(matches), sum(distances[row, col] for row, col in matches.items())
        candidates = []
        for cols in permutations(range(7), 3):
            hits = [
                distances[row, col]
                for row, col in enumerate(cols)
                if col < 4 and distances[row, col] <= gate
            ]
            candidates.append((-len(hits), sum(hits)))
        expected = min(candidates)
        assert actual[0] == expected[0]
        assert np.isclose(actual[1], expected[1])


def test_encoder_rollover_mismatch_and_offscreen_prediction_preserve_track_id():
    registry = builtin_registry()
    track = registry.create(
        "sequence.track_conveyor",
        parameters={
            "motion": "encoder",
            "encoder_modulus": 100,
            "pixels_per_count": [2.0, 0.0],
            "maximum_missed_frames": 2,
            "offscreen_margin": 20,
            "draw_overlay": False,
        },
    )
    image = np.zeros((32, 48), np.uint8)
    clock = TicketClock()

    def activation(count, x=None, mismatch=False):
        ticket = clock.issue("camera", "auto_number", None)
        encoder_ticket = clock.issue("encoder", "auto_number", None) if mismatch else ticket
        poses = [make_pose(local_offset(x, 12, 0), [48, 32], {})] if x is not None else []
        return track.execute(
            {
                "image": ValuePacket(image, {"ticket": ticket}),
                "detections": ValuePacket(poses, {"ticket": ticket}),
                "encoder": ValuePacket(count, {"ticket": encoder_ticket}),
            }
        )

    initial = activation(97, 36)
    identifier = initial.outputs["tracks"]["items"][0]["track_id"]
    skipped = activation(99, mismatch=True)
    assert skipped.outputs["status"]["reason"] == "encoder_ticket_mismatch"
    assert skipped.outputs["tracks"] is NO_VALUE
    predicted = activation(1)
    item = predicted.outputs["tracks"]["items"][0]
    assert item["track_id"] == identifier and item["pose"]["origin"] == [44, 12]
    assert not item["observed"] and item["in_frame"]
    assert predicted.outputs["poses"] == []
    outside = activation(4).outputs["tracks"]["items"][0]
    assert outside["pose"]["origin"] == [50, 12] and not outside["in_frame"]
    expired = activation(5)
    assert expired.outputs["count"] == 0
    assert expired.outputs["status"]["expired"] == 1


@pytest.mark.local_examples
def test_conveyor_example_keeps_ids_after_missing_detection_and_leaving_view():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/applications/conveyor_tracking/conveyor_tracking.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        tracker = graph.blocks["track"]
        history = []
        for _ in range(15):
            graph.run()
            assert tracker.last_result is not None, tracker.error
            history.append(tracker.last_result.outputs["tracks"]["items"])
        first_ids = [item["track_id"] for item in history[0]]
        assert len(first_ids) == 2
        assert [item["observed"] for item in history[4]] == [True, False]
        assert [item["track_id"] for item in history[5]] == first_ids
        assert all(item["observed"] for item in history[5])
        assert history[-1][1]["pose"]["origin"][0] > 139
        assert not history[-1][1]["in_frame"]
    finally:
        graph.close()


def test_blob_collection_and_image_encoder_keep_large_integer_precision():
    registry = builtin_registry()
    blobs = registry.create("inspect.blobs", parameters={"min_area": 8})
    track = registry.create(
        "sequence.track_conveyor",
        parameters={
            "motion": "encoder",
            "pixels_per_count": [3.0, 0.0],
            "association_distance": 1,
        },
    )
    image = np.zeros((40, 60), np.uint8)
    image[10:18, 15:25] = 255
    detections = blobs.execute({"image": image}).outputs["collection"]
    first = track.execute(
        {
            "image": ValuePacket(image, {"encoder": 2**60}),
            "detections": detections,
        }
    )
    assert first is not None, track.error
    image[10:18, 15:25] = 0
    image[10:18, 18:28] = 255
    detections = blobs.execute({"image": image}).outputs["collection"]
    second = track.execute(
        {
            "image": ValuePacket(image, {"encoder": 2**60 + 1}),
            "detections": detections,
        }
    )
    assert second is not None, track.error
    assert second.outputs["status"]["displacement"] == [3.0, 0.0]
    assert second.outputs["count"] == 1
    assert second.outputs["poses"][0]["track_id"] == first.outputs["poses"][0]["track_id"]
