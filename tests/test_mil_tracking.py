"""Native annotated motion and deterministic guards/lifecycle for the MIL wrapper."""

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.mil_tracking import NATIVE_FIELDS, MILSequence
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.sequence import share_native_sequence_runtime
from opencv_dataflow.tickets import TicketClock, ticket_key


def frames():
    rng = np.random.default_rng(156)
    background = rng.integers(15, 35, (128, 160), np.uint8)
    patch = rng.integers(70, 230, (32, 32), np.uint8)
    cv2.rectangle(patch, (1, 1), (30, 30), 250, 2)
    cv2.line(patch, (3, 10), (28, 24), 15, 3)
    for index in range(7):
        image = background.copy()
        x, y = 44 + index * 2, 42
        image[y : y + 32, x : x + 32] = patch
        yield image, [x, y, 32, 32]


def execute(block, image, **inputs):
    result = block.execute({"image": image, **inputs})
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("encoding", ["gray", "rgb"])
def test_native_tracker_follows_annotated_translation_and_has_no_confidence_claim(encoding):
    cv2.setRNGSeed(156)
    block = MILSequence(
        parameters={"initial_bounds": [44, 42, 32, 32], "warmup": 3, "search_radius": 8}
    )
    errors = []
    try:
        for index, (image, expected) in enumerate(frames()):
            if encoding == "rgb":
                image = ImageArray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), "rgb")
            before = image.copy()
            result = execute(block, image)
            np.testing.assert_array_equal(image, before)
            assert (
                result.score is None
                and result.outputs["status"]["native_confidence_available"] is False
            )
            if index < 2:
                assert result.outputs["status"]["phase"] == (
                    "initialized" if index == 0 else "tracking"
                )
                assert result.outputs["bounds"] is NO_VALUE and not result.outputs["ready"]
            else:
                assert result.outputs["ready"] and result.passed
                bounds = result.outputs["bounds"]
                errors.append(float(np.linalg.norm(np.array(bounds[:2]) - expected[:2])))
                assert bounds[2:] == [32, 32]
                assert result.outputs["tracked_region"]["rectangle"] == bounds
                assert result.drawings and result.outputs["overlay"] is NO_VALUE
        assert max(errors) <= 8, errors
    finally:
        block.close()


class FakeTracker:
    def __init__(self, update=(True, (46, 42, 32, 32)), initialized=None):
        self.update_result = update
        self.initialized = initialized

    def init(self, image, bounds):
        assert image.dtype == np.uint8 and image.ndim == 2
        return self.initialized

    def update(self, image):
        return self.update_result


def fake(monkeypatch, **kwargs):
    tracker = FakeTracker(**kwargs)
    observed = []

    def create(options):
        observed.append(
            {native: getattr(options, native) for native, _, _ in NATIVE_FIELDS.values()}
        )
        return tracker

    monkeypatch.setattr(cv2.TrackerMIL, "create", create)
    return tracker, observed


def test_every_native_parameter_is_forwarded_with_limits_and_coupled_validation(monkeypatch):
    _, observed = fake(monkeypatch)
    parameters = {
        "initial_bounds": [44, 42, 32, 32],
        "features": 120,
        "init_positive_radius": 2.5,
        "search_radius": 14.0,
        "init_negative_samples": 40,
        "track_positive_radius": 3.0,
        "track_positive_samples": 200,
        "track_negative_samples": 45,
    }
    block = MILSequence(parameters=parameters)
    try:
        execute(block, next(frames())[0])
        assert set(observed[0]) == {
            name for name in dir(cv2.TrackerMIL.Params()) if not name.startswith("_")
        }
        for name, (native, _, _) in NATIVE_FIELDS.items():
            assert observed[0][native] == pytest.approx(parameters[name])
    finally:
        block.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"features": 0},
        {"features": 1025},
        {"search_radius": 3},
        {"track_positive_samples": 0},
        {"draw_overlay": "true"},
        {"search_radius": "25"},
        {"initial_bounds": [1, 1, 7, 7]},
        {"initial_bounds": [True, 1, 8, 8]},
        {"selection_mode": "unsupported"},
        {"max_bytes_mb": 0},
        {"max_misses": 0},
    ],
)
def test_static_bad_configuration_does_not_create_native_model(monkeypatch, bad):
    _, observed = fake(monkeypatch)
    block = MILSequence(parameters=bad)
    assert block.execute({"image": next(frames())[0]}) is None
    assert block.error_kind == "invalid" and not observed and not block.current_emissions


@pytest.mark.parametrize(
    "reason",
    ["target_low_contrast", "center_jump_above_maximum", "center_outside_region", "target_missing"],
)
def test_failure_gates_expire_and_require_an_explicit_new_seed(monkeypatch, reason):
    tracker, _ = fake(monkeypatch, update=(reason != "target_missing", (65, 42, 32, 32)))
    parameters = {"initial_bounds": [44, 42, 32, 32], "max_misses": 1}
    if reason == "center_jump_above_maximum":
        parameters["maximum_center_jump"] = 2
    block = MILSequence(parameters=parameters)
    image = next(frames())[0]
    selection = np.zeros_like(image)
    selection[40:78, 40:80] = 255
    try:
        execute(
            block,
            image,
            mask=selection if reason == "center_outside_region" else np.full_like(image, 255),
        )
        if reason == "target_low_contrast":
            image = np.zeros_like(image)
        result = execute(
            block, image, **({"mask": selection} if reason == "center_outside_region" else {})
        )
        assert not result.passed and not result.outputs["ready"]
        assert result.outputs["bounds"] is NO_VALUE and result.outputs["pose"] is NO_VALUE
        assert result.outputs["status"]["reason"] == reason
        assert result.outputs["status"]["last_accepted_bounds"] == [44, 42, 32, 32]
        assert result.fail_reasons[0]["code"] == "mil_" + reason
        waiting = execute(block, next(frames())[0])
        assert waiting.outputs["status"]["reason"] == "tracker_expired"
        tracker.update_result = (True, (44, 42, 32, 32))
        reseeded = execute(block, next(frames())[0], seed=[44, 42, 32, 32])
        assert reseeded.outputs["status"]["phase"] == "initialized"
    finally:
        block.close()


@pytest.mark.parametrize("mode,accepted", [("center", False), ("seed_only", True)])
def test_selection_can_supply_seed_and_optionally_gate_later_centers(monkeypatch, mode, accepted):
    fake(monkeypatch, update=(True, (65, 42, 32, 32)))
    image = next(frames())[0]
    mask = np.zeros_like(image)
    mask[42:74, 44:76] = 255
    block = MILSequence(parameters={"selection_mode": mode})
    try:
        seeded = execute(block, image, mask=mask)
        assert seeded.outputs["status"]["last_accepted_bounds"] == [44, 42, 32, 32]
        result = execute(block, image, mask=mask)
        assert result.outputs["ready"] is accepted and result.passed is accepted
        if not accepted:
            assert result.outputs["status"]["reason"] == "center_outside_region"
    finally:
        block.close()


def test_roi_seed_precedence_circle_gate_and_ticket_metadata(monkeypatch):
    _, _ = fake(monkeypatch, update=(True, (44, 42, 32, 32)))
    image = next(frames())[0]
    clock = TicketClock()
    ticket = clock.issue("camera", "fixed", "first")
    packet = ValuePacket(
        image, {"ticket": ticket, "spatial": frame([160, 128], "camera"), "lot": "A"}
    )
    region = {
        "model": "image_region_v1",
        "shape": "circle",
        "center": [60, 58],
        "radius": 22,
        "image_size": [160, 128],
    }
    block = MILSequence(parameters={"initial_bounds": [5, 5, 12, 12]})
    try:
        result = execute(block, packet, seed={"bounds": [44, 42, 32, 32]}, region=region)
        first_id = result.outputs["status"]["track_id"]
        result = execute(block, packet, region=region)
        assert result.outputs["ready"]
        assert result.outputs["pose"]["track_id"] == first_id
        assert ticket_key(result.payloads["pose"]["ticket"]) == ticket_key(ticket)
        assert result.payloads["pose"]["lot"] == "A"
        reset = execute(block, packet, reset=True)
        assert reset.outputs["status"]["track_id"] != first_id
    finally:
        block.close()


def test_work_memory_init_and_update_exceptions_are_contained(monkeypatch):
    tracker, observed = fake(monkeypatch)
    image = next(frames())[0]
    for extra in ({"maximum_work": 1}, {"max_bytes_mb": 0.1}):
        block = MILSequence(parameters={"initial_bounds": [44, 42, 32, 32], **extra})
        assert block.execute({"image": image}) is None and not observed
        block.close()
    tracker.initialized = False
    block = MILSequence(parameters={"initial_bounds": [44, 42, 32, 32]})
    assert block.execute({"image": image}) is None
    assert not block.memory["streams"]
    tracker.initialized = None
    execute(block, image)

    def broken(*args):
        raise cv2.error("controlled update exception")

    tracker.update = broken
    assert block.execute({"image": image}) is None and not block.current_emissions
    assert not block.memory["streams"]
    block.close()


def test_graphics_off_preserves_tracking_and_explicit_raster_is_optional(monkeypatch):
    fake(monkeypatch)
    block = MILSequence(parameters={"initial_bounds": [44, 42, 32, 32]})
    block.set_graphics_mode("off")
    try:
        execute(block, next(frames())[0])
        result = execute(block, next(frames())[0])
        assert result.outputs["ready"] and not result.drawings and not result.graphics
        assert result.outputs["overlay"] is NO_VALUE
        block.exposed_outputs.append("overlay")
        result = execute(block, next(frames())[0])
        assert result.outputs["overlay"].shape == (128, 160, 3)
    finally:
        block.close()


def test_worker_sharing_persistence_and_editor_choices(app, tmp_path, monkeypatch):
    from opencv_dataflow.app import MainWindow

    fake(monkeypatch)
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.sequence_demo"))
    tracker = graph.add(
        registry.create("sequence.track_mil", parameters={"initial_bounds": [44, 42, 32, 32]})
    )
    graph.connect(source.id, "image", tracker.id, "image")
    try:
        for index in range(3):
            worker = Graph.from_dict(graph.to_dict(), registry)
            worker.copy_runtime_from(graph)
            share_native_sequence_runtime(worker, graph)
            worker.run()
            assert worker.blocks[tracker.id].last_result.outputs["activations"] == index + 1
            share_native_sequence_runtime(graph, worker)
            worker.close()
        xml = tmp_path / "tracker.sequence.track_mil.xml"
        save_block_xml(tracker, xml)
        assert read_block_xml(xml)["parameters"] == tracker.parameters
        graph.save(tmp_path / "project.json")
        reopened = Graph.load(tmp_path / "project.json", registry)
        assert not reopened.blocks[tracker.id].memory["streams"]
        reopened.close()
        window = MainWindow()
        try:
            window.replace_graph(graph)
            window.inspector.select(tracker)
            for name in tracker.parameter_choices:
                assert isinstance(window.inspector.fields[name], QComboBox)
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    finally:
        graph.close()
