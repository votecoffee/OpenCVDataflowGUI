"""Native local evidence, periodic scheduling and detector-refresh provenance."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.refresh_tracking import local_motion
from opencv_dataflow.tickets import TicketClock

TYPE = "sequence.track_refreshed"


def scene(index, second=False, absent=False):
    image = np.zeros((112, 224), np.uint8)
    texture = np.random.default_rng(21).integers(80, 240, (32, 40), np.uint8)
    if not absent:
        x = 20 + 3 * index
        image[30:62, x : x + 40] = texture
    if second:
        image[76:100, 160:190] = texture[:24, :30]
    return image


def detection(index):
    return {"center": [39.5 + 3 * index, 45.5], "size": [40, 32], "angle_degrees": 0}


def runner(**parameters):
    block, clock = (
        builtin_registry().create(TYPE, parameters={"emit_tentative": True, **parameters}),
        TicketClock(),
    )

    def step(index, refresh, items=None, image=None):
        payload = {"spatial": frame([224, 112], "camera"), "ticket": clock.issue("camera")}
        values = {
            "image": ValuePacket(scene(index) if image is None else image, payload),
            "timestamp_ms": ValuePacket(index * 100, payload),
            "refresh": ValuePacket(refresh, payload),
        }
        if items is not None:
            values["detections"] = ValuePacket(items, payload)
        result = block.execute(values)
        assert result is not None, block.error
        return result

    return block, step


def graph_fixture(period=4, **parameters):
    class Source(Block):
        type_id = "input.tracker_test"
        outputs = (Port("image"),)
        image = scene(0)

        def process(self, inputs):
            return Result(
                {"image": ImageArray(self.image, "gray", {"spatial": frame([224, 112], "camera")})}
            )

    registry, graph = builtin_registry(), Graph()
    source = graph.add(Source(block_id="source"))
    graph.add(
        registry.create("flow.detection_cadence", block_id="cadence", parameters={"period": period})
    )
    graph.add(registry.create("inspect.blobs", block_id="detector", parameters={"min_area": 40}))
    tracker = graph.add(
        registry.create(TYPE, block_id="track", parameters={"emit_tentative": True, **parameters})
    )
    for a, b, c, d in (
        ("source", "image", "cadence", "image"),
        ("cadence", "image", "detector", "image"),
        ("source", "image", "track", "image"),
        ("cadence", "refresh", "track", "refresh"),
        ("detector", "collection", "track", "detections"),
    ):
        graph.connect(a, b, c, d)
    return graph, source, tracker


@pytest.mark.parametrize(
    "error_metric,fit_method", [("patch", "ransac"), ("min_eigenvalue", "lmeds")]
)
def test_local_native_similarity_matches_independent_translation(error_metric, fit_method):
    p = builtin_registry().create(TYPE).parameters
    p.update(local_error_metric=error_metric, fit_method=fit_method)
    first, second = scene(0), scene(1)
    mask = np.zeros(first.shape, np.uint8)
    mask[30:62, 20:60] = 255
    points = cv2.goodFeaturesToTrack(first, 48, 0.01, 3, mask=mask, blockSize=5)
    (matrix, retained), info = local_motion(first, second, points, p, None)
    np.testing.assert_allclose(matrix, [[1, 0, 3], [0, 1, 0]], atol=0.04)
    assert info["reason"] == "accepted" and len(retained) >= 4 and info["inlier_fraction"] > 0.9


def test_periodic_graph_calls_detector_only_on_refresh_and_discovers_entering_part(monkeypatch):
    graph, source, tracker = graph_fixture()
    calls = []
    native = graph.blocks["detector"].process

    def detect(inputs):
        calls.append(len(calls))
        return native(inputs)

    monkeypatch.setattr(graph.blocks["detector"], "process", detect)
    identity = None
    try:
        for index in range(11):
            source.image = scene(index, second=index >= 5)
            graph.run()
            assert tracker.state == "pass", tracker.error
            report = tracker.last_result.outputs["report"]
            assert report["refresh"] is (index % 4 == 0)
            assert len(calls) == index // 4 + 1
            assert graph.blocks["detector"].state == ("pass" if index % 4 == 0 else "inactive")
            active = report["active"]
            assert len(active) == (2 if index >= 8 else 1)
            if index == 0:
                identity = active[0]["track_id"]
            assert active[0]["track_id"] == identity
            np.testing.assert_allclose(active[0]["center"], detection(index)["center"], atol=0.5)
            assert active[0]["measurement_source"] == (
                "detector" if index % 4 == 0 else "local_flow"
            )
            assert active[0]["detector_age_frames"] == index % 4
            item = tracker.last_result.outputs["tracks"]["items"][0]
            assert item["pose"]["detector_observed"] is (index % 4 == 0)
            assert (
                item["pose"]["target_payload"]["ticket"]
                == tracker.last_result.payloads["tracks"]["ticket"]
            )
    finally:
        graph.close()


def test_detector_age_expires_even_when_local_evidence_keeps_succeeding():
    _, step = runner(maximum_detector_age_frames=2)
    identity = step(0, True, [detection(0)]).outputs["tracks"]["items"][0]["track_id"]
    assert step(1, False).outputs["report"]["active"][0]["measurement_source"] == "local_flow"
    assert step(2, False).outputs["count"] == 1
    expired = step(3, False)
    assert expired.outputs["count"] == 0
    assert expired.outputs["report"]["expired"] == [{"id": 1, "reason": "detector_age"}]
    assert step(4, True, [detection(4)]).outputs["tracks"]["items"][0]["track_id"] != identity


def test_lost_local_support_is_not_reseeded_from_predicted_background():
    _, step = runner(maximum_missed_frames=1)
    step(0, True, [detection(0)])
    missing = step(1, False, image=scene(1, absent=True))
    assert missing.outputs["report"]["active"][0]["measurement_source"] == "prediction"
    expired = step(2, False, image=scene(2, absent=True))
    assert (
        expired.outputs["count"] == 0
        and expired.outputs["report"]["expired"][0]["reason"] == "missed_frames"
    )
    assert step(3, False).outputs["count"] == 0
    assert step(4, True, [detection(4)]).outputs["count"] == 1


def test_full_frame_refresh_reduces_controlled_local_drift(monkeypatch):
    import opencv_dataflow.refresh_tracking as module

    native = module.local_motion

    def biased(*args):
        value, info = native(*args)
        if value is not None:
            matrix, points = value
            matrix = matrix.copy()
            matrix[0, 2] += 1.0
            value = matrix, points
        return value, info

    monkeypatch.setattr(module, "local_motion", biased)
    _, step = runner(position_noise=0.5, position_acceleration=10, innovation_gate=1000)
    step(0, True, [detection(0)])
    for index in range(1, 4):
        previous = step(index, False)
    before = abs(previous.outputs["report"]["active"][0]["center"][0] - detection(3)["center"][0])
    after = step(4, True, [detection(4)])
    error = abs(after.outputs["report"]["active"][0]["center"][0] - detection(4)["center"][0])
    assert before > 1 and error < before * 0.7


@pytest.mark.parametrize("refresh,items", [(True, None), (False, []), (1, None)])
def test_mismatched_refresh_contract_is_invalid_and_retires_memory(refresh, items):
    block, step = runner()
    step(0, True, [detection(0)])
    values = {"image": scene(1), "refresh": refresh}
    if items is not None:
        values["detections"] = items
    assert block.execute(values) is None
    assert (
        block.error_kind == "invalid"
        and not block.memory["streams"]
        and not any(block.current_emissions.values())
    )


def test_memory_copy_is_independent_and_parameters_round_trip():
    block, step = runner()
    step(0, True, [detection(0)])
    worker = builtin_registry().create(TYPE, parameters=copy.deepcopy(block.parameters))
    worker.copy_runtime_from(block)
    original = block.snapshot_runtime()
    points = next(iter(worker.memory["streams"].values()))["local_points"][1]
    points[:] = 0
    assert np.any(next(iter(block.memory["streams"].values()))["local_points"][1])
    assert next(iter(original["streams"].values()))["activations"] == 1


def test_native_local_rotation_and_scale_against_independent_affine_truth():
    image = cv2.GaussianBlur(scene(0), (5, 5), 1.2)
    matrix = cv2.getRotationMatrix2D((39.5, 45.5), 7, 1.05)
    changed = cv2.warpAffine(image, matrix, image.shape[::-1])
    mask = np.zeros(image.shape, np.uint8)
    mask[32:60, 22:58] = 255
    points = cv2.goodFeaturesToTrack(image, 48, 0.01, 3, mask=mask, blockSize=5)
    p = builtin_registry().create(TYPE, parameters={"local_window": 11}).parameters
    result, info = local_motion(image, changed, points, p, None)
    assert result is not None, info
    controls = np.array([[22, 32, 1], [58, 32, 1], [58, 60, 1], [22, 60, 1]])
    assert np.max(np.linalg.norm(controls @ result[0].T - controls @ matrix.T, axis=1)) < 0.3
    assert info["scale"] == pytest.approx(1.05, abs=0.01)
    assert info["angle_degrees"] == pytest.approx(7, abs=0.5)


def test_forced_cadence_restarts_budget_and_reset_restarts_first_frame():
    block = builtin_registry().create("flow.detection_cadence", parameters={"period": 4})
    calls = []
    for index in range(8):
        result = block.execute({"image": scene(0), "force": index == 2})
        assert result is not None, block.error
        if result.outputs["refresh"]:
            calls.append(index)
        assert bool(result.emissions.get("image")) is result.outputs["refresh"]
    assert calls == [0, 2, 6]
    assert result.outputs["report"]["detector_calls"] == 3
    assert block.execute({"image": scene(0), "reset": True}).outputs["refresh"]


def test_acquisition_age_expires_before_frame_age_limit():
    block, step = runner(maximum_detector_age_ms=150, maximum_detector_age_frames=100)
    step(0, True, [detection(0)])
    step(1, False)
    expired = step(2, False)
    assert expired.outputs["report"]["expired"] == [{"id": 1, "reason": "detector_age"}]
    assert not expired.outputs["report"]["active"]


def test_local_assignment_failure_never_creates_a_new_identity():
    _, step = runner(association_distance=0.1)
    first = step(0, True, [detection(0)])
    next_result = step(1, False)
    assert len(next_result.outputs["report"]["active"]) == 1
    assert (
        next_result.outputs["report"]["active"][0]["track_id"]
        == first.outputs["report"]["active"][0]["track_id"]
    )
    assert next_result.outputs["report"]["active"][0]["measurement_source"] == "prediction"


@pytest.mark.parametrize(
    "parameters",
    [
        {"local_min_points": 49},
        {"local_window": 4},
        {"local_max_points": 100},
        {"local_min_inliers": 0},
        {"fit_confidence": 1},
        {"local_error_metric": "unknown"},
    ],
)
def test_invalid_controls_or_native_work_ceiling_emit_no_result(parameters):
    block, step = runner()
    step(0, True, [detection(0)])
    block.parameters.update(parameters)
    assert block.execute({"image": scene(1), "refresh": False}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_native_failure_is_contained_and_recovery_resets_generation(monkeypatch):
    import opencv_dataflow.refresh_tracking as module

    block, step = runner()
    first = step(0, True, [detection(0)])
    native = module.cv2.calcOpticalFlowPyrLK
    monkeypatch.setattr(
        module.cv2,
        "calcOpticalFlowPyrLK",
        lambda *a, **k: (_ for _ in ()).throw(cv2.error("Controlled local failure")),
    )
    payload = {"spatial": frame([224, 112], "camera"), "ticket": TicketClock().issue("camera")}
    assert (
        block.execute(
            {"image": ValuePacket(scene(1), payload), "refresh": False, "timestamp_ms": 100}
        )
        is None
    )
    assert block.error_kind == "exception" and not block.memory["streams"]
    monkeypatch.setattr(module.cv2, "calcOpticalFlowPyrLK", native)
    recovered = step(2, True, [detection(2)])
    assert (
        recovered.outputs["report"]["active"][0]["track_id"]
        != first.outputs["report"]["active"][0]["track_id"]
    )


def test_offscreen_dropdowns_xml_and_graphics_off_do_not_execute_native(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml

    graph, _, tracker = graph_fixture()
    snapshot = graph.to_dict()
    path = tmp_path / "refresh.xml"
    export_project_xml(snapshot, path)
    registry = builtin_registry()
    registry.register(type(graph.blocks["source"]))
    loaded = Graph.from_dict(read_project_xml(path), registry)
    assert loaded.to_dict() == snapshot
    loaded.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        with monkeypatch.context() as guard:
            guard.setattr(
                cv2,
                "goodFeaturesToTrack",
                lambda *a, **k: pytest.fail("Editing invoked native work"),
            )
            window.inspector.select(tracker)
        assert all(
            isinstance(window.inspector.fields[n], QComboBox)
            for n in ("local_error_metric", "fit_method", "local_gradient_size")
        )
        assert not tracker.memory["streams"]
        tracker.set_graphics_mode("off")
        graph.run()
        assert tracker.last_result is not None and not tracker.last_result.drawings
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


def test_total_local_work_is_guarded_before_native_calls(monkeypatch):
    block, step = runner(local_work_limit=1)
    step(0, True, [detection(0)])
    monkeypatch.setattr(
        cv2, "calcOpticalFlowPyrLK", lambda *a, **k: pytest.fail("Work guard ran after native work")
    )
    with pytest.raises(AssertionError, match="Total local-flow work"):
        step(1, False)
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_selection_removes_local_evidence_and_refresh_ticket_must_match():
    block, step = runner()
    step(0, True, [detection(0)])
    payload = {"spatial": frame([224, 112], "camera"), "ticket": TicketClock().issue("camera")}
    result = block.execute(
        {
            "image": ValuePacket(scene(1), payload),
            "refresh": ValuePacket(False, payload),
            "timestamp_ms": 100,
            "mask": ValuePacket(np.zeros((112, 224), np.uint8), payload),
        }
    )
    assert result is not None, block.error
    assert result.outputs["count"] == 0
    assert result.outputs["report"]["active"][0]["measurement_source"] == "prediction"
    wrong = {**payload, "ticket": TicketClock().issue("other-camera")}
    assert (
        block.execute(
            {
                "image": ValuePacket(scene(2), payload),
                "refresh": ValuePacket(False, wrong),
                "timestamp_ms": 200,
            }
        )
        is None
    )
    assert block.error_kind == "invalid" and "ticket" in block.error


def test_toolbox_separates_control_flow_gate_from_opencv_tracker():
    from opencv_dataflow.toolbox import block_entry

    registry = builtin_registry()
    assert block_entry(registry.create("flow.detection_cadence")).path == (
        "Custom",
        "Control flow",
        "Scheduling",
    )
    path = block_entry(registry.create(TYPE)).path
    assert path[0] == "OpenCV" and "(video)" in path[1]


def test_unsupported_depth_is_invalid_before_native_work(monkeypatch):
    block = builtin_registry().create(TYPE)
    monkeypatch.setattr(
        cv2, "goodFeaturesToTrack", lambda *a, **k: pytest.fail("Invalid depth reached native work")
    )
    assert (
        block.execute({"image": scene(0).astype(np.uint16), "refresh": True, "detections": []})
        is None
    )
    assert block.error_kind == "invalid" and "uint8" in block.error
