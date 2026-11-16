"""Analytic motion, covariance, crossing assignments and clocked track lifecycle."""

from itertools import permutations

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.image_tracking import angle_delta, correct_track, predict_track
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import collection, validate_collection
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.tracking_assignment import associate_costs

TYPE = "sequence.track_image"


def detector(x, y=32, angle=None, size=None, label=None):
    result = {"center": [x, y]}
    if angle is not None:
        result["angle_degrees"] = angle
    if size is not None:
        result["size"] = size
    if label is not None:
        result["label"] = label
    return result


def runner(**parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    clock = TicketClock()

    def step(time, items, **extra):
        payload = {
            "spatial": frame([256, 96], "camera"),
            "ticket": clock.issue("camera"),
            "custom": {"batch": "A"},
        }
        values = {
            "image": ValuePacket(np.zeros((96, 256), np.uint8), payload),
            "detections": ValuePacket(items, payload),
            "timestamp_ms": ValuePacket(time, payload),
            **extra,
        }
        result = block.execute(values)
        assert result is not None, block.error
        return result

    return block, step


def test_native_prediction_and_joseph_correction_match_independent_matrix_reference():
    p = builtin_registry().create(TYPE).parameters
    mean = np.array([10, 20, np.log(20), np.log(12), 179, 8, -2, 0.1, -0.1, 20.0])
    cov = np.diag([4, 9, 0.01, 0.02, 25, 16, 25, 0.04, 0.09, 100.0])
    track = {"mean": mean, "covariance": cov, "size_known": True, "angle_known": True}
    dt = 0.2
    a = np.eye(10)
    a[:5, 5:] = np.eye(5) * dt
    q = np.zeros((10, 10))
    for index, acceleration in enumerate([10, 10, 0.1, 0.1, 30]):
        q[np.ix_([index, index + 5], [index, index + 5])] = acceleration**2 * np.array(
            [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]
        )
    predicted = a @ mean
    covariance = a @ cov @ a.T + q
    native = predict_track(track, dt, p)
    np.testing.assert_allclose(native.statePre[:, 0], predicted)
    np.testing.assert_allclose(native.errorCovPre, covariance)
    observed = predicted[:5] + [0.5, -0.2, 0.02, -0.01, 2]
    observed[4] = angle_delta(observed[4], 0, 360)
    item = {"value": observed, "available": np.ones(5, bool)}
    m, posterior = correct_track(native, item, track, p)
    h = np.eye(5, 10)
    r = np.diag([1, 1, 0.05**2, 0.05**2, 25])
    gain = np.linalg.solve(h @ covariance @ h.T + r, h @ covariance).T
    delta = np.array([0.5, -0.2, 0.02, -0.01, 2])
    expected = predicted + gain @ delta
    np.testing.assert_allclose(m, expected, atol=1e-12)
    remaining = np.eye(10) - gain @ h
    np.testing.assert_allclose(
        posterior, remaining @ covariance @ remaining.T + gain @ r @ gain.T, atol=1e-12
    )


def test_gated_assignment_maximizes_cardinality_then_cost_against_bruteforce():
    rng = np.random.default_rng(171)
    for _ in range(8):
        costs = rng.random((3, 4))
        allowed = rng.random((3, 4)) > 0.4
        matches = associate_costs(costs, allowed)
        actual = (-len(matches), sum(costs[r, c] for r, c in matches.items()))
        candidates = []
        for columns in permutations(range(7), 3):
            accepted = [costs[r, c] for r, c in enumerate(columns) if c < 4 and allowed[r, c]]
            candidates.append((-len(accepted), sum(accepted)))
        expected = min(candidates)
        assert actual[0] == expected[0] and actual[1] == pytest.approx(expected[1])


def test_center_only_tracks_preserve_identity_through_crossing_and_detection_reordering():
    _, step = runner(position_noise=0.1, position_acceleration=0, minimum_observations=2)
    identities = None
    for index, time in enumerate((0, 100, 200, 300, 400, 500, 600, 700, 800)):
        left, right = 40 + index * 12, 180 - index * 12
        measurements = [detector(left), detector(right)]
        if index % 2:
            measurements.reverse()
        result = step(time, measurements)
        items = result.outputs["tracks"]["items"]
        if index == 1:
            identities = [item["track_id"] for item in items]
        if index > 1:
            assert [item["track_id"] for item in items] == identities
            np.testing.assert_allclose(
                [item["center"][0] for item in items], [left, right], atol=0.25
            )
            assert all(not item["size_available"] and not item["angle_available"] for item in items)
            assert all(item["metrics"]["width"] is None for item in items)
        validate_collection(result.outputs["tracks"])


@pytest.mark.parametrize(
    "period,angles",
    [("360", [168, 174, -180, -174, -168, -162, -156]), ("180", [78, 84, -90, -84, -78, -72, -66])],
)
def test_size_angle_wrap_prediction_reacquisition_and_current_pose_payload(period, angles):
    block, step = runner(
        angle_period=period,
        position_noise=0.2,
        angle_noise=0.2,
        scale_noise=0.005,
        position_acceleration=0,
        angular_acceleration=0,
        scale_acceleration=0,
    )
    identity = None
    before = None
    for index, angle in enumerate(angles):
        missing = index in (3, 4)
        size = [20 * np.exp(0.02 * index), 12 * np.exp(0.01 * index)]
        result = step(index * 100, [] if missing else [detector(40 + 3 * index, 40, angle, size)])
        if index == 1:
            identity = result.outputs["tracks"]["items"][0]["track_id"]
        if index < 2:
            continue
        item = result.outputs["tracks"]["items"][0]
        assert item["track_id"] == identity
        assert item["observed"] is not missing
        assert abs(angle_delta(item["angle_degrees"], angle, float(period))) < 1.8
        np.testing.assert_allclose(item["center"], [40 + 3 * index, 40], atol=0.4)
        np.testing.assert_allclose(item["size"], size, rtol=0.015)
        if index == 2:
            before = item["covariance"][0][0]
        if missing:
            assert item["covariance"][0][0] > before
        assert np.linalg.eigvalsh(np.asarray(item["covariance"])).min() > -1e-9
        pose = item["pose"]
        assert pose["target_payload"]["ticket"] == result.payloads["tracks"]["ticket"]
        assert result.payloads["tracks"]["custom"] == {"batch": "A"}
    assert block.last_result.outputs["report"]["associations"][0]["id"] == 1


def test_optional_size_and_orientation_initialize_when_first_measured():
    _, step = runner(emit_tentative=True)
    first = step(0, [detector(30)]).outputs["tracks"]["items"][0]
    second = step(100, [detector(31, angle=150, size=[140, 80])]).outputs["tracks"]["items"][0]
    assert second["track_id"] == first["track_id"]
    np.testing.assert_allclose(second["size"], [140, 80], atol=1e-10)
    assert second["angle_degrees"] == pytest.approx(150)


@pytest.mark.parametrize(
    "limit,reason",
    [
        ({"maximum_missed_frames": 1}, "missed_frames"),
        ({"maximum_unobserved_ms": 150}, "unobserved_time"),
        ({"maximum_position_sigma": 12}, "position_uncertainty"),
    ],
)
def test_expiry_reports_reason_and_reacquisition_never_reuses_expired_identity(limit, reason):
    _, step = runner(emit_tentative=True, minimum_observations=1, **limit)
    identity = step(0, [detector(30)]).outputs["tracks"]["items"][0]["track_id"]
    step(100, [])
    expired = step(200, [])
    assert expired.outputs["count"] == 0
    assert expired.outputs["report"]["expired"][0]["reason"] == reason
    new = step(300, [detector(30)]).outputs["tracks"]["items"][0]
    assert new["track_id"] != identity and new["id"] == 2


def test_hard_angle_size_label_gates_and_capacity_never_merge_incompatible_parts():
    _, step = runner(
        emit_tentative=True,
        maximum_tracks=2,
        label_policy="required",
        maximum_angle_difference=10,
        maximum_scale_ratio=1.2,
    )
    initial = step(0, [detector(30, angle=0, size=[20, 10], label="A")]).outputs["tracks"]["items"][
        0
    ]
    result = step(
        100,
        [
            detector(30, angle=80, size=[20, 10], label="A"),
            detector(30, angle=0, size=[60, 10], label="B"),
        ],
    )
    items = result.outputs["tracks"]["items"]
    assert items[0]["track_id"] == initial["track_id"] and not items[0]["observed"]
    assert result.outputs["report"]["capacity_dropped"] == 1
    assert not result.outputs["report"]["associations"]


def test_irregular_intervals_and_long_gap_retire_before_native_prediction(monkeypatch):
    import opencv_dataflow.image_tracking as module

    _, step = runner(emit_tentative=True, position_noise=0.01, position_acceleration=0)
    for time in (0, 40, 150, 210, 390):
        result = step(time, [detector(25 + time * 0.08)])
    identity = result.outputs["tracks"]["items"][0]["track_id"]
    prediction = step(490, []).outputs["tracks"]["items"][0]
    assert prediction["center"][0] == pytest.approx(25 + 490 * 0.08, abs=0.1)
    monkeypatch.setattr(
        module, "predict_track", lambda *args: pytest.fail("Long gap reached native prediction")
    )
    result = step(4000, [detector(60)])
    assert result.outputs["report"]["expired"] == [{"id": 1, "reason": "interval_gap"}]
    assert result.outputs["tracks"]["items"][0]["track_id"] != identity


def test_ties_are_marked_and_prediction_does_not_erase_association_ambiguity():
    _, step = runner(emit_tentative=True, minimum_observations=1)
    step(0, [detector(30), detector(34)])
    result = step(100, [detector(32), detector(32)])
    assert all(item["association_ambiguous"] for item in result.outputs["tracks"]["items"])
    predicted = step(200, [])
    assert all(item["association_ambiguous"] for item in predicted.outputs["tracks"]["items"])


@pytest.mark.parametrize("invalid", [0, -10, float("nan")])
def test_bad_time_is_contained_and_recovery_starts_a_new_generation(invalid):
    block, step = runner(emit_tentative=True)
    old = step(0, [detector(20)]).outputs["tracks"]["items"][0]["track_id"]
    assert (
        block.execute(
            {
                "image": ValuePacket(
                    np.zeros((96, 256), np.uint8),
                    {
                        "spatial": frame([256, 96], "camera"),
                        "ticket": TicketClock().issue("camera"),
                    },
                ),
                "detections": [],
                "timestamp_ms": invalid,
            }
        )
        is None
    )
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())
    result = step(100, [detector(20)])
    assert result.outputs["tracks"]["items"][0]["track_id"] != old


def test_explicit_empty_detections_predict_and_ticket_mismatch_retires_state():
    block, step = runner(emit_tentative=True)
    step(0, [detector(20)])
    assert step(100, []).outputs["tracks"]["items"][0]["state"] == "tentative"
    clock = TicketClock()
    wrong = {"ticket": clock.issue("other")}
    assert (
        block.execute(
            {
                "image": ValuePacket(
                    np.zeros((96, 256), np.uint8), {"ticket": clock.issue("camera")}
                ),
                "detections": ValuePacket([], wrong),
                "timestamp_ms": 200,
            }
        )
        is None
    )
    assert block.error_kind == "invalid"
    assert block.last_result is None and not any(block.current_emissions.values())
    resumed = step(200, [])
    assert resumed.outputs["activations"] == 1


def test_nominal_and_media_clocks_xml_copy_reset_and_lean_graphics(tmp_path):
    registry, graph = builtin_registry(), Graph()
    block = graph.add(registry.create(TYPE, parameters={"emit_tentative": True}))
    image = ImageArray(np.zeros((48, 64), np.uint8), "gray", {"spatial": frame([64, 48], "camera")})
    try:
        for index in range(3):
            result = block.execute({"image": image, "detections": [detector(10 + index)]})
            assert result.outputs["report"]["clock"] == "nominal"
            assert result.outputs["report"]["timestamp_ms"] == index * 100
        assert result.drawings
        snapshot = graph.to_dict()
        path = tmp_path / "tracks.xml"
        export_project_xml(snapshot, path)
        reopened = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert reopened.to_dict() == snapshot
        finally:
            reopened.close()
        worker = Graph.from_dict(snapshot, registry)
        worker.copy_runtime_from(graph)
        try:
            other = worker.blocks[block.id].execute({"image": image, "detections": []})
            assert other.outputs["activations"] == 4
            assert block.last_result.outputs["activations"] == 3
        finally:
            worker.close()
        block.reset_runtime()
        block.set_graphics_mode("off")
        for time in (200, 350):
            result = block.execute(
                {
                    "image": ValuePacket(
                        image,
                        {**image.metadata, "video_frame": {"timestamp_ms": time, "frame_index": 1}},
                    ),
                    "detections": [],
                }
            )
            assert result.outputs["report"]["clock"] == "media" and not result.drawings
    finally:
        graph.close()


def test_collection_adapter_and_region_select_current_pixel_centers():
    block, step = runner(emit_tentative=True)
    image = ImageArray(
        np.zeros((96, 256), np.uint8), "gray", {"spatial": frame([256, 96], "camera")}
    )
    shapes = collection(
        "boxes",
        [
            {
                "id": 1,
                "metrics": {
                    "center_x": 30,
                    "center_y": 32,
                    "width": 20,
                    "height": 10,
                    "angle": -25,
                },
                "geometry": {},
            }
        ],
        image,
        image.metadata,
    )
    result = block.execute({"image": image, "detections": shapes, "timestamp_ms": 0})
    assert result is not None, block.error
    item = result.outputs["tracks"]["items"][0]
    assert item["angle_degrees"] == 25
    region = {
        "model": "image_region_v1",
        "image_size": [256, 96],
        "shape": "rectangle",
        "rectangle": [100, 0, 100, 90],
    }
    result = block.execute(
        {"image": image, "detections": shapes, "timestamp_ms": 100, "region": region}
    )
    assert result.outputs["count"] == 0 and result.outputs["report"]["active"][0]["misses"] == 1


def test_pattern_pose_preserves_trained_scale_and_offset_outline():
    from opencv_dataflow.poses import local_offset, make_pose

    block, step = runner(emit_tentative=True)
    payload = {"spatial": frame([256, 96], "camera")}
    transform = local_offset(60, 40, 25) @ np.diag([1.5, 1.5, 1])
    model_box = np.array([[-4, -6], [16, -6], [16, 4], [-4, 4]])
    corners = (np.c_[model_box, np.ones(4)] @ transform.T)[:, :2]
    pose = make_pose(transform, [256, 96], payload)
    matches = {
        "model": "pattern_matches_v1",
        "count": 1,
        "items": [{"pose": pose, "corners": corners.tolist(), "score": 0.9}],
    }
    result = step(0, matches)
    track = result.outputs["tracks"]["items"][0]
    np.testing.assert_allclose(track["pose"]["source_to_target"], transform, atol=1e-10)
    np.testing.assert_allclose(track["geometry"]["points"], corners, atol=1e-10)
    invalid = transform.copy()
    invalid[0, 1] += 0.5
    matches["items"][0]["pose"] = make_pose(invalid, [256, 96], payload)
    assert block.execute({"image": np.zeros((96, 256), np.uint8), "detections": matches}) is None
    assert "uniform affine" in block.error


def test_graph_holds_missing_detection_event_without_advancing_memory():
    from opencv_dataflow.core import Block, Port, Result
    from opencv_dataflow.payloads import NO_VALUE

    class Source(Block):
        outputs = (Port("image"), Port("detections", "data"))
        items = [detector(20)]

        def process(self, inputs):
            return Result({"image": np.zeros((96, 256), np.uint8), "detections": self.items})

    graph = Graph()
    source = graph.add(Source(block_id="source"))
    block = graph.add(
        builtin_registry().create(TYPE, block_id="tracker", parameters={"emit_tentative": True})
    )
    graph.connect("source", "image", "tracker", "image")
    graph.connect("source", "detections", "tracker", "detections")
    try:
        graph.run()
        assert block.last_result.outputs["activations"] == 1
        source.items = NO_VALUE
        graph.run()
        assert block.last_result is None and not block.last_inputs
        assert next(iter(block.memory["streams"].values()))["activations"] == 1
        assert not any(block.current_emissions.values())
        source.items = []
        graph.run()
        assert block.last_result.outputs["activations"] == 2
        assert block.last_result.outputs["report"]["active"][0]["misses"] == 1
    finally:
        graph.close()


def test_metadata_mismatch_is_rejected_before_native_prediction(monkeypatch):
    block, step = runner(emit_tentative=True)
    step(0, [detector(20)])

    def forbidden(*args):
        pytest.fail("Native prediction must not run on a wrong image grid")

    monkeypatch.setattr("opencv_dataflow.image_tracking.predict_track", forbidden)
    assert (
        block.execute(
            {
                "image": ValuePacket(
                    np.zeros((96, 256), np.uint8), {"spatial": frame([256, 96], "camera")}
                ),
                "detections": ValuePacket([], {"spatial": frame([256, 96], "other-camera")}),
            }
        )
        is None
    )
    assert block.error_kind == "invalid" and "frame" in block.error.lower()


@pytest.mark.parametrize(
    "measurement",
    [
        [{"center": [float("nan"), 20]}],
        [{"center": [20, 20], "size": [-1, 2]}],
        [{"center": [20, 20], "angle_degrees": float("inf")}],
        [{"center": [20, 20], "score": 2}],
        [{"center": [20, 20], "label": []}],
        {"model": "image_detections_v1", "count": 2, "items": []},
    ],
)
def test_invalid_detection_data_emits_nothing_and_discards_memory(measurement):
    block, step = runner(emit_tentative=True)
    step(0, [detector(20)])
    assert (
        block.execute({"image": np.zeros((96, 256), np.uint8), "detections": measurement}) is None
    )
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())
    assert not block.memory["streams"]


def test_native_exception_is_contained_and_recovery_does_not_reuse_identity(monkeypatch):
    import cv2

    import opencv_dataflow.image_tracking as module

    block, step = runner(emit_tentative=True)
    identity = step(0, [detector(20)]).outputs["tracks"]["items"][0]["track_id"]
    original = module.predict_track
    monkeypatch.setattr(
        module,
        "predict_track",
        lambda *args: (_ for _ in ()).throw(cv2.error("Controlled failure")),
    )
    assert (
        block.execute(
            {
                "image": ValuePacket(
                    np.zeros((96, 256), np.uint8),
                    {
                        "spatial": frame([256, 96], "camera"),
                        "ticket": TicketClock().issue("camera"),
                    },
                ),
                "detections": [],
                "timestamp_ms": 100,
            }
        )
        is None
    )
    assert block.error_kind == "exception" and not any(block.current_emissions.values())
    monkeypatch.setattr(module, "predict_track", original)
    assert step(100, [detector(20)]).outputs["tracks"]["items"][0]["track_id"] != identity


def test_offscreen_controls_do_not_create_native_state(app, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    import opencv_dataflow.image_tracking as module
    from opencv_dataflow.app import MainWindow

    graph = Graph()
    block = graph.add(builtin_registry().create(TYPE))
    monkeypatch.setattr(
        module.cv2, "KalmanFilter", lambda *args: pytest.fail("Editing created native state")
    )
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name in ("clock", "angle_period", "label_policy"):
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert block.documentation and block.description and not block.memory["streams"]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
