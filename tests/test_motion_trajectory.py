"""Content/timing binding, exact replay geometry, persistence and no-fit lifecycle."""

import copy
import hashlib

import cv2
import numpy as np
import pytest

from opencv_dataflow import motion_trajectory, motion_trajectory_io
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.motion_trajectory import capture_trajectory, seal_trajectory
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.stabilization import _transform_payload
from opencv_dataflow.tickets import TicketClock

SHIFTS = ((0, 0), (2, -1), (-1, 2))


def fixture(kind="gray8", root="camera", media=False):
    clock = TicketClock()
    base = np.random.default_rng(167).integers(10, 180, (48, 64), np.uint8)
    encoding = "gray"
    if kind.startswith("rgb"):
        base = np.dstack((base, base + 1, base + 2))
        encoding = "rgb"
        if kind.startswith("rgba"):
            base = np.dstack((base, np.full((48, 64), 220, np.uint8)))
            encoding = "rgba"
    dtype = {"16": np.uint16, "32": np.float32, "64": np.float64}.get(kind[-2:], np.uint8)
    base = base.astype(dtype)
    if kind.endswith("16"):
        base *= 257
    elif dtype != np.uint8:
        base /= 123
    packets, transforms = [], []
    for i, (dx, dy) in enumerate(SHIFTS):
        context = {
            "spatial": frame([64, 48], root),
            "ticket": clock.issue("camera", "external", f"{root}-{i}"),
            "custom": {"frame": i},
        }
        if media:
            context["video_frame"] = {"frame_index": 10 + i * 2, "timestamp_ms": 400 + i * 80.0}
        image = ImageArray(np.roll(base, (dy, dx), axis=(0, 1)), encoding)
        matrix = np.array([[1.0, 0, -dx], [0, 1.0, -dy], [0, 0, 1.0]])
        target = frame([64, 48], "trained-target")
        transform = _transform_payload(matrix, [64, 48], context, target)
        packets.append(ValuePacket(image, context))
        transforms.append(ValuePacket(transform, copy.deepcopy(context)))
    labels = tuple(f"frame-{i}" for i in range(3))
    group = FlowGroup(tuple(packets), labels)
    fit = FlowGroup(tuple(transforms), labels)
    record = capture_trajectory(
        group, fit, timestamp_mode="media" if media else "indexed", interpolation="nearest"
    )
    return group, fit, record, base


def replay(group, record, **parameters):
    block = builtin_registry().create("sequence.replay_trajectory", parameters=parameters)
    result = block.execute({"frames": group, "trajectory": record})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("kind", ["gray8", "gray16", "gray32", "rgb64", "rgba16"])
@pytest.mark.parametrize("media", [False, True])
def test_exact_replay_matches_independent_shift_oracle_and_retains_current_tickets(kind, media):
    group, _, record, base = fixture(kind, media=media)
    current, _, _, _ = fixture(kind, root="fresh-acquisition", media=media)
    original = copy.deepcopy(current)
    block, result = replay(current, record)
    assert result.passed and result.outputs["report"]["status"] == "complete"
    y, x = np.indices(base.shape[:2])
    for i, (packet, mask, transform) in enumerate(
        zip(
            result.outputs["corrected"].items,
            result.outputs["coverage"].items,
            result.outputs["transforms"].items,
            strict=True,
        )
    ):
        dx, dy = SHIFTS[i]
        valid = (x + dx >= 0) & (x + dx < 64) & (y + dy >= 0) & (y + dy < 48)
        expected = base.copy()
        expected[~valid] = 0
        np.testing.assert_array_equal(packet.value, expected)
        np.testing.assert_array_equal(mask.value, valid.astype(np.uint8) * 255)
        assert (
            packet.value.dtype == base.dtype
            and packet.value.color_format == group.items[0].value.color_format
        )
        assert packet.payload["ticket"] == current.items[i].payload["ticket"]
        assert packet.payload["ticket"] != group.items[i].payload["ticket"]
        assert packet.payload["custom"] == {"frame": i}
        assert (
            packet.payload["spatial"]["root_id"] == "trajectory:" + record["trajectory_id"] + ":0"
        )
        assert transform.value["source_spatial"]["root_id"] == "fresh-acquisition"
        np.testing.assert_array_equal(
            transform.value["source_to_target"], [[1, 0, -dx], [0, 1, -dy], [0, 0, 1]]
        )
        assert not np.shares_memory(packet.value, current.items[i].value)
    for before, after in zip(original.items, current.items, strict=True):
        np.testing.assert_array_equal(before.value, after.value)
        assert before.payload == after.payload
    assert result.outputs["image"].shape == base.shape
    # No cursor/cache survives reset or a new complete clip activation.
    block.reset_runtime()
    repeated = block.execute({"frames": current, "trajectory": record})
    np.testing.assert_array_equal(repeated.outputs["image"], result.outputs["image"])


@pytest.mark.parametrize(
    "bad",
    [
        "pixels",
        "order",
        "time",
        "media_index",
        "grid",
        "pyramid",
        "depth",
        "direction",
        "digest",
        "singular",
        "validity",
        "nonfinite",
    ],
)
def test_mismatched_or_corrupt_clip_rejects_before_native_warp(bad, monkeypatch):
    group, _, record, _ = fixture(media=True)
    if bad == "pixels":
        group.items[1].value[10, 10] += 1
    elif bad == "order":
        group = FlowGroup(group.items[::-1], group.labels[::-1])
    elif bad == "time":
        group.items[1].payload["video_frame"]["timestamp_ms"] += 0.1
    elif bad == "media_index":
        group.items[1].payload["video_frame"]["frame_index"] += 1
    elif bad == "grid":
        group.items[0].payload["spatial"]["to_root"][0][2] = 1
    elif bad == "pyramid":
        group.items[0].payload["pyramid"] = {
            "level": 1,
            "largest_size": [128, 96],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        }
    elif bad == "depth":
        group = FlowGroup(
            tuple(ValuePacket(p.value.astype(np.uint16), p.payload) for p in group.items),
            group.labels,
        )
    elif bad == "direction":
        record["direction"] = "target_to_source"
    elif bad == "digest":
        record["trajectory_id"] = "0" * 64
    elif bad == "singular":
        record["frames"][1]["source_to_target"][0][0] = 0
    elif bad == "validity":
        record["frames"][1]["valid"] = "true"
    else:
        record["frames"][2]["source_to_target"][0][0] = float("nan")
    monkeypatch.setattr(
        motion_trajectory,
        "_warp_correction",
        lambda *a: pytest.fail("Invalid replay entered native warp"),
    )
    block = builtin_registry().create("sequence.replay_trajectory")
    assert block.execute({"frames": group, "trajectory": record}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


@pytest.mark.parametrize("policy", ["reject", "fail_partial", "skip"])
def test_explicit_invalid_frames_never_invent_identity_motion(policy):
    group, fit, _, _ = fixture()
    fit.items[1].value = {"valid": False, "reason": "occluded"}
    record = capture_trajectory(group, fit, interpolation="nearest")
    _, result = replay(group, record, invalid_frames=policy, preview_frame=1)
    if policy == "reject":
        assert not result.passed and result.outputs["corrected"] is NO_VALUE
    else:
        assert result.passed == (policy == "skip")
        assert len(result.outputs["corrected"].items) == 2
        assert result.outputs["report"]["omitted_indices"] == [1]
        assert result.outputs["image"] is NO_VALUE
        assert result.outputs["corrected"].labels == (group.labels[0], group.labels[2])


def test_bad_capture_ticket_inverse_and_clock_are_invalid():
    group, fit, _, _ = fixture(media=True)
    block = builtin_registry().create(
        "sequence.capture_trajectory", parameters={"timestamp_mode": "media"}
    )
    fit.items[1].payload["ticket"] = fit.items[0].payload["ticket"]
    assert (
        block.execute({"frames": group, "transforms": fit}) is None
        and block.error_kind == "invalid"
    )
    group, fit, _, _ = fixture(media=True)
    fit.items[1].value["target_to_source"][0][2] += 1
    assert (
        block.execute({"frames": group, "transforms": fit}) is None
        and block.error_kind == "invalid"
    )
    group, fit, _, _ = fixture(media=True)
    group.items[1].payload["video_frame"]["timestamp_ms"] = group.items[0].payload["video_frame"][
        "timestamp_ms"
    ]
    assert (
        block.execute({"frames": group, "transforms": fit}) is None
        and block.error_kind == "invalid"
    )


def test_native_fit_capture_replay_is_pixel_identical_without_refitting(monkeypatch):
    from opencv_dataflow import offline_stabilization

    group, _, _, _ = fixture()
    native = builtin_registry().create(
        "sequence.stabilize_batch",
        parameters={"min_inliers": 6, "motion_model": "translation", "interpolation": "nearest"},
    )
    fitted = native.execute({"frames": group})
    assert fitted is not None and fitted.passed, native.error
    capture = builtin_registry().create(
        "sequence.capture_trajectory", parameters={"interpolation": "nearest"}
    )
    saved = capture.execute({"frames": group, "transforms": fitted.outputs["transforms"]})
    assert saved is not None, capture.error
    monkeypatch.setattr(
        offline_stabilization, "_estimate_pose", lambda *a: pytest.fail("Replay refitted")
    )
    _, result = replay(group, saved.outputs["trajectory"])
    for actual, expected in zip(
        result.outputs["corrected"].items, fitted.outputs["corrected"].items, strict=True
    ):
        np.testing.assert_array_equal(actual.value, expected.value)


def test_json_files_pin_bytes_preserve_existing_and_copy_on_save_as(app, tmp_path):
    _, _, record, _ = fixture()
    destination = tmp_path / "original"
    (destination / "assets").mkdir(parents=True)
    path = destination / "assets/trajectory.json"
    registry = builtin_registry()
    writer = registry.create("output.motion_trajectory", parameters={"path": str(path)})
    assert writer.execute({"trajectory": record}) is not None, writer.error
    raw = path.read_bytes()
    assert writer.execute({"trajectory": record}) is None and path.read_bytes() == raw
    writer.parameters["write_mode"] = "replace"
    assert writer.execute({"trajectory": record}) is not None
    assert not list(path.parent.glob(".trajectory-*.tmp"))
    graph = Graph()
    loader = graph.add(
        registry.create(
            "sequence.load_trajectory",
            parameters={
                "path": "assets/trajectory.json",
                "expected_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
    )
    replay_block = graph.add(
        registry.create("sequence.replay_trajectory", parameters={"invalid_frames": "skip"})
    )
    graph.save(destination / "program.json")
    graph.save(tmp_path / "copy/program.json")
    assert (tmp_path / "copy/assets/trajectory.json").read_bytes() == raw
    reopened = Graph.load(tmp_path / "copy/program.json", registry)
    assert reopened.blocks[loader.id].execute({}).outputs["trajectory"] == record
    reopened.close()
    xml = tmp_path / "replay.sequence.replay_trajectory.xml"
    save_block_xml(replay_block, xml)
    assert read_block_xml(xml)["parameters"] == replay_block.parameters
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(replay_block)
        assert isinstance(window.inspector.fields["interpolation"], QComboBox)
        assert isinstance(window.inspector.fields["invalid_frames"], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


@pytest.mark.parametrize(
    "raw", [b'{"x":0,"x":1}', b'{"x":NaN}', b"\xef\xbb\xbf{}", b"[" + b" " * (1024**2)]
)
def test_bad_files_are_rejected_without_events(raw, tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    block = builtin_registry().create("sequence.load_trajectory", parameters={"path": str(path)})
    assert block.execute({}) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())


def test_native_failure_is_contained_and_replay_recovers(monkeypatch):
    group, _, record, _ = fixture()
    native = motion_trajectory._warp_correction

    def broken(*a):
        raise cv2.error("Controlled trajectory warp failure")

    monkeypatch.setattr(motion_trajectory, "_warp_correction", broken)
    block = builtin_registry().create("sequence.replay_trajectory")
    assert block.execute({"frames": group, "trajectory": record}) is None
    assert block.error_kind == "exception" and not any(block.current_emissions.values())
    monkeypatch.setattr(motion_trajectory, "_warp_correction", native)
    assert block.execute({"frames": group, "trajectory": record}).passed


def test_single_frame_trajectory_and_stable_segment_roots():
    group, fit, _, _ = fixture()
    one = FlowGroup(group.items[:1], group.labels[:1])
    transform = FlowGroup(fit.items[:1], fit.labels[:1])
    record = capture_trajectory(one, transform)
    _, result = replay(one, record)
    assert result.passed and len(result.outputs["corrected"].items) == 1
    fit.items[2].value["target_spatial"]["root_id"] = "scene-cut"
    segmented = capture_trajectory(group, fit)
    _, result = replay(group, segmented)
    roots = [p.payload["spatial"]["root_id"] for p in result.outputs["corrected"].items]
    assert roots[0] == roots[1] and roots[2] != roots[1]


def test_segment_cannot_claim_conflicting_root_extents():
    _, _, record, _ = fixture()
    record["frames"][1]["target_spatial"]["root_size"][0] += 1
    with pytest.raises(ValueError, match="consistent target root size"):
        seal_trajectory(record)


@pytest.mark.parametrize("policy", ["reject", "fail_partial", "skip"])
def test_coverage_policy_and_all_invalid_batches(policy):
    group, fit, record, _ = fixture()
    record["frames"][1]["source_to_target"][0][2] = 1000
    record = seal_trajectory(record)
    _, result = replay(group, record, invalid_frames=policy)
    if policy == "reject":
        assert not result.passed and result.outputs["corrected"] is NO_VALUE
    else:
        assert result.passed == (policy == "skip") and len(result.outputs["corrected"].items) == 2
    for p in fit.items:
        p.value = {"valid": False, "reason": "no reliable motion"}
    invalid = capture_trajectory(group, fit)
    _, result = replay(group, invalid, invalid_frames=policy)
    assert not result.passed and result.outputs["corrected"] is NO_VALUE


def test_atomic_failure_preserves_previous_file_and_cleans_owned_temporary(tmp_path, monkeypatch):
    _, _, record, _ = fixture()
    path = tmp_path / "trajectory.json"
    old = b"previous reviewed bytes"
    path.write_bytes(old)
    block = builtin_registry().create(
        "output.motion_trajectory", parameters={"path": str(path), "write_mode": "replace"}
    )

    def blocked(*a):
        raise OSError("Controlled publication failure")

    monkeypatch.setattr(motion_trajectory_io.os, "replace", blocked)
    assert block.execute({"trajectory": record}) is None
    assert path.read_bytes() == old and not list(tmp_path.glob(".trajectory-*.tmp"))
