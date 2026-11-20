"""Finite and looping point-cloud replay preserve exact recorded scan provenance."""

import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import cloud_recording
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.cloud_recording import MANIFEST_FORMAT, RecordedCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.runner import main as run_project

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/recorded_cloud_sequence/recorded_cloud_sequence.json"
)
EPOCH = 1_900_000_000_000_000_000


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_sequence_uses_manifest_order_and_preserves_group_members(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        replay, batch, unbatch = (graph.blocks[key] for key in ("replay", "batch", "unbatch"))
        graph.set_run_profile(profile)
        graph.set_branch_workers(1 if profile == "debug" else 2)
        expected_names = ["z_entrance.ply", "middle.pcd", "a_review.obj"]
        expected_stamps = [EPOCH + n for n in (3, 45_000_007, 105_000_017)]
        observations = []
        for index, (name, stamp) in enumerate(zip(expected_names, expected_stamps, strict=True)):
            assert graph.has_finite_sources() and not graph.sources_exhausted()
            graph.run()
            assert replay.state == "pass", replay.error
            result = replay.last_result
            cloud, info = result.outputs["cloud"], result.outputs["info"]
            assert info["scan_count"] == 3 and info["point_count"] == 5
            assert info["scan_index"] == index and info["path"] == name
            assert info["clock"] == "scanner_clock" and info["timestamp_ns"] == stamp
            context = cloud.metadata["recorded_cloud"]
            assert context == result.payloads["cloud"]["recorded_cloud"]
            assert context == result.payloads["info"]["recorded_cloud"]
            assert (cloud.frame, cloud.units) == [
                ("conveyor", "mm"),
                ("conveyor", "m"),
                ("tool_camera", "m"),
            ][index]
            first_point = ([0, 0, 0], [0.03, 0, 0], [0.01, 0.02, 0.25])[index]
            np.testing.assert_allclose(cloud.points[0], first_point, atol=1e-8)
            expected_ids = [21, 16, 8, 31, 13] if index == 1 else np.arange(5)
            np.testing.assert_array_equal(cloud.attributes["source_index"], expected_ids)
            if index < 2:
                np.testing.assert_array_equal(cloud.colors[0], [32, 172, 200])
                assert batch.last_result.outputs["group"] is NO_VALUE
            assert not replay._pending
            observations.append((context, result.outputs["ticket"]))

        assert graph.sources_exhausted()
        group = batch.last_result.outputs["group"]
        emitted = unbatch.current_emissions["value"]
        assert len(group.items) == len(emitted) == 3
        for item, packet, (provenance, ticket) in zip(
            group.items, emitted, observations, strict=True
        ):
            assert item.payload["ticket"] == packet.payload["ticket"] == ticket
            assert item.payload["recorded_cloud"] == packet.payload["recorded_cloud"] == provenance
            np.testing.assert_array_equal(item.value.points, packet.value.points)
        assert len({ticket["sequence"] for _, ticket in observations}) == 3

        graph.run()
        assert not any(replay.current_emissions.values())
        assert replay.last_result.outputs["cloud"] is NO_VALUE
        replay.configure_replay("loop", rewind=True)
        for index in (0, 1, 2, 0):
            graph.run()
            result = replay.last_result
            assert result.outputs["info"]["scan_index"] == index
            assert result.outputs["info"]["timestamp_ns"] == expected_stamps[index]
            assert result.outputs["ticket"]["sequence"] > observations[-1][1]["sequence"]
        assert not graph.sources_exhausted()
    finally:
        graph.close()


def _recording(directory):
    for i in (0, 1):
        write_cloud(
            directory / f"scan-{i}.ply",
            PointCloud([[i, 30, 100], [i, 32, 100]], "mm", "table"),
            binary=True,
        )
    return {
        "format": MANIFEST_FORMAT,
        "clock": "capture_clock",
        "scans": [
            {"path": f"scan-{i}.ply", "timestamp_ns": EPOCH + i, "frame": "table", "units": "mm"}
            for i in (0, 1)
        ],
    }


def _manifest(directory, data):
    path = directory / "recording.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_playback_state_handoff_and_project_save_as_keep_original_recording(tmp_path):
    manifest = _manifest(tmp_path, _recording(tmp_path))
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.cloud_recording",
            block_id="source",
            parameters={"path": manifest.name, "track_source_rows": False},
        )
    )
    try:
        graph.save(tmp_path / "original.json")
        graph.run()
        assert not source.last_result.outputs["cloud"].attributes
        state = source.snapshot_runtime()
        clone = registry.create(source.type_id, parameters=dict(source.parameters))
        clone.project_dir = source.project_dir
        clone.copy_runtime_from(source)
        clone_result = clone.execute({})
        assert clone_result.outputs["info"]["scan_index"] == 1
        assert clone.exhausted
        graph.run()
        assert source.exhausted
        source.restore_runtime(state)
        assert not source.exhausted
        graph.run()
        assert source.last_result.outputs["info"]["scan_index"] == 1

        other = tmp_path / "move"
        other.mkdir()
        graph.save(other / "moved.json")
        moved = Graph.load(other / "moved.json", registry)
        try:
            moved.run()
            assert moved.blocks[source.id].last_result.outputs["info"]["scan_index"] == 0
        finally:
            moved.close()
        assert not list(other.glob("scan-*.ply"))
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("timestamp_ns", EPOCH - 1, "strictly increasing"),
        ("timestamp_ns", True, "strictly increasing"),
        ("timestamp_ns", 2**63, "strictly increasing"),
        ("path", "../escape.ply", "relative PLY"),
        ("path", "G:/outside.pcd", "relative PLY"),
        ("path", "assets\\escape.obj", "relative PLY"),
        ("units", "cm", "m/mm/unitless"),
        ("frame", "", "needs a frame"),
    ],
)
def test_invalid_manifest_halts_before_issuing_a_ticket(tmp_path, field, value, diagnostic):
    record = _recording(tmp_path)
    record["scans"][1][field] = value
    block = RecordedCloud(parameters={"path": str(_manifest(tmp_path, record))})
    assert block.execute({}) is None and diagnostic in block.error
    assert block.exhausted and block.replay_stalled and block._source_ticket is None
    assert not block.current_emissions


@pytest.mark.parametrize(
    ("fault", "diagnostic"),
    [
        ("missing", "Scan file is missing"),
        ("corrupted", "Truncated binary PLY"),
        ("frame", "frame or units differ"),
        ("units", "frame or units differ"),
        ("empty", "no usable points"),
        ("point_budget", "Vertex count"),
    ],
)
def test_bad_scan_withholds_cloud_and_ticket_in_loop_mode(tmp_path, fault, diagnostic):
    record = _recording(tmp_path)
    cfg = {"path": str(_manifest(tmp_path, record)), "repeat": "loop"}
    scan = tmp_path / "scan-0.ply"
    if fault == "missing":
        scan.unlink()
    elif fault == "corrupted":
        scan.write_bytes(scan.read_bytes()[:-1])
    elif fault in ("frame", "units"):
        defaults = {"frame": "other_camera", "units": "m"}
        record["scans"][0][fault] = defaults[fault]
        _manifest(tmp_path, record)
    elif fault == "empty":
        scan.unlink()
        write_cloud(scan, PointCloud(np.empty((0, 3)), "mm", "table"), binary=True)
    else:
        cfg["max_points"] = 1
    block = RecordedCloud(parameters=cfg)
    assert block.execute({}) is None and diagnostic in block.error
    assert block.replay_stalled and not block.exhausted
    assert block._source_ticket is None and not block.current_emissions
    paused = block.execute({})
    assert paused.outputs["cloud"] is NO_VALUE and paused.outputs["ticket"] is NO_VALUE


def test_late_missing_scan_exits_finite_cli_run_after_one_reported_error(tmp_path):
    manifest = _manifest(tmp_path, _recording(tmp_path))
    (tmp_path / "scan-1.ply").unlink()
    graph = Graph()
    graph.add(RecordedCloud(block_id="replay", parameters={"path": manifest.name}))
    path = tmp_path / "project.json"
    graph.save(path)
    graph.close()
    receipt = tmp_path / "report.json"
    assert run_project([str(path), "--all-images", "--report", str(receipt)]) == 1
    data = json.loads(receipt.read_bytes())
    assert [row["passed"] for row in data["runs"]] == [True, False]
    fault = data["runs"][1]["blocks"][0]
    assert fault["state"] == "error" and fault["emitted"] == {}
    assert "Scan 2 (scan-1.ply)" in fault["message"]


def test_finite_cli_does_not_retry_invalid_source_parameters_forever(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, _recording(tmp_path))
    graph = Graph()
    graph.add(RecordedCloud(block_id="replay", parameters={"path": manifest.name, "max_points": 0}))
    path = tmp_path / "project.json"
    graph.save(path)
    graph.close()
    original_run = Graph.run
    calls = []

    def run_once(current):
        assert not calls, "Headless finite replay tried to repeat an invalid input configuration"
        calls.append(True)
        return original_run(current)

    monkeypatch.setattr(Graph, "run", run_once)
    receipt = tmp_path / "report.json"
    assert run_project([str(path), "--all-images", "--report", str(receipt)]) == 1
    data = json.loads(receipt.read_bytes())
    assert len(data["runs"]) == 1 and data["runs"][0]["blocks"][0]["state"] == "error"


def test_cancelled_read_keeps_cursor_and_next_execution_retries_scan(tmp_path):
    block = RecordedCloud(parameters={"path": str(_manifest(tmp_path, _recording(tmp_path)))})
    checks = iter((False, True))
    cancelled = block.execute({}, cancelled=lambda: next(checks, True))
    assert cancelled.outputs["cloud"] is NO_VALUE and cancelled.outputs["ticket"] is NO_VALUE
    assert not block.exhausted
    assert block.execute({}).outputs["info"]["scan_index"] == 0
    assert block.execute({}).outputs["info"]["scan_index"] == 1
    assert block.exhausted


def test_manifest_read_is_bounded_and_duplicate_keys_reject(tmp_path, monkeypatch):
    path = _manifest(tmp_path, _recording(tmp_path))
    path.write_bytes(path.read_bytes().replace(b'"scans":', b'"clock": "second", "scans":'))
    block = RecordedCloud(parameters={"path": str(path)})
    assert block.execute({}) is None and "Duplicate JSON field" in block.error
    monkeypatch.setattr(cloud_recording, "_MAX_MANIFEST_BYTES", 64)
    block.reset_runtime()
    assert block.execute({}) is None and "at most 8 MiB" in block.error
