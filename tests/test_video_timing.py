"""Native explicit-time clips, clock policies and bounded stream diagnostic ownership."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/video_timing"


def monitor(**parameters):
    return builtin_registry().create("sequence.video_timing", parameters=parameters)


def sample(index, stamp, *, raw="same", source="decoder", fps=25, ticket=None):
    info = {
        "filename": "test.avi",
        "frame_index": index,
        "timestamp_ms": stamp,
        "decoder_timestamp_ms": stamp if raw == "same" else raw,
        "timestamp_source": source,
        "fps": fps,
        "backend": "test",
    }
    context = {"video_frame": info, "spatial": frame([40, 30], "camera")}
    if ticket is not None:
        context["ticket"] = ticket
    return {"image": ValuePacket(ImageArray(np.full((30, 40, 3), 100, np.uint8), "bgr"), context)}


@pytest.mark.parametrize(
    "clip,errors",
    [
        ("constant", 0),
        ("variable", 0),
        ("duplicate", 2),
        ("backward", 1),
        ("zero_time", 7),
    ],
)
def test_real_container_times_and_unreliable_native_properties_are_observed_without_repair(
    clip, errors
):
    truth = json.loads((BASE / "truth.json").read_bytes())
    source = builtin_registry().create(
        "input.video_file",
        parameters={
            "path": str(BASE / f"assets/{clip}.mkv"),
            "backend": "ffmpeg",
        },
    )
    block = monitor(reference_timestamps_ms=truth["clocks_ms"][clip])
    try:
        for index, expected in enumerate(truth["clocks_ms"][clip]):
            capture = source.execute({})
            assert capture is not None, source.error
            assert capture.outputs["info"]["decoder_timestamp_ms"] == pytest.approx(expected)
            result = block.execute(
                {"image": ValuePacket(capture.outputs["image"], capture.payloads["image"])}
            )
            assert result is not None, block.error
            np.testing.assert_array_equal(result.outputs["image"], capture.outputs["image"])
            assert result.payloads["image"]["ticket"] == capture.outputs["ticket"]
        report = result.outputs["report"]
        assert report["counts"]["frames"] == 8 and report["counts"]["rejected"] == errors
        assert result.passed is (errors == 0) and result.outputs["ok"] is (errors == 0)
        if clip == "variable":
            assert report["cadence"] == "variable" and result.outputs["variable_rate"]
        if clip == "zero_time":
            assert report["counts"]["estimated"] == 7
            assert [r["decoder_timestamp_ms"] for r in report["rows"]] == [0] * 8
            assert [r["effective_timestamp_ms"] for r in report["rows"]] == list(range(8))
        assert source.exhausted
    finally:
        source.close()
        block.close()


@pytest.mark.parametrize(
    "cadence,passed",
    [("allow_variable", True), ("require_constant", False), ("match_nominal_fps", False)],
)
def test_variable_cadence_and_wrong_nominal_rate_have_explicit_acceptance_policy(cadence, passed):
    block = monitor(cadence_policy=cadence)
    for i, stamp in enumerate((0, 40, 130, 160)):
        result = block.execute(sample(i, stamp))
        assert result is not None, block.error
    assert result.passed is passed and result.outputs["variable_rate"]
    assert "nominal_rate_mismatch" in result.outputs["report"]["current"]["warnings"] or not passed


def test_wrong_declared_fps_is_rejected_even_with_constant_actual_clock():
    block = monitor(cadence_policy="match_nominal_fps")
    for i in range(4):
        result = block.execute(sample(i, i * 40, fps=10))
        assert result is not None, block.error
    report = result.outputs["report"]
    assert not result.passed and report["cadence"] == "constant"
    assert report["error_counts"]["nominal_rate_mismatch"] == 3


def test_required_nominal_clock_and_extreme_metadata_are_diagnosed_without_overflow():
    block = monitor(cadence_policy="match_nominal_fps")
    result = block.execute(sample(0, 0, fps=None))
    assert result is not None, block.error
    assert (
        not result.passed and "missing_nominal_fps" in result.outputs["report"]["current"]["errors"]
    )
    block = monitor()
    result = block.execute(sample(2**2000, 10**500, fps=10**500))
    assert result is not None, block.error
    assert not result.passed and result.outputs["report"]["current"]["frame_index"] is None


def test_reconnected_network_stream_can_restart_media_time_at_global_nonzero_index():
    values = sample(8, 0)
    info = values["image"].payload["video_frame"]
    info["connection_frame_index"] = 0
    result = monitor().execute(values)
    assert result is not None and result.passed


def test_decoder_label_cannot_hide_a_conflicting_effective_position():
    block = monitor()
    result = block.execute(sample(0, 20, raw=0))
    assert result is not None, block.error
    assert not result.passed
    assert "decoder_effective_mismatch" in result.outputs["report"]["current"]["errors"]


def test_encoded_video_group_preserves_raw_clock_for_the_same_monitor():
    registry = builtin_registry()
    decoded = registry.create("cv.decode_video").execute(
        {
            "encoded": (BASE / "assets/variable.mkv").read_bytes(),
        }
    )
    assert decoded is not None
    block = monitor()
    for packet in decoded.outputs["frames"].items:
        assert "decoder_timestamp_ms" in packet.payload["video_frame"]
        result = block.execute({"image": packet})
        assert result is not None, block.error
    assert result.passed and result.outputs["report"]["cadence"] == "variable"


@pytest.mark.parametrize("policy,passed", [("decoder_only", False), ("allow_estimates", True)])
def test_missing_raw_clock_cannot_masquerade_as_measured_time(policy, passed):
    block = monitor(timestamp_policy=policy)
    for i in range(4):
        result = block.execute(sample(i, i * 40, raw=None, source="frame_rate_estimate"))
        assert result is not None, block.error
    assert result.passed is passed
    row = result.outputs["report"]["current"]
    assert row["estimated"] and row["decoder_timestamp_ms"] is None
    assert row["timestamp_ms"] == (120 if passed else None)
    assert "estimated_timestamp" in (row["warnings"] if passed else row["errors"])


def test_history_eviction_retains_session_error_counts_until_explicit_reset():
    block = monitor(history_size=3)
    for i, stamp in enumerate((0, 0, 80, 120, 160, 200, 240)):
        result = block.execute(sample(i, stamp))
        assert result is not None, block.error
    report = result.outputs["report"]
    assert len(report["rows"]) == 3 and report["counts"]["frames"] == 7
    assert report["counts"]["rejected"] == 1 and not result.passed and report["current_ok"]
    result = block.execute({**sample(0, 0), "reset": True})
    assert result.passed and result.outputs["report"]["counts"]["frames"] == 1


def test_non_latching_mode_reports_recovery_but_keeps_historical_quality_score():
    block = monitor(latch_failures=False)
    assert block.execute(sample(0, 0)).passed
    assert not block.execute(sample(1, 0)).passed
    result = block.execute(sample(2, 80))
    assert result.passed and result.score == pytest.approx(2 / 3)
    assert not result.fail_reasons


def test_strided_frames_normalize_elapsed_time_without_inventing_drop_counts():
    block = monitor(cadence_policy="match_nominal_fps")
    for i in (0, 2, 4, 6):
        result = block.execute(sample(i, i * 40))
        assert result is not None, block.error
    assert result.passed and result.outputs["report"]["interval_per_frame_ms"]["median"] == 40
    assert result.outputs["report"]["current"]["delta_ms"] == 80


def test_independent_reference_disagreement_and_reference_change_reset_context():
    block = monitor(reference_timestamps_ms=[0, 40, 80, 120])
    for i, stamp in enumerate((0, 40, 130)):
        result = block.execute(sample(i, stamp))
        assert result is not None, block.error
    assert not result.passed and result.outputs["report"]["current"]["reference_error_ms"] == 50
    result = block.execute({**sample(2, 130), "reference": [0, 40, 130]})
    assert result.passed and result.outputs["report"]["counts"]["frames"] == 1


def test_timeline_chart_has_its_own_grid_and_graphics_off_skips_rendering(monkeypatch):
    from opencv_dataflow import video_timing

    block = monitor()
    result = block.execute(sample(0, 0))
    assert result.outputs["chart"].shape == (320, 640, 3)
    assert result.payloads["chart"]["spatial"] != result.payloads["image"]["spatial"]
    assert result.drawings and result.graphics
    block.graphics_mode = "off"
    monkeypatch.setattr(
        video_timing, "timing_chart", lambda rows: pytest.fail("Graphics off rendered")
    )
    result = block.execute(sample(1, 40))
    assert result is not None, block.error
    assert result.outputs["chart"] is NO_VALUE and not result.drawings and not result.graphics
    assert result.passed


def test_mismatched_info_is_contained_and_does_not_emit_or_mutate_source_image():
    block = monitor()
    values = sample(0, 0)
    before = values["image"].value.copy()
    assert block.execute({**values, "info": {"frame_index": 1}}) is None
    assert block.error_kind == "invalid" and "does not match" in block.error
    assert not any(block.current_emissions.values())
    np.testing.assert_array_equal(values["image"].value, before)


def test_identical_info_values_from_another_ticket_cannot_describe_the_current_image():
    from opencv_dataflow.tickets import TicketClock

    clock = TicketClock()
    values = sample(0, 0, ticket=clock.issue("video"))
    info = values["image"].payload["video_frame"]
    result = monitor().execute(
        {**values, "info": ValuePacket(info, {"ticket": clock.issue("video")})}
    )
    assert result is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"history_size": 2},
        {"warmup": 0},
        {"latch_failures": "yes"},
        {"timestamp_policy": "invent"},
        {"cadence_policy": "anything"},
        {"reference_timestamps_ms": [0, -1]},
        {"reference_timestamps_ms": "bad"},
        {"relative_tolerance_percent": 101},
        {"history_size": 3.0},
    ],
)
def test_parameter_validation_does_not_allocate_sequence_history(parameters):
    block = monitor(**parameters)
    assert block.execute(sample(0, 0)) is None and block.error_kind == "invalid"
    assert not block.memory["streams"]


def test_runtime_copies_are_independent_and_context_filename_changes_reset():
    first, second = monitor(), monitor()
    assert first.execute(sample(0, 0)).passed
    second.copy_runtime_from(first)
    assert second.execute(sample(1, 40)).outputs["report"]["counts"]["frames"] == 2
    assert first.execute(sample(1, 0)).outputs["report"]["counts"]["frames"] == 2
    assert second.execute(sample(2, 80)).passed
    values = sample(0, 0)
    values["image"].payload["video_frame"]["filename"] = "new.avi"
    assert first.execute(values).outputs["report"]["counts"]["frames"] == 1


def test_xml_editor_and_toolbox_are_pure_and_preserve_all_controls(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.toolbox import block_entry

    registry, graph = builtin_registry(), Graph()
    block = graph.add(monitor(reference_timestamps_ms=[0, 40, 90]))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "timing.xml"
    export_project_xml(snapshot, path)
    restored = Graph.from_dict(read_project_xml(path), registry)
    assert restored.to_dict() == snapshot
    assert "videoio" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert all(
            isinstance(window.inspector.fields[n], QComboBox)
            for n in ("timestamp_policy", "cadence_policy", "context_change")
        )
        assert not block.memory["streams"]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
