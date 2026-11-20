"""Qualification of timestamped RGB-D replay and confidence-preserving registration."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import rgbd_registration
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.rgbd_recording import RecordedRGBD, Sample, _pair
from opencv_dataflow.rgbd_registration import RegisterRGBD
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/recorded_rgbd/recorded_rgbd.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_recording_runs_to_exhaustion_with_independent_grids_and_common_ticket(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.filmstrip_enabled = True
        assert graph.has_finite_sources()
        tickets = []
        for i, (skew, name, count) in enumerate(zip((2, 3, 1), (0, 2, 3), (1334, 1348, 1348))):
            assert not graph.sources_exhausted()
            graph.run()
            replay, registration, cloud = (graph.blocks[k] for k in ("replay", "register", "cloud"))
            assert all(block.state == "pass" for block in (replay, registration, cloud))
            source, result = replay.last_result, registration.last_result
            info = source.outputs["info"]
            assert (info["paired_frames"], info["unmatched_depth"], info["unmatched_color"]) == (
                3,
                1,
                1,
            )
            assert info["pair_index"] == i and info["skew_ns"] == skew * 1_000_000
            assert (info["color_path"], info["depth_path"]) == (
                f"color_{name}.png",
                f"depth_{name}.png",
            )
            assert source.outputs["depth"].shape == (32, 48)
            assert result.outputs["depth"].shape == (48, 80)
            assert (
                result.outputs["report"]["registered_pixels"]
                == count
                == cloud.last_result.outputs["count"]
            )
            assert result.outputs["report"]["confidence_rejected_pixels"] == 72
            mask = result.outputs["valid_mask"] != 0
            assert np.isnan(result.outputs["depth"][~mask]).all()
            np.testing.assert_allclose(result.outputs["confidence"][mask], 220 / 255)
            np.testing.assert_array_equal(result.outputs["color"], source.outputs["image"])
            for output in ("depth", "confidence", "valid_mask"):
                assert result.payloads[output]["ticket"] == source.payloads["image"]["ticket"]
                assert result.payloads[output]["spatial"] == source.payloads["image"]["spatial"]
            assert (
                result.payloads["depth"]["recorded_depth_sample"]["timestamp_ns"]
                == info["depth_timestamp_ns"]
            )
            tickets.append(source.payloads["image"]["ticket"]["sequence"])
        assert graph.sources_exhausted()
        assert len(set(tickets)) == 3
        frames = graph.filmstrip.snapshot()
        assert [row["filename"] for row in frames] == ["color_0.png", "color_2.png", "color_3.png"]
        assert all(row["thumbnail"] for row in frames)
        graph.blocks["replay"].configure_replay("loop", rewind=True)
        for index in (0, 1, 2, 0):
            graph.run()
            assert graph.blocks["replay"].last_result.outputs["info"]["pair_index"] == index
        assert not graph.sources_exhausted()
    finally:
        graph.close()


def test_pairing_inclusive_ns_bound_and_fifo_precedence_without_reuse():
    origin = 1_900_000_000_000_000_000
    color = [Sample(origin + n, "a") for n in (0, 10, 12, 19)]
    depth = [Sample(origin + n, "b") for n in (9, 13, 20, 200)]
    assert [
        (a.timestamp_ns - origin, b.timestamp_ns - origin) for a, b in _pair(color, depth, 3)
    ] == [
        (10, 9),
        (12, 13),
        (19, 20),
    ]
    assert not _pair(color, depth, 0)
    assert _pair([color[1], color[2]], [depth[1]], 3) == ((color[1], depth[1]),)


def _recording(directory):
    for name, image in (
        ("color.png", np.zeros((2, 3, 3), np.uint16)),
        ("depth.png", np.array([[0, 80]], np.uint16)),
    ):
        assert cv2.imwrite(str(directory / name), image)
    return {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "ns",
        "color_frame": "rgb",
        "depth_frame": "ir",
        "depth_m_per_sample": 0.01,
        "color": [{"path": "color.png", "timestamp_ns": 20}],
        "depth": [{"path": "depth.png", "timestamp_ns": 20}],
    }


def _write_manifest(directory, manifest):
    path = directory / "recording.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_exact_replay_without_confidence_handles_finish_restore_and_save_as(tmp_path):
    path = _write_manifest(tmp_path, _recording(tmp_path))
    graph = Graph()
    source = graph.add(RecordedRGBD(parameters={"path": path.name, "maximum_skew_ms": 0}))
    try:
        graph.save(tmp_path / "original.json")
        graph.run()
        result = source.last_result
        assert result and result.passed, source.error
        assert result.outputs["image"].dtype == np.uint16
        np.testing.assert_allclose(
            result.outputs["depth"], [[np.nan, 0.8]], atol=1e-6, equal_nan=True
        )
        assert result.outputs["confidence"] is NO_VALUE
        assert not source.current_emissions.get("confidence")
        assert graph.sources_exhausted()
        source.reset_runtime()
        destination = tmp_path / "moved"
        destination.mkdir()
        graph.save(destination / "relocated.json")
        reopened = Graph.load(destination / "relocated.json", builtin_registry())
        try:
            reopened.run()
            assert reopened.blocks[source.id].last_result.outputs["info"]["pair_index"] == 0
        finally:
            reopened.close()
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("time_domain", "Name the recording clock"),
        ("out_of_order", "strictly increasing"),
        ("over_skew", "meet the time tolerance"),
        ("outside_path", "inside the recording folder"),
        ("wrong_depth_encoding", "Invalid depth encoding"),
        ("confidence_size", "confidence must match"),
    ],
)
def test_malformed_recording_halts_without_image_emission(tmp_path, fault, message):
    manifest = _recording(tmp_path)
    if fault == "time_domain":
        manifest["clock"] = ""
    elif fault == "out_of_order":
        manifest["depth"].append(manifest["depth"][0])
    elif fault == "over_skew":
        manifest["depth"][0]["timestamp_ns"] = 5_000_021
    elif fault == "outside_path":
        manifest["depth"][0]["path"] = "../depth.png"
    elif fault == "wrong_depth_encoding":
        manifest["depth"][0]["path"] = "color.png"
    else:
        assert cv2.imwrite(str(tmp_path / "wrong.png"), np.zeros((2, 3), np.uint8))
        manifest["depth"][0]["confidence"] = "wrong.png"
    block = RecordedRGBD(parameters={"path": str(_write_manifest(tmp_path, manifest))})
    assert block.execute({}) is None
    assert message in block.error and block.state == "error" and block.exhausted
    assert not block.current_emissions.get("image")
    assert not block.current_emissions.get("depth")


def _rig():
    # Depth pixels x=0,1 map into target cell 0 and x=2,3 into cell 1.
    # A positive 100 mm Z offset changes the target depths.
    return {
        "depth": DepthArray(np.array([[2.0, 1.0, 2.0, 2.0]], np.float32), "m"),
        "color": np.zeros((1, 2, 3), np.uint8),
        "confidence": np.array([[0.9, 0.2, 0.6, 0.8]], np.float32),
        "depth_camera": camera_record(np.eye(3), [], (4, 1), frame="depth_optical"),
        "color_camera": camera_record(np.diag([0.4, 1, 1]), [], (2, 1), frame="color_optical"),
        "transform": pose_record(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 100], [0, 0, 0, 1]]),
            "depth_optical",
            "color_optical",
            "mm",
        ),
    }


def test_cross_band_nearest_surface_controls_confidence_and_earliest_exact_tie(monkeypatch):
    monkeypatch.setattr(rgbd_registration, "_BAND_SAMPLES", 1)
    block = RegisterRGBD()
    pair = _rig()
    result = block.execute(pair)
    assert result and result.passed, block.error
    np.testing.assert_allclose(result.outputs["depth"], [[1.1, 2.1]], atol=1e-6)
    np.testing.assert_allclose(result.outputs["confidence"], [[0.2, 0.6]], atol=1e-6)
    assert result.outputs["report"]["occluded_samples"] == 2
    block.parameters["minimum_confidence"] = 0.5
    result = block.execute(pair)
    assert result and result.passed, block.error
    np.testing.assert_allclose(result.outputs["depth"], [[np.nan, 2.1]], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(result.outputs["confidence"], [[0, 0.6]], atol=1e-6)
    assert result.outputs["report"]["confidence_rejected_pixels"] == 1
    block.parameters["minimum_confidence"] = 0.8
    result = block.execute(pair)
    assert result and not result.passed
    assert [reason["code"] for reason in result.fail_reasons] == ["rgbd_no_registered_depth"]


@pytest.mark.parametrize("fault", ["ticket", "confidence_grid", "optical_frame", "pose"])
def test_registration_rejects_inconsistent_capture_contract(fault):
    rig = _rig()
    clock = TicketClock()
    if fault == "ticket":
        for name in ("color", "depth"):
            rig[name] = ValuePacket(rig[name], {"ticket": clock.issue(name)})
    elif fault == "confidence_grid":
        for name in ("depth", "confidence"):
            rig[name] = ValuePacket(rig[name], {"spatial": frame((4, 1), name)})
    elif fault == "optical_frame":
        rig["color"] = ValuePacket(rig["color"], {"optical_frame": "different-camera"})
    else:
        rig["transform"]["source_frame"] = "different-depth-camera"
    block = RegisterRGBD()
    assert block.execute(rig) is None and block.state == "error"
    assert {
        "ticket": "same ticket",
        "confidence_grid": "different depth grid",
        "optical_frame": "does not match the camera model",
        "pose": "map the depth optical frame",
    }[fault] in block.error
