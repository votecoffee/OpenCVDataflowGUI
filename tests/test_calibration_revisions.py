"""Qualify revision file identity, capture provenance, and derived stereo/lens grids."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import APPLIED, CAPTURE, REVISION
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/calibration_revisions"
STORE = BASE / "assets/revisions"


def file_block(registry, direction, profile, revision, path=STORE):
    return registry.create(
        f"{direction}.calibration",
        parameters={"path": str(path), "profile_id": profile, "revision": revision},
    )


def test_revision_selects_exact_version_preserves_store_and_checks_checksum(tmp_path):
    registry = builtin_registry()
    writer = file_block(registry, "output", "station", "r1", tmp_path)
    params = {"width": 32, "height": 24, "frame": "fixture_camera"}
    profile = registry.create("3d.camera", parameters=params).execute({}).outputs["camera"]
    assert writer.execute({"profile": profile}).passed
    target = tmp_path / "station/r1.json"
    before = target.read_bytes()
    assert writer.execute({"profile": profile}) is None
    assert target.read_bytes() == before

    writer.parameters["revision"] = "r2"
    profile["camera_matrix"][0][0] = 525.0
    assert writer.execute({"profile": profile}).passed
    reader = file_block(registry, "input", "station", "r1", tmp_path)
    v1 = reader.execute({}).outputs["profile"]
    reader.parameters["revision"] = "r2"
    v2 = reader.execute({}).outputs["profile"]
    assert v1["camera_matrix"][0][0] == 500.0
    assert v2["camera_matrix"][0][0] == 525.0
    assert v1[REVISION]["sha256"] != v2[REVISION]["sha256"]

    reader.parameters["revision"] = "r3"
    assert reader.execute({}) is None
    assert not reader.current_emissions
    reader.parameters["revision"] = "../r2"
    assert reader.execute({}) is None

    reader.parameters["revision"] = "r1"
    saved = json.loads(before)
    saved["profile"]["camera_matrix"][1][1] += 1
    target.write_text(json.dumps(saved), encoding="utf-8")
    assert reader.execute({}) is None and "checksum" in reader.error
    target.write_bytes((tmp_path / "station/r2.json").read_bytes())
    assert reader.execute({}) is None and "selected profile and revision" in reader.error


def test_capture_ticket_camera_grid_and_profile_tampering_are_rejected():
    registry = builtin_registry()
    model = file_block(registry, "input", "left-camera", "r2").execute({}).outputs["profile"]
    rectify = registry.create("3d.rectify_camera")
    w, h = model["image_size"]
    pix = np.zeros((h, w), np.uint8)
    clock = TicketClock()
    ticket = clock.issue("source")
    raw = ValuePacket(pix, {"spatial": frame([w, h], "raw"), "ticket": ticket})
    assert rectify.execute({"image": raw, "camera": model}) is None
    assert "matching capture metadata" in rectify.error

    declare = registry.create(
        "calibration.declare_capture",
        parameters={
            "profile_id": "left-camera",
            "revision": "r2",
            "camera_frame": "left_camera",
            "sha256": model[REVISION]["sha256"],
        },
    )
    result = declare.execute({"image": raw})
    capture = packet_for(result, "image")
    accepted = rectify.execute({"image": capture, "camera": model})
    assert accepted and accepted.passed, rectify.error
    props = accepted.payloads["image"]
    assert CAPTURE not in props and props[APPLIED] == model[REVISION]
    assert props["ticket"] == ticket

    for field, change, reason in (
        ("camera_frame", "right_camera", "camera frame"),
        ("revision", "r1", "revision"),
        ("sha256", "0" * 64, "revision"),
    ):
        bad = copy.deepcopy(capture.payload)
        bad[CAPTURE][field] = change
        assert rectify.execute({"image": ValuePacket(pix, bad), "camera": model}) is None
        assert reason in rectify.error
    bad = copy.deepcopy(capture.payload)
    bad["spatial"]["to_root"][0][2] = 2
    assert rectify.execute({"image": ValuePacket(pix, bad), "camera": model}) is None
    assert "original full" in rectify.error
    bad = copy.deepcopy(capture.payload)
    bad["spatial"]["root_id"] = "different-native-grid"
    assert rectify.execute({"image": ValuePacket(pix, bad), "camera": model}) is None
    assert "different pixel grid" in rectify.error
    bad = copy.deepcopy(capture.payload)
    bad["ticket"] = clock.issue("source")
    assert rectify.execute({"image": ValuePacket(pix, bad), "camera": model}) is None
    assert "different ticket" in rectify.error

    declare.parameters["revision"] = "r1"
    assert declare.execute({"image": capture}) is None
    assert "cannot be changed" in declare.error
    forged = copy.deepcopy(model)
    forged["camera_matrix"][0][0] += 12
    assert rectify.execute({"image": capture, "camera": forged}) is None
    assert "changed after its revision" in rectify.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_stereo_and_map_projects_enforce_revision(profile):
    registry = builtin_registry()
    for name in ("stereo_r2", "stereo_mismatch", "camera_maps"):
        graph = Graph.load(BASE / f"{name}.json", registry)
        try:
            assert validate_graph(graph)["valid"]
            graph.set_run_profile(profile)
            graph.run()
            if name == "stereo_mismatch":
                assert graph.blocks["rectify"].state == "error"
                assert "camera frame or calibration revision" in graph.blocks["rectify"].error
                assert not graph.blocks["cloud"].current_emissions
                continue
            assert all(b.state == "pass" for b in graph.processing_blocks())
            if name == "stereo_r2":
                rect = graph.blocks["rectify"].last_result
                cloud = graph.blocks["cloud"].last_result.outputs["cloud"]
                assert len(cloud.points) > 1000
                assert np.nanmedian(cloud.points[:, 2]) == pytest.approx(3.0, abs=0.03)
                assert rect.outputs["rectification"][REVISION]["revision"] == "r2"
            else:
                result = graph.blocks["corrected"].last_result
                assert result.outputs["image"].shape == (135, 240)
                assert result.payloads["image"][APPLIED]["revision"] == "r2"
        finally:
            graph.close()


def test_versioned_stereo_rejects_mutated_extrinsic_or_q():
    registry = builtin_registry()
    graph = Graph.load(BASE / "stereo_r2.json", registry)
    try:
        graph.run()
        rig = copy.deepcopy(graph.blocks["profile"].last_result.outputs["profile"])
        rig["transform"]["source_to_target"][0][3] = -0.24
        left, right = (
            packet_for(graph.blocks[side].last_result, "image") for side in ("left", "right")
        )
        rectify = registry.create("3d.stereo_rectify")
        assert rectify.execute({"left": left, "right": right, "rig": rig}) is None
        assert "Stereo calibration changed" in rectify.error
        rec = copy.deepcopy(graph.blocks["rectify"].last_result.outputs["rectification"])
        rec["Q"][3][2] *= 2
        disparity = packet_for(graph.blocks["disparity"].last_result, "disparity")
        reproject = registry.create("3d.reproject")
        assert reproject.execute({"disparity": disparity, "rectification": rec}) is None
        assert "differs from sealed stereo Q" in reproject.error
    finally:
        graph.close()


def test_npz_revision_rejects_other_capture_and_metadata_damage(tmp_path):
    registry = builtin_registry()
    default = BASE / "assets/left-r2.npz"
    loader = registry.create("input.lens_maps", parameters={"path": str(default)})
    maps = loader.execute({})
    assert maps and maps.passed, loader.error
    capture = registry.create("input.stereo_demo").execute({}).outputs["left"]
    declare = registry.create(
        "calibration.declare_capture",
        parameters={"profile_id": "left-camera", "revision": "r1", "camera_frame": "left_camera"},
    )
    raw = packet_for(declare.execute({"image": capture}), "image")
    remap = registry.create("cv.remap")
    assert (
        remap.execute({"image": raw, **{p: packet_for(maps, p) for p in ("map_x", "map_y")}})
        is None
    )
    assert "camera frame or calibration revision" in remap.error

    with np.load(default) as data:
        manifest = json.loads(data["metadata"].tobytes())
        payloads = {key: data[key] for key in ("map_x", "map_y")}
    manifest["info"][REVISION]["revision"] = "r1"
    damaged = tmp_path / "changed.npz"
    np.savez(damaged, metadata=np.frombuffer(json.dumps(manifest).encode(), np.uint8), **payloads)
    loader.parameters["path"] = str(damaged)
    assert loader.execute({}) is None and "metadata checksum" in loader.error
