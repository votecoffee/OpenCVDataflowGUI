"""Native unordered cameras against independent focal, rotation and world-ray truth."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.panorama_geometry import project_points, validate_layout

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/rotating_panorama"
pytestmark = pytest.mark.local_examples


def fixture_module():
    spec = importlib.util.spec_from_file_location("rotation_fixture", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "case",
    [
        "unordered",
        "reordered",
        "disconnected",
        "allow_partial",
        "fail_partial",
        "uint16_plane",
        "masked",
        "distorted",
    ],
)
def test_native_rotating_panorama_and_source_measurements_match_known_cameras(profile, case):
    truth = json.loads((BASE / "truth.json").read_bytes())
    options = truth["cases"][case]
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["panorama"]
        result = block.last_result
        assert result is not None, block.error
        assert block.state == ("pass" if options.get("passed", True) else "fail"), block.error
        assert not any(b.state == "error" for b in graph.processing_blocks()), [
            (b.id, b.error) for b in graph.processing_blocks()
        ]
        report = result.outputs["report"]
        if case in ("disconnected", "distorted"):
            assert report["status"] != "ok" and not block.current_emissions.get("image")
            if case == "disconnected":
                assert report["dropped_labels"] == ["unrelated"]
            if profile == "debug":
                assert graph.blocks["diagnostic_filmstrip"].last_result is not None
            return
        layout = validate_layout(result.outputs["layout"])
        cameras = {c["label"]: c for c in layout["cameras"]}
        assert set(cameras) == {"left", "right", "center"}
        if "partial" in case:
            assert not report["complete"] and report["dropped_labels"] == ["unrelated"]
        else:
            assert report["complete"] and report["dropped_labels"] == []
        accept = truth["acceptance"]
        for label, camera in cameras.items():
            K = np.asarray(camera["K"])
            assert (
                max(
                    abs(K[0, 0] / truth["focal_pixels"] - 1),
                    abs(K[1, 1] / truth["focal_pixels"] - 1),
                )
                < accept["focal_relative_error"]
            )
            np.testing.assert_allclose(K[:2, 2], np.array(truth["source_size"]) / 2, atol=0.01)
            rotation = np.array(camera["camera_to_anchor"])
            expected = fixture_module().yaw_matrix(truth["yaw_degrees"][label])
            angle = np.degrees(np.arccos(np.clip((np.trace(rotation.T @ expected) - 1) / 2, -1, 1)))
            assert angle < accept["relative_rotation_degrees"]
            landmarks = truth["landmarks"][label]
            points, valid = project_points(
                landmarks["pixels"], camera, layout["projection"], layout["scale"]
            )
            world = np.asarray(landmarks["world_rays"])
            x, y, z = world.T
            if layout["projection"] == "plane":
                reference = np.c_[280 * x / z, 280 * y / z]
            else:
                reference = np.c_[
                    280 * np.arctan2(x, z), 280 * (np.pi / 2 + np.arctan2(y, np.hypot(x, z)))
                ]
            assert valid.all()
            rms = np.sqrt(np.mean(np.sum((points - reference) ** 2, axis=1)))
            assert rms < accept["landmark_rms_pixels"], (label, rms)
        pairs = {tuple(sorted((r["source_index"], r["target_index"]))): r for r in report["pairs"]}
        if case == "unordered":
            assert pairs[(0, 1)]["status"] != "validated"
            assert pairs[(0, 2)]["status"] == pairs[(1, 2)]["status"] == "validated"
        supported = result.outputs["coverage"] != 0
        np.testing.assert_array_equal(result.outputs["contributors"] > 0, supported)
        assert not result.outputs["image"][~supported].any()
        assert report["overlap_pixels"] == int(
            np.count_nonzero(result.outputs["contributors"] >= 2)
        )
        assert result.outputs["image"].dtype == (np.uint16 if case == "uint16_plane" else np.uint8)
        if case == "uint16_plane":
            assert result.outputs["image"].max() > 255
        mapped = graph.blocks["points"].last_result
        assert mapped is not None, graph.blocks["points"].error
        assert 1 <= mapped.outputs["count"] <= 32
        assert mapped.payloads["points"]["spatial"] == layout["spatial"]
        assert mapped.payloads["points"]["ticket"] == result.payloads["image"]["ticket"]
        assert graph.blocks["filmstrip"].last_result is not None
        assert bool(result.drawings) is (profile == "debug")
        if profile == "debug":
            assert "Overlaid 2 layers" in graph.blocks["overlay"].last_result.message
    finally:
        graph.close()


def test_original_camera_assets_projects_and_independent_truth_regenerate_exactly(tmp_path):
    module = fixture_module()
    truth = json.loads((BASE / "truth.json").read_bytes())
    names = (
        ["truth.json"]
        + [f"{case}.json" for case in module.CASES]
        + ["assets/" + name for name in truth["asset_sha256"]]
    )
    before = {name: (BASE / name).read_bytes() for name in names}
    module.HERE = tmp_path
    module.build()
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
    assert {name: (BASE / name).read_bytes() for name in names} == before
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256(before["assets/" + name]).hexdigest() == digest
