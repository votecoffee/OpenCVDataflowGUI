"""Verify pixel-to-rig calibration with sparse IDs and independent camera closure."""

import copy
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.coded_rig_observations import LoadCodedRigObservations
from opencv_dataflow.core import Graph
from opencv_dataflow.multicamera_calibration import CalibrateMulticameraRig, ValidateMulticameraRig
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/coded_multicamera_rig"


def _asset(name):
    return json.loads((EXAMPLE / "assets" / name).read_text(encoding="utf-8"))


@pytest.fixture
def graph():
    graph = Graph.load(EXAMPLE / "coded_multicamera_rig.json", builtin_registry())
    try:
        yield graph
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_fits_disjoint_ids_and_checks_independent_camera_closure(graph, profile):
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.processing_blocks():
        assert block.last_result and block.last_result.passed, (block.id, block.error)

    train, fit, check = [graph.blocks[name].last_result for name in ("train", "fit", "check")]
    report = train.outputs["report"]
    for f in range(14):
        outer = report["views"][0 if f < 7 else 2]["views"][f]
        bridge = report["views"][1]["views"][f]
        assert len(outer["marker_ids"]) == len(bridge["marker_ids"]) == 5
        assert set(outer["marker_ids"]).isdisjoint(bridge["marker_ids"])
    fitted = fit.outputs["report"]
    assert fitted["solver"] == "partial_observations" and fitted["converged"]
    assert fitted["initialization_pairs"] == [["camera_a", "camera_b"], ["camera_b", "camera_c"]]
    assert fitted["shared_captures"] == [[7, 7, 0], [7, 14, 7], [0, 7, 7]]
    assert set(np.asarray(fitted["point_counts"]).ravel()) == {0, 20}
    assert fitted["rms_px"] < 0.75
    rig = fit.outputs["rig"]
    known = np.array(_asset("ground_truth.json")["anchor_to_camera"])
    solved = np.array([record["source_to_target"] for record in rig["anchor_to_camera"]])
    translation_error = np.linalg.norm(solved[:, :3, 3] - known[:, :3, 3], axis=1)
    assert translation_error.max() < 0.0015
    for got, expected in zip(solved, known, strict=True):
        angle = cv2.Rodrigues(got[:3, :3] @ expected[:3, :3].T)[0]
        assert np.rad2deg(np.linalg.norm(angle)) < 0.12

    held = check.outputs["report"]
    assert set(held["capture_ids"]).isdisjoint(rig["training_capture_ids"])
    assert len(held["cross_projections"]) == 12
    assert all(row["test_points"] == 28 for row in held["cross_projections"])
    assert any(
        {row["fit_camera"], row["test_camera"]} == {"camera_a", "camera_c"}
        for row in held["cross_projections"]
    )
    assert held["rms_px"] < 0.8
    assert bool(train.graphics) == bool(check.graphics) == (profile == "debug")


@pytest.mark.local_examples
def test_heldout_geometry_gate_and_sparse_initialization(graph):
    graph.run()
    fitter, checking = (graph.blocks[name] for name in ("fit", "check"))
    train, held = [graph.blocks[name].last_result.outputs for name in ("train", "holdout")]
    rig = fitter.last_result.outputs["rig"]
    invalid = copy.deepcopy(held["observations"])
    for pixels in invalid["image_points"][2]:
        if pixels is not None:
            for point in pixels:
                if point is not None:
                    point[0] += 4.0
    result = checking.execute({"rig": rig, "observations": invalid})
    assert result is not None and not result.passed
    assert "multicamera_holdout_view_rms" in {r["code"] for r in result.fail_reasons}

    fitter.parameters["initialization"] = "stereo_calibrate"
    inputs = {"observations": train["observations"], "camera_models": train["camera_models"]}
    assert fitter.execute(inputs) is None and "require Register cameras" in fitter.error
    fitter.parameters.update(initialization="register_cameras", max_iterations=1)
    limited = fitter.execute(inputs)
    assert limited is not None and not limited.passed and limited.outputs["rig"] is NO_VALUE
    assert not limited.outputs["report"]["converged"]


@pytest.mark.local_examples
def test_duplicate_ids_and_empty_captures_are_reported(tmp_path):
    record = _asset("train.json")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    for name, right in (("duplicate", 5), ("usable", 37)):
        image = np.full((160, 220), 255, np.uint8)
        for x, tag in ((30, 5), (120, right)):
            image[38:98, x : x + 60] = cv2.aruco.generateImageMarker(dictionary, tag, 60)
        ok, data = cv2.imencode(".png", image)
        assert ok
        (tmp_path / f"{name}.png").write_bytes(data.tobytes())
    record.pop("camera_models")
    record.update(
        image_sizes=[[220, 160]] * 3,
        capture_ids=["partial", "second", "absent"],
        images=[
            ["duplicate.png", "usable.png", None],
            ["usable.png", None, None],
            ["usable.png", None, None],
        ],
    )
    manifest = tmp_path / "capture.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    block = LoadCodedRigObservations(parameters={"path": str(manifest)})
    result = block.execute({})
    assert result and result.passed, block.error
    assert result.outputs["camera_models"] is NO_VALUE
    observations, report = (result.outputs[key] for key in ("observations", "report"))
    assert observations["capture_ids"] == ["partial", "second"]
    assert observations["visibility"] == [[0, 1], [1, 0], [1, 0]]
    assert report["views"][0]["views"][0]["status"] == "duplicate_markers"
    assert report["views"][0]["views"][0]["duplicate_ids"] == [5]
    assert report["dropped_captures"] == ["absent"]


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ["pinhole", "fisheye"])
def test_fits_intrinsics_from_different_point_subsets(lens):
    builder = runpy.run_path(str(EXAMPLE.with_name("multicamera_rig") / "build_example.py"))
    train, held = [
        builder["observations"](label, count, seed, lens)
        for label, count, seed in (("fit", 16, 85147), ("held", 6, 15198))
    ]
    for record in (train, held):
        for c, f in zip(*np.where(record["visibility"]), strict=True):
            for index in range(c + f % 3, len(record["object_points"]), 4):
                record["image_points"][c][f][index] = None
    fit = CalibrateMulticameraRig(parameters={"lens": lens, "max_iterations": 90})
    result = fit.execute({"observations": train})
    assert result and result.passed, fit.error
    assert result.outputs["report"]["rms_px"] < 0.05
    check = ValidateMulticameraRig()
    validation = check.execute({"rig": result.outputs["rig"], "observations": held})
    assert validation and validation.passed, check.error
    assert validation.outputs["report"]["rms_px"] < 0.14
