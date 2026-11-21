"""Qualify board recovery against independently rendered image geometry."""

import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/aruco_board_recovery"
RECOVER = "3d.recover_aruco_board"


@pytest.fixture(scope="module")
def generator():
    return run_path(str(EXAMPLE / "build_example.py"))


@pytest.fixture(scope="module")
def setup_board(generator):
    registry = builtin_registry()
    renderer = registry.create(
        "calibration.generate_fiducial_target", parameters=generator["SETTINGS"]
    )
    target = renderer.execute({}).outputs["target"]
    camera = camera_record(generator["INTRINSICS"], [], generator["SIZE"], "mm")
    return registry, target, camera


def inspect(setup_board, image, inputs=None, **settings):
    registry, target, camera = setup_board
    block = registry.create(RECOVER, parameters={"error_correction_rate": 0, **settings})
    result = block.execute({"image": image, "target": target, "camera": camera, **(inputs or {})})
    return block, result


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_board_recovery_is_calibrated_and_ordered(profile):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(EXAMPLE / "aruco_board_recovery.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(2)
        graph.run()
        seed, recover = (graph.blocks[key].last_result for key in ("seed", "recover"))
        assert seed and seed.passed and recover and recover.passed
        assert seed.outputs["count"] == 4
        assert recover.outputs["count"] == 5
        report = recover.outputs["report"]
        assert report["decoded_board_ids"] == [3, 18, 1, 22]
        assert report["recovered_ids"] == [7]
        assert report["missing_ids"] == [12]
        assert report["foreign_ids"] == [42]
        assert report["seed_rms_px"] == pytest.approx(seed.outputs["report"]["final_rms_px"])
        assert report["final_rms_px"] < 0.35
        pose = recover.outputs["pose"]
        assert (pose["source_frame"], pose["target_frame"], pose["units"]) == (
            "board",
            "camera",
            "mm",
        )
        estimated = np.asarray(pose["source_to_target"])
        delta = estimated[:3, :3] @ np.asarray(truth["rotation"]).T
        assert np.degrees(np.linalg.norm(cv2.Rodrigues(delta)[0])) < 0.2
        assert np.linalg.norm(estimated[:3, 3] - truth["translation"]) < 0.4
        matched = recover.outputs["matches"]
        assert [row["id"] for row in matched] == [7, 3, 18, 1, 22]
        assert 0 <= matched[0]["rejected_index"] < report["rejected_candidate_count"]
        assert matched[0]["origin"] == "recovered"
        assert all(row["origin"] == "detected" for row in matched[1:])
        for row in matched:
            key = str(row["id"])
            np.testing.assert_array_equal(row["object_corners"], truth["object_corners"][key])
            np.testing.assert_allclose(row["image_corners"], truth["ideal_pixels"][key], atol=1.0)
            assert max(row["reprojection_errors_px"]) < 0.6
        if profile == "debug":
            assert set(recover.drawings) == {
                "Decoded board markers",
                "Recovered board markers",
                "Ignored foreign markers",
            }
        else:
            assert not recover.drawings
    finally:
        graph.close()


def test_foreign_rejected_candidate_cannot_fill_missing_id(generator, setup_board):
    image, _, _ = generator["capture"](damaged_foreign=True)
    block, result = inspect(setup_board, image, minimum_markers=6)
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "aruco_board_insufficient_matches"
    assert result.outputs["pose"] is NO_VALUE
    report = result.outputs["report"]
    assert report["recovered_ids"] == [7]
    assert report["missing_ids"] == [12]
    assert report["foreign_ids"] == []  # The damaged foreign code was rejected on its own.


def test_ambiguous_duplicate_seeds_withhold_pose(generator, setup_board):
    image, _, _ = generator["capture"](foreign_id=3)
    block, result = inspect(setup_board, image)
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "aruco_board_duplicate_ids"
    assert result.outputs["report"]["duplicate_ids"] == [3]
    assert result.outputs["pose"] is NO_VALUE


def test_seed_minimum_guard_and_candidate_gate(generator, setup_board):
    image, _, pixels = generator["capture"]()
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, np.rint(pixels[3]).astype(np.int32), 255)
    block, result = inspect(setup_board, image, {"mask": mask})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "aruco_board_insufficient_seed"
    assert result.outputs["report"]["decoded_board_ids"] == [3]
    assert result.outputs["report"]["recovered_ids"] == []
    assert result.outputs["pose"] is NO_VALUE

    block, result = inspect(setup_board, image, max_recovered_corner_error_px=0)
    assert result and result.passed, block.error
    assert result.outputs["count"] == 4
    assert result.outputs["report"]["missing_ids"] == [7, 12]
    assert set(result.outputs["report"]["excluded_recovery_indices"]) == {7}


@pytest.mark.parametrize(
    ("kind", "changes", "message"),
    [
        ("camera", {"units": "m"}, "same metric length units"),
        ("camera", {"image_size": [640, 450]}, "image size differs"),
        ("target", {"border_bits": 0}, "Board marker border bits"),
    ],
)
def test_invalid_calibration_or_layout_is_reported(generator, setup_board, kind, changes, message):
    image, _, _ = generator["capture"]()
    source = setup_board[1 if kind == "target" else 2]
    tool, result = inspect(setup_board, image, {kind: {**source, **changes}})
    assert result is None
    assert message in tool.error
