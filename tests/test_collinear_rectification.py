"""Verify common rows, signed metric Q, mask support and baseline provenance."""

import copy
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.stereo3d import STEREO_GRID

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/collinear_stereo"
TYPE = "3d.rectify_collinear"


@pytest.fixture(scope="module")
def example():
    return run_path(str(EXAMPLE / "build_example.py"))


def _inputs(example, centers=None, units="m"):
    if centers is None:
        centers = example["CENTERS"]
    rotations = [example["yaw"](angle) for angle in example["YAWS"]]
    images, optics = {}, []
    size = example["SIZE"]
    for name, (fx, fy, cx, cy) in zip(example["NAMES"], example["INTRINSICS"], strict=True):
        optics.append(
            camera_record(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                example["DISTORTION"],
                size,
                units,
                name,
            )
        )
        images[name] = np.full(size[::-1], 137, np.uint16)
    poses = []
    for center, rotation, name in zip(centers, rotations, example["NAMES"], strict=True):
        transform = np.eye(4)
        transform[:3, :3] = rotation @ rotations[0].T
        transform[:3, 3] = (-rotation @ center) * (1000 if units == "mm" else 1)
        poses.append(pose_record(transform, "reference", name, units))
    return images, {
        "model": "multicamera_rig_v1",
        "cameras": optics,
        "anchor_to_camera": poses,
        "lens": "pinhole",
        "anchor_frame": "reference",
        "units": units,
        "training_capture_ids": ["fit-a", "fit-b"],
    }


@pytest.mark.parametrize("vertical,units", [(False, "m"), (True, "mm")])
def test_known_geometry_and_signed_disparity_on_both_baselines(example, vertical, units):
    # Keep the cameras on opposite sides of the reference to exercise negative disparity.
    centers = (
        ((0, 0, 0), (0, 0.056, 0), (0, -0.126, 0))
        if vertical
        else ((0, 0, 0), (0.056, 0, 0), (-0.126, 0, 0))
    )
    images, rig = _inputs(example, centers, units)
    check = example["check_observations"](centers=centers, units=units)
    block = builtin_registry().create(
        TYPE, parameters={"width": 224, "height": 144, "alpha": 1.0, "interpolation": "cubic"}
    )
    result = block.execute({**images, "rig": rig, "check": check})
    assert result and result.passed, block.error
    out = result.outputs
    report = out["report"]
    assert report["rotated_vertical_pair"] is vertical
    assert report["image_size"] == [224, 144]
    assert report["third_center_offset_mm"] < 1e-8
    assert all(v["row_rms_px"] < 1e-5 for v in report["check"].values())
    assert out["reference_12"].dtype == np.uint16
    np.testing.assert_array_equal(out["reference_12"], out["reference_13"])
    assert [float(np.mean(out[f"{name}_valid"] > 0)) for name in example["NAMES"]] == (
        report["valid_fractions"]
    )
    for suffix, other in (("12", "second"), ("13", "third")):
        left_stamp = result.payloads[f"reference_{suffix}"][STEREO_GRID]
        right_stamp = result.payloads[other][STEREO_GRID]
        assert (left_stamp["side"], right_stamp["side"]) == ("left", "right")
        assert left_stamp["sha256"] == right_stamp["sha256"]
        assert result.payloads[other]["optical_frame"] == f"{other}:rectified"
    assert (
        result.payloads["reference_12"][STEREO_GRID]
        != (result.payloads["reference_13"][STEREO_GRID])
    )

    world = np.array([0.06, -0.04, 1.3])
    disparity_sign = []
    for suffix, index in (("12", 1), ("13", 2)):
        rec = out[f"rectification_{suffix}"]
        k = np.asarray(rec["P1"])[:, :3]
        pixels = []
        for rotation_key, camera in (("R1", 0), ("R2", index)):
            vector = example["yaw"](example["YAWS"][camera]) @ (world - centers[camera])
            rectified = np.asarray(rec[rotation_key]) @ vector
            pixels.append((k @ rectified)[:2] / rectified[2])
        assert pixels[0][1] == pytest.approx(pixels[1][1], abs=1e-9)
        d = pixels[0][0] - pixels[1][0]
        disparity_sign.append(np.sign(d))
        actual = np.asarray(rec["Q"]) @ np.array([*pixels[0], d, 1])
        expected = np.asarray(rec["R1"]) @ example["yaw"](example["YAWS"][0]) @ world
        np.testing.assert_allclose(
            actual[:3] / actual[3], expected * (1000 if units == "mm" else 1), atol=1e-8
        )
    assert disparity_sign[0] == -disparity_sign[1]


@pytest.mark.parametrize(
    "offset,code", [(0.004, "collinear_center_offset"), (0.0, "collinear_baseline")]
)
def test_misaligned_or_repeated_camera_publishes_no_geometry(example, offset, code):
    third_x = 0.117 if offset else 0.054
    images, rig = _inputs(example, ((0, 0, 0), (0.054, 0, 0), (third_x, offset, 0)))
    block = builtin_registry().create(TYPE)
    result = block.execute({**images, "rig": rig})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == code
    assert all(value is NO_VALUE for name, value in result.outputs.items() if name != "report")


def test_independent_points_detect_rotational_error_and_training_reuse(example):
    images, rig = _inputs(example)
    check = example["check_observations"]()
    drift = copy.deepcopy(check)
    for view in drift["image_points"][2]:
        for pt in view:
            pt[1] += 2
    block = builtin_registry().create(TYPE)
    result = block.execute({**images, "rig": rig, "check": drift})
    assert result and not result.passed and result.outputs["rectification_13"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "collinear_row_error"
    assert result.outputs["report"]["check"]["1-3"]["row_rms_px"] > 1.5
    check["capture_ids"][1] = "fit-b"
    assert block.execute({**images, "rig": rig, "check": check}) is None
    assert "separate from rig training IDs" in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_matchers_use_distinct_baselines_and_independent_depth(profile):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / "collinear_stereo.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.run()
        for suffix, expected_coverage, max_rmse in (("12", 0.82, 27), ("13", 0.77, 17)):
            recorded = graph.blocks[f"check_{suffix}"].last_result
            assert recorded and recorded.passed, graph.blocks[f"check_{suffix}"].error
            report = recorded.outputs["report"]
            assert report["coverage"] > expected_coverage
            assert report["rmse_mm"] < max_rmse
        rect = graph.blocks["rectify"].last_result
        assert rect.outputs["report"]["check"]["1-3"]["row_rms_px"] < 0.00001
        match = graph.blocks["match_12"].last_result
        disparity = ValuePacket(match.outputs["disparity"], match.payloads["disparity"])
        output = registry.create("3d.reproject")
        assert (
            output.execute(
                {"disparity": disparity, "rectification": rect.outputs["rectification_13"]}
            )
            is None
        )
        assert "differs from sealed stereo Q" in output.error
    finally:
        graph.close()
