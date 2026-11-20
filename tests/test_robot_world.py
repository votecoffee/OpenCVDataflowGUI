"""Qualify both robot-world configurations and guard held-out calibration evidence."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record, rigid
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/robot_world"
ASSETS = EXAMPLE / "assets"
TYPE_ID = "calibration.robot_world_hand_eye"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def detect_points(detector, camera, index):
    data = np.frombuffer((ASSETS / f"heldout-{index:02}.png").read_bytes(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    detection = detector.execute({"image": image, "camera": camera})
    assert detection and detection.passed, detector.error
    points = np.asarray(detection.outputs["image_points"], np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(
        image,
        points,
        (4, 4),
        (-1, -1),
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 32, 0.001),
    )
    return np.asarray(detection.outputs["object_points"], float), points


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("arrangement", "profile", "mount_frame", "station_frame", "mm", "pixels"),
    (
        ("fixed_camera", "debug", "target", "camera", 0.25, 0.18),
        ("robot_mounted_camera", "lean", "camera", "target", 1.35, 0.55),
    ),
)
def test_saved_joint_calibration_predicts_held_out_target_images(
    arrangement, profile, mount_frame, station_frame, mm, pixels
):
    truth = read_json(ASSETS / "truth.json")
    registry = builtin_registry()
    detector = registry.create(
        "calibration.charuco",
        parameters={**truth["board"], "units": "mm", "minimum_corners": 16},
    )
    graph = Graph.load(EXAMPLE / f"{arrangement}.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["fit"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed, result.fail_reasons
        report = result.outputs["report"]
        assert report["converged"]
        assert report["fit"]["sample_count"] == 12
        assert report["heldout"]["sample_count"] == 4
        assert report["heldout"]["translation_rms_mm"] < mm
        assert report["heldout"]["rotation_rms_degrees"] < 0.15
        mount_pose = result.outputs["mount_to_gripper"]
        station_pose = result.outputs["station_to_base"]
        assert (mount_pose["source_frame"], mount_pose["target_frame"]) == (mount_frame, "gripper")
        assert (station_pose["source_frame"], station_pose["target_frame"]) == (
            station_frame,
            "base",
        )
        assert mount_pose["units"] == station_pose["units"] == "mm"
        mount, station = rigid(mount_pose), rigid(station_pose)
        for role, recovered in (("mount", mount), ("station", station)):
            expected = np.asarray(truth["expected"][arrangement][role])
            assert np.linalg.norm(recovered[:3, 3] - expected[:3, 3]) < mm
            relative = recovered[:3, :3] @ expected[:3, :3].T
            assert np.degrees(Rotation.from_matrix(relative).magnitude()) < 0.15
        camera = truth["camera"]
        intrinsics = np.asarray(camera["camera_matrix"])
        observations = read_json(ASSETS / f"{arrangement}.json")
        for index, robot_pose in enumerate(observations["heldout_gripper_poses"]):
            mount_in_base = rigid(robot_pose) @ mount
            if arrangement == "fixed_camera":
                predicted = np.linalg.inv(station) @ mount_in_base
            else:
                predicted = np.linalg.inv(mount_in_base) @ station
            xyz, observed = detect_points(detector, camera, index)
            projected, _ = cv2.projectPoints(
                xyz, cv2.Rodrigues(predicted[:3, :3])[0], predicted[:3, 3], intrinsics, None
            )
            assert np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=-1))) < pixels
    finally:
        graph.close()


@pytest.mark.local_examples
def test_failed_single_holdout_keeps_the_training_report_and_withholds_transforms():
    data = read_json(ASSETS / "fixed_camera.json")
    train = {key: value for key, value in data.items() if not key.startswith("heldout_")}
    block = builtin_registry().create(TYPE_ID)
    before = block.execute(train)
    assert before and before.passed
    assert "heldout" not in before.outputs["report"]
    target = rigid(data["heldout_target_poses"][0])
    target[0, 3] += 15
    after = block.execute(
        {
            **train,
            "heldout_gripper_poses": data["heldout_gripper_poses"][:1],
            "heldout_target_poses": [pose_record(target, "target", "camera", "mm")],
        }
    )
    assert after and not after.passed
    assert after.outputs["report"]["fit"] == before.outputs["report"]["fit"]
    assert after.outputs["report"]["heldout"]["sample_count"] == 1
    assert after.outputs["report"]["heldout"]["translation_rms_mm"] > 14
    assert [reason["code"] for reason in after.fail_reasons] == [
        "robot_world_heldout_translation_rms"
    ]
    assert after.outputs["mount_to_gripper"] is after.outputs["station_to_base"] is NO_VALUE


@pytest.mark.local_examples
def test_ticketed_samples_pair_by_identity_and_meters_preserve_the_pose_contract():
    data = read_json(ASSETS / "fixed_camera.json")
    block = builtin_registry().create(TYPE_ID)
    baseline = block.execute(data)
    assert baseline and baseline.passed
    clock = TicketClock(session="robot-world")

    def group(poses, source, start, *, reverse=False):
        packets = []
        for index, pose in enumerate(poses, start):
            transform = rigid(pose)
            transform[:3, 3] *= 0.001
            converted = pose_record(transform, pose["source_frame"], pose["target_frame"], "m")
            packets.append(ValuePacket(converted, {"ticket": clock.issue(source, "fixed", index)}))
        return FlowGroup(tuple(reversed(packets) if reverse else packets), (source,) * len(packets))

    paired = {
        "gripper_poses": group(data["gripper_poses"], "robot", 0),
        "target_poses": group(data["target_poses"], "camera", 0, reverse=True),
        "heldout_gripper_poses": group(data["heldout_gripper_poses"], "robot", 20),
        "heldout_target_poses": group(data["heldout_target_poses"], "camera", 20, reverse=True),
    }
    result = block.execute(paired)
    assert result and result.passed, block.error
    for key in ("station_to_base", "mount_to_gripper"):
        assert result.outputs[key]["units"] == "m"
        transform = rigid(result.outputs[key])
        transform[:3, 3] *= 1000
        np.testing.assert_allclose(transform, rigid(baseline.outputs[key]), atol=1e-6)
    assert [v["ticket"] for v in result.outputs["report"]["heldout"]["samples"]] == list(
        range(20, 24)
    )

    paired["heldout_gripper_poses"] = group(data["heldout_gripper_poses"], "robot", 0)
    paired["heldout_target_poses"] = group(data["heldout_target_poses"], "camera", 0)
    assert block.execute(paired) is None
    assert "must be disjoint" in block.error


def test_single_axis_motion_fails_to_constrain_both_transforms():
    block = builtin_registry().create(TYPE_ID)
    poses = []
    for angle in (0.0, 0.16, 0.32, 0.54):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        poses.append(pose)
    result = block.execute(
        {
            "gripper_poses": [pose_record(pose, "gripper", "base", "mm") for pose in poses],
            "target_poses": [pose_record(pose, "target", "camera", "mm") for pose in poses],
        }
    )
    assert result is None
    assert "two independent axes" in block.error
