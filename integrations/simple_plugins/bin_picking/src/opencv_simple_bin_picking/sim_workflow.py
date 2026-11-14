"""App-owned one-cell pick/place workflow, independent of ROS and graph scheduling."""

import time

import numpy as np

from .parallel_jaw import propose_box_tops
from .sim_client import unpack
from .specs import Camera


def associate(candidate, truth, excluded=()):
    """Associate measured pose to collision ID; never substitute truth into detector."""
    rows = []
    for obj in truth["objects"]:
        if obj["object_id"] in excluded or obj["world_pose"] is None:
            continue
        if obj["receive_age_s"] is None or obj["receive_age_s"] > 0.5:
            raise ValueError("Fresh collision object observations required")
        distance = np.linalg.norm(
            np.asarray(obj["world_pose"]["position_m"]) - candidate["estimated_center_m"]
        )
        if distance <= 0.008:
            rows.append(obj)
    if len(rows) != 1:
        raise ValueError(
            "Observed candidate has no unique collision-object association within 8 mm"
        )
    return rows[0]


def pick_one(client, task, settings, emit=lambda *_: None, excluded=()):
    receipts = []
    report = dict(
        schema="bin_grid_pick_v1",
        run_id=task["run_id"],
        cell=task["cell"],
        grid_xyz_m=task["xyz_m"],
        status="failed",
        simulation_only=True,
        assistance="pose_gated_fixed_joint",
        friction_grasp_validated=False,
        perception="first_return_depth",
        collision_geometry="privileged_simulator",
        receipts=receipts,
    )
    captures = {}
    phase = "initializing"

    def call(operation, **parameters):
        nonlocal phase
        phase = operation
        emit(operation, parameters)
        started = time.monotonic()
        result = client.call(operation, **parameters)
        receipts.append(
            dict(
                operation=operation,
                parameters=parameters,
                result=result,
                wall_s=time.monotonic() - started,
            )
        )
        return result

    def move(target, cartesian=False):
        plan = call(
            "plan_cartesian" if cartesian else "plan_pose",
            world_from_tcp=target.tolist(),
            velocity_scale=0.3,
            acceleration_scale=0.3,
        )
        return call("execute_simulation", plan_id=plan["plan_id"])

    try:
        call("gripper", opening_m=0.056)
        raw = call("capture")
        # Arrays are retained separately, not duplicated into each JSON receipt.
        receipts[-1]["result"] = {k: v for k, v in raw.items() if k not in {"rgb_bgr", "depth_m"}}
        depth, rgb = unpack(raw["depth_m"]), unpack(raw["rgb_bgr"])
        camera = Camera(
            tuple(np.asarray(raw["camera_k"]).reshape(3, 3)),
            (depth.shape[1], depth.shape[0]),
            raw["frame"],
        )
        detections, support = propose_box_tops(depth, camera, raw["world_from_camera"], settings)
        captures["before"] = dict(
            depth=depth,
            rgb=rgb,
            camera=camera.record("m"),
            world_from_camera=raw["world_from_camera"],
            support=support,
            capture_id=raw["capture_id"],
            stamp_ns=raw["stamp_ns"],
        )
        report["detections"] = detections
        if not detections["candidates"]:
            raise ValueError("No supported visible box top in the bin")
        candidate = detections["candidates"][0]
        call("sync_scene")
        truth = call("evaluation")
        obj = associate(candidate, truth, excluded)
        name = obj["object_id"]
        report["object_id"] = name
        report["candidate"] = candidate
        finger_pairs = [
            [name, link] for link in ("gripper_left_finger_link", "gripper_right_finger_link")
        ]
        support_pairs = []
        xyz = np.asarray(obj["world_pose"]["position_m"])
        for other in truth["objects"]:
            if other["object_id"] == name or other["world_pose"] is None:
                continue
            delta = xyz - other["world_pose"]["position_m"]
            if np.linalg.norm(delta[:2]) < 0.01 and 0.03 < delta[2] < 0.05:
                support_pairs.append([name, other["object_id"]])
        # Resting support tolerance is limited to the picked object's floor/support.
        if xyz[2] < settings.floor_z + settings.part_height:
            support_pairs += [[name, "bin_floor"], [name, "table_top"]]
        call("touch", pairs=finger_pairs, allowed=True)
        call("gripper", opening_m=candidate["jaw_opening_m"])
        pre = np.array(candidate["world_from_pregrasp"])
        grasp = np.array(candidate["world_from_tcp"])
        move(pre)
        move(grasp, True)
        call("gripper", opening_m=candidate["jaw_grasp_m"])
        call("attach", object_id=name)
        call("touch", pairs=support_pairs, allowed=True)
        move(pre, True)
        call("touch", pairs=support_pairs, allowed=False)
        lifted = call("evaluation")
        lifted_obj = next(o for o in lifted["objects"] if o["object_id"] == name)
        if lifted_obj["world_pose"]["position_m"][2] - xyz[2] < settings.approach_height - 0.015:
            raise RuntimeError("Object did not follow the tool during the observed assisted lift")
        report["observed_lift_m"] = lifted_obj["world_pose"]["position_m"][2] - xyz[2]
        place = np.eye(4)
        place[:3, :3] = np.diag([1, -1, -1])
        place[:3, 3] = task["xyz_m"]
        place[2, 3] += settings.part_height / 2 - settings.tcp_below_top + 0.003
        above = place.copy()
        above[2, 3] += settings.approach_height
        move(above)
        move(place, True)
        call("gripper", opening_m=candidate["jaw_opening_m"])
        call("detach", object_id=name)
        move(above, True)
        call("touch", pairs=finger_pairs, allowed=False)
        # Wait for independent simulator feedback; no set-pose operation exists.
        settled = []
        settled_stamps = []
        stable = False
        until = time.monotonic() + 4
        while time.monotonic() < until:
            evaluation = call("evaluation")
            placed = next(o for o in evaluation["objects"] if o["object_id"] == name)
            actual = np.array(placed["world_pose"]["position_m"])
            if placed["receive_age_s"] is None or placed["receive_age_s"] > 0.5:
                raise RuntimeError("Placement feedback is stale")
            stamp = placed.get("source_stamp_ns", len(settled_stamps))
            if settled_stamps and stamp <= settled_stamps[-1]:
                time.sleep(0.15)
                continue
            settled.append(actual)
            settled_stamps.append(stamp)
            if (
                len(settled) >= 3
                and np.max(np.linalg.norm(np.array(settled[-3:]) - actual, axis=1)) < 0.0005
            ):
                stable = True
                break
            time.sleep(0.15)
        error = float(np.linalg.norm(actual - task["xyz_m"]))
        report.update(
            actual_xyz_m=actual.tolist(),
            placement_error_m=error,
            observed_detached=placed["attachment"] == "detached",
            settled=stable,
        )
        qx, qy, qz, qw = placed["world_pose"]["quaternion_xyzw"]
        up_z = 1 - 2 * (qx * qx + qy * qy)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        yaw_error = abs((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        report.update(upright_cosine=float(up_z), yaw_error_rad=float(yaw_error))
        if (
            error > 0.01
            or not report["observed_detached"]
            or not stable
            or up_z < 0.98
            or yaw_error > np.deg2rad(5)
        ):
            raise RuntimeError("Placed object did not settle within 10 mm of its grid cell")
        if "park_world_from_tcp" in task:
            move(np.asarray(task["park_world_from_tcp"]))
        raw = call("capture")
        receipts[-1]["result"] = {k: v for k, v in raw.items() if k not in {"rgb_bgr", "depth_m"}}
        captures["after"] = dict(
            depth=unpack(raw["depth_m"]),
            rgb=unpack(raw["rgb_bgr"]),
            camera=camera.record("m"),
            world_from_camera=raw["world_from_camera"],
            capture_id=raw["capture_id"],
            stamp_ns=raw["stamp_ns"],
        )
        report["status"] = "placed"
    except Exception as error:
        report.update(status="held", phase=phase, reason=str(error), automatic_retry=False)
    return report, captures
