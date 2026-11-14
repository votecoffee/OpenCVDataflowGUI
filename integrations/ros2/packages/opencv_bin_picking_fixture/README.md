# OpenCV bin-picking fixture

Additive, simulation-only UR5e scene and source-camera support for the OpenCVDataflowGUI bin-picking example. The application owns candidate perception, pick sequencing, the placement grid, MoveIt collision-scene transitions and trajectory execution. This package supplies no pick loop and sends no robot trajectories.

Status: native headless scene/camera readiness is verified. Assisted lift/release, reset and complete multi-pick acceptance must be measured by the application owner; do not treat pure state-machine or camera tests as those results.

## Scene and ownership

The configurable initial example contains six original 40 mm, 50 g boxes in three two-high stacks, a bin and a table with a 0.20 m work surface. A fixed overhead RGB-D camera uses the simulator's original RGB, depth and CameraInfo. Smaller/lower part surfaces may still be visible because of perspective; actual first-return occlusion must be checked in rendered depth. No object pose is substituted for source-camera data.

`fixture_prepare` makes a **new private overlay outside the installed runtime**. It copies/checksums only the needed native launch/config files, redirects their generated output and appends an isolated grasp hook to the copied gripper description. It never modifies `C:/dev/ros2-sim`, `ros2-sim2`, their manifests, package caches, original launchers or other demos. Its preparation receipt records source and plugin SHA256 values. Logs, generated descriptions and the MoveIt warehouse database stay in the overlay.

The package path is reserved for IO-Link Workbench's environment work. OpenCVSimpleTools owns the app/planning adapters. Runtime root/ROS-domain/Gazebo-partition allocation belongs to the ROS installer owner. Do not launch until that allocation is confirmed.

## Prepare and launch

Use the allocated native RoboStack environment. From this package directory, offline preparation needs only Python:

```powershell
python -m opencv_bin_picking_fixture.prepare --runtime-root C:/dev/ALLOCATED_RUNTIME --output C:/dev/NEW_PRIVATE_OVERLAY
python -m unittest test.test_fixture -v
```

An optional `--scene scene.json` supplies the same metre-based scene schema as `default_scene()` in `scene.py`. Existing output directories are rejected. Use one overlay per runtime session. The default scene has no placement targets; those are application settings.

Set the agreed isolation before a headless source launch. Domain 130 is reserved for this fixture; use a unique partition per session and localhost discovery:

```powershell
pixi run --locked --manifest-path C:/dev/ALLOCATED_RUNTIME/pixi.toml python -m opencv_bin_picking_fixture.run --overlay C:/dev/NEW_PRIVATE_OVERLAY --partition opencv_bin_pick_UNIQUE_SESSION
```

Run from this package directory or expose its path to the activated child. The wrapper applies domain 130, localhost discovery, the unique partition and overlay-local ROS state/logs **after Pixi activation**, which otherwise overrides outer shell settings with the native manifest's saved defaults. Add `--plan-only` to inspect without starting ROS. One launch record is created exclusively per overlay. Use a new overlay after stopping/restarting a session.

This launches the copied native UR5e control/MoveIt files, bridges and fixture node. Default headless mode does not start Gazebo GUI or RViz. Source launching needs no installation into the shared environment; the standard `ament_python` package can also be built into a separate ROS overlay. Never launch through the baseline wrapper that hardcodes shared generated paths.

## Public ROS contract

World frame is `world`; the camera optical frame is `opencv_fixture_camera_optical_frame`. Status includes the scene-defined `world_from_source_camera` transform and the same transform is published on static TF. Scene extrinsics are known simulation geometry, not estimated calibration. Intrinsics come from the original CameraInfo. The source optical convention is +X right, +Y down, +Z forward; depth is interpreted using its original encoding and K.

| Endpoint | Type | Meaning |
|---|---|---|
| `/opencv_fixture/source/rgb` | `sensor_msgs/msg/Image` | Original simulator RGB, not an evaluation rendering |
| `/opencv_fixture/source/depth` | `sensor_msgs/msg/Image` | Original axial depth, not filled from object poses |
| `/opencv_fixture/source/camera_info` | `sensor_msgs/msg/CameraInfo` | Original image geometry/K/distortion |
| `/opencv_fixture/status` topic | `std_msgs/msg/String` JSON | Session, run, source/actuator metadata, clock activity, pending operations and attachment states |
| `/opencv_fixture/status` service | `std_srvs/srv/Trigger` | Same JSON snapshot in `message` |
| `/opencv_fixture/evaluation/<id>/pose` | `tf2_msgs/msg/TFMessage` | Privileged raw simulator pose; parent may be the named Gazebo world |
| `/opencv_fixture/evaluation/poses` | `std_msgs/msg/String` JSON | Privileged normalized world poses, dimensions, ages and attachment states |
| `/opencv_fixture/evaluation/snapshot` | `std_srvs/srv/Trigger` | Same privileged geometry snapshot |
| `/opencv_fixture/objects/<id>/attach` | `std_srvs/srv/Trigger` | Request gated assistance; does not report success as an observed grasp |
| `/opencv_fixture/objects/<id>/detach` | `std_srvs/srv/Trigger` | Request release of an observed attachment |
| `/opencv_fixture/objects/<id>/state` | `std_srvs/srv/Trigger` | Attachment state and pending operation |
| `/opencv_fixture/reset` | `std_srvs/srv/Trigger` | Explicit whole-simulation reset after detached/stationary checks; requires observed settling |

Default IDs are `part_01` through `part_06`. Trigger `success=false` means rejected before issuing the requested operation. A successful request contains an operation UUID with `outcome=requested`; **wait for `outcome=observed` and the requested attachment state before moving**. Native DetachableJoint output has no operation UUID, so this UUID is local correlation, not a native transaction guarantee. A two-second missing acknowledgment becomes `outcome_unknown`, blocks further requests and must not be retried automatically. A matching late native observation can resolve it; otherwise stop/restart the isolated fixture.

Every response carries `schema=opencv.bin_picking_fixture/v1`, `session_id`, `run_id` and `simulation_only=true`. Session UUID changes on node startup. Run UUID changes only after a successful reset-control response and fresh settled initial part poses. Match these IDs and `scene_sha256` before consuming results; node restarts are reconnect boundaries. A failed/unconfirmed reset remains fenced. Reset is a separately labeled simulation operation, never a pick or place.

Status `ready` requires fresh RGB/depth, observed CameraInfo, advancing simulation time, TCP/joint feedback, every object pose and no pending operation/reset. `phase` is `waiting_for_observations`, `ready`, `operation_pending` or `resetting`. CameraInfo is retained calibration within a session, while image/pose/TCP freshness uses host monotonic receive age. The status heartbeat uses a steady clock so it continues when simulation time pauses. Source image/frame validity and native behavior still need runtime acceptance.

Status declares `assistance=pose_gated_fixed_joint`, `friction_grasp_validated=false`, source topic/frame names, camera transform, planning group `ur_manipulator`, TCP `gripper_tcp`, arm action `/joint_trajectory_controller/follow_joint_trajectory`, execution action `/execute_trajectory` and gripper action `/gripper_controller/follow_joint_trajectory`. Native MoveIt IK/planning/scene services remain unchanged. `physics.time_recently_advancing` describes observed clock progress; it is not an assertion of contact stability.

## Assisted-grasp limits

Gazebo's DetachableJoint is added per object to the **private generated robot** with `initial_attach=false`; no part is attached at startup. `gripper_base_link` is preserved against fixed-joint lumping. The fixture sends attach only for fresh world-object/TCP/joint feedback, a stationary arm, an upright square box, downward aligned jaws, horizontal error <=6 mm, TCP 6..18 mm above its centre and both observed finger positions within 2 mm of half box width. Source-camera observation must also be available. The generic fingers extend 25 mm beyond TCP: a 40 mm box normally needs TCP at source top minus 5..10 mm to clear its support.

These are explicitly **truth-based pose gates**, not a validated contact sensor or friction grasp. No set-pose command is used for attach, lift, release or placement. An acknowledgment establishes the simulated joint state; verify actual object following during lift and settling after release separately. The app must update world/attached collision objects using the privileged geometry contract and keep it separate from source-depth candidate estimation.

## Verification

Pure tests cover actual initial geometry separation, source/evaluation separation, optical-axis geometry, initially-unattached per-object plugin settings, fresh/invalid/stale/jaw gates, single-object ownership and unknown/late outcomes. Native SDF parsing, camera frame/encoding/K, rendered occlusion, collision-free reach, observed lift/release and repeated reset remain required runtime checks. No physical camera/robot or commercial simulation license is part of this fixture.

The initial source checkpoint passed **8 offline tests**, installed Gazebo's world SDF validation and actual URDF-to-SDF conversion retaining `gripper_base_link` and all six initially-unattached plugins. Native ROS interface imports passed without initializing ROS. These are parser/source checks, not a running scene, a verified camera stream or a successful grasp. Runtime allocation and all dynamic checks remain pending.

The subsequent native headless check used the allocated second RoboStack runtime with domain 130 and a unique Gazebo partition. It passed fresh exact-stamp original **960×720 rgb8 / 32FC1** pairs, CameraInfo/frame/geometry consistency, three first-return-visible upper stack centres and three lower centres occluded by nearer surfaces. All five required MoveIt services, ExecuteTrajectory and both controller trajectory actions were available; GetPlanningScene returned eight arm/gripper joints. No motion goals were sent by this verifier. This proves centre-ray occlusion, not complete lower-part silhouette occlusion.

Run the maintained read-only verifier **before application motion**, inside the same activated/isolation environment: `python -m opencv_bin_picking_fixture.verify_native --output NEW_EVIDENCE_DIRECTORY`. It retains original RGB/depth bytes, stride/encoding/K, hashes, session context and explicitly evaluation-only projection checks. It assumes the initial three-stack scene and refuses an existing evidence destination. The application owner has sole arm/gripper/attachment/reset control after handoff; the fixture owner must not run overlapping actions.

Primary references: [Gazebo 10 DetachableJoint source](https://github.com/gazebosim/gz-sim/blob/gz-sim10/src/systems/detachable_joint/DetachableJoint.cc), [PosePublisher source](https://github.com/gazebosim/gz-sim/blob/gz-sim10/src/systems/pose_publisher/PosePublisher.cc), [official RGB-D bridge example](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim_demos/config/rgbd_camera_bridge.yaml). Installed plugin support is checked separately; documentation for other versions is not proof of native behavior.

MIT license; see [LICENSE](LICENSE). All scene geometry is original primitive geometry.
