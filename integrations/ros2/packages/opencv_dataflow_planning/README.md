# opencv_dataflow_planning

Experimental ROS 2 **plan-only** adapter and simulation-result summary tools.
The application owns goals, frames, scene/robot revisions, tool configuration,
workflow and result review. This package forwards bounded planning jobs to an
existing MoveIt installation and publishes standard trajectory previews.

## Contents

- [Install](#install)
- [Usage](#usage)
- [Contracts and limits](#contracts-and-limits)
- [Verification](#verification)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Install

Source is a conventional `ament_python` package with `package.xml`, a resource
marker, setup files and two console entry points. Build it in an isolated ROS
workspace using the same Python/ROS installation as MoveIt. Keep colcon
build/install/log directories outside tracked source. Do not install rclpy or
ROS binary extensions into OpenCVDataflowGUI's interpreter.

```powershell
colcon --log-base C:/path/local/log build --base-paths C:/path/opencv_dataflow_planning --build-base C:/path/local/build --install-base C:/path/local/install
```

Activate that workspace overlay in the matching ROS environment. Runtime
dependencies are `rclpy`, `rosidl_runtime_py`, `std_msgs` and `moveit_msgs`.
No Open3D/PCL/cuRobo/GPU dependency is required by this initial package.
The inspected Windows Lyrical runtime has the ROS interfaces but does not yet
have `colcon_core`. This increment built an isolated overlay using external
build tools (colcon-core 0.21.3, colcon-ros 0.5.0, colcon-python-setup-py 0.2.9,
setuptools 79.0.1) without installing anything into the ROS or app environment.
Provide colcon build tools in your development environment before building.

## Usage

Launch the backend with a configured robot revision and service/topic names
supplied by the app's deployment profile:

```powershell
ros2 run opencv_dataflow_planning plan_bridge --ros-args -p robot_revision:=YOUR_ROBOT_CONFIG_REVISION
```

Defaults are `/get_planning_scene`, `/plan_kinematic_path`, and
`/simple_tools/planning/{request,result,display_trajectory}`. These are ordinary
ROS node parameters; no robot file, application sequence, bin region or gripper
configuration is hardcoded into the backend. The bridge starts no MoveIt server,
simulator, controller or hardware connection.

The existing app data publisher/reader can send/receive `std_msgs/msg/String`
in JSON-string mode. A first request is:

```json
{"schema":"opencv_dataflow_plan_request_v1","operation":"snapshot","request_id":"snapshot-1","run_id":"run-1"}
```

The result returns `server_id`, `scene_revision`, configured `robot_revision`,
full robot state and scene summary. A plan request has those IDs plus:

```json
{
  "schema": "opencv_dataflow_plan_request_v1",
  "operation": "plan",
  "request_id": "plan-1",
  "run_id": "run-1",
  "server_id": "COPY_FROM_SNAPSHOT",
  "robot_revision": "YOUR_ROBOT_CONFIG_REVISION",
  "scene_revision": "COPY_64_CHARACTER_DIGEST_FROM_SNAPSHOT",
  "plan": {
    "group_name": "arm", "frame_id": "base_link",
    "joint_names": ["joint_a", "joint_b"],
    "joint_types": ["revolute", "prismatic"],
    "start": [0.0, 0.0], "goal": [0.2, 0.1],
    "tolerance": [0.001, 0.001], "start_tolerance": 0.0001,
    "workspace_min": [-1.0, -1.0, -1.0], "workspace_max": [1.0, 1.0, 1.0],
    "allowed_planning_time": 2.0,
    "velocity_scale": 0.2, "acceleration_scale": 0.2,
    "pipeline_id": "", "planner_id": ""
  }
}
```

The two-joint example describes an analytic fixture, not the UR5e. Supply actual
group/joint/type configuration and a current complete scene. Angles are radians,
prismatic positions and workspace lengths are meters. Scale factors must be
strictly positive: MoveIt would interpret zero as its full-speed default, so
this adapter rejects zero before sending. Full non-group joints and attached
bodies are retained from the actual scene snapshot.

Successful structural checks return status **preview**, metadata/digest/timing
and a standard `moveit_msgs/msg/DisplayTrajectory` for RViz. This is a planning
preview, never motion authorization. No execute/action client or joint command
publisher exists. App-side request controls, trajectory rendering and a finished
clutter example are future integration work, not new GUI blocks in this package.

`cancel` uses the active request's ID/run/server; `reset` uses a new request ID,
the current server ID and the new run ID. A reset is rejected while an unresolved
service request is draining. Provider services have no action-cancellation
protocol: timeout/cancel retires the output and waits for the outstanding reply
without launching another provider call or publishing a late preview. A stuck
provider needs an operator-managed process restart. There is no automatic retry.

## Contracts and limits

- One in-flight job; bounded 1 MiB request/result JSON, 16 MiB scene hashing and
  10,000 trajectory points. Invalid, duplicate, old-run and old-server inputs
  retain explicit failure statuses. Recent duplicate IDs are remembered for
  128 requests; use globally unique IDs within each run.
- The scene digest retains world geometry/octomap, robot state/attached bodies,
  transforms, allowed collisions and padding while ignoring changing header
  stamps. The requested start must agree with the sampled scene.
- GetMotionPlan uses the server's live scene. Equal snapshots before/after are
  **not an atomic scene lock** and do not prove that nothing changed between
  them. Robot revision is an operator configuration assertion, not independently
  read-back URDF/SRDF identity. State freshness and clock synchronization remain
  explicit future checks.
- Checks cover provider status/group, complete expected joint set, finite
  samples, strictly increasing times, start/goal agreement and original start
  state. The initial route supports fixed-base **joint goals**. Pose IK, Cartesian
  insertion/retreat, continuous-joint equivalent-angle handling, task sequences,
  attachment phases, dense independent collision checking and execution are
  future work. Collision validity is recorded as provider-reported.
- RViz may retain an older trajectory. Only a correlated fresh **preview** result
  identifies a new valid output event. Keep its request/run/scene/trajectory hash
  visible in the app's review; a retained rendering is not new evidence.

`planning_benchmark EPISODES.jsonl` reports P50/P95 planning wall latency by frozen
case/backend/warm-or-cold group and separates previews from simulated outcomes.
Each line has exactly `episode_id`, `case_id`, uint32 `seed`, `backend`, boolean
`warm`, `planning_status` (preview/no_plan/timed_out/invalid/cancelled),
`planning_wall_s`, `simulated_outcome` (success/failed/not_run/unknown), and
`simulated_cycle_s` (number or null). A success needs a preview and recorded
cycle time. Not-run/unknown outcomes are counted explicitly and never become
successful picks. Preserve the frozen case's robot, physics, sensor and scene
hashes separately; the script does not run a simulator or verify physical truth.

## Verification

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest test/test_contracts.py -q
python test/native_smoke.py C:/path/new-native-receipt.json
```

The native check explicitly uses localhost DDS domain **129** and unique fixture
topics/services. It needs the ROS runtime and creates only analytic ROS services,
the bridge and observers, then shuts them down. It does **not** invoke MoveIt
planning, Gazebo or robot motion. It checks native field/transport compatibility,
previews, complete start state, stale scene/duplicate/run handling, malformed
success, timeout and cancellation with late-response suppression. Run it in a
reserved test domain; do not treat its analytic trajectory as a planner result.

Initial verification: **22 core checks**, isolated ament_python colcon build,
and native DDS fixture replay. Neither a real MoveIt planning scene nor a
Gazebo contact/grasp benchmark has been qualified yet.

## Maintainers

votecoffee.

## Contributing

Keep application settings/workflow in the app. Preserve standard ROS interfaces,
optional imports, headless tests, bounded jobs and explicit qualification limits.
MoveIt/cuRobo comparison must use identical frozen scenes/goals and include
warm-up, conversion, scene validation, transport and failure outcomes.

## License

Original package code is [MIT](LICENSE). Runtime dependencies retain their own
terms. [MoveIt 2](https://github.com/moveit/moveit2/blob/main/LICENSE.txt) uses
BSD-3-Clause. No upstream source, robot asset or learned model is vendored here.
