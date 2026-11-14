# Optional ROS 2 packages

Custom ROS backend source lives under `packages/`. These are separately built
ROS packages, not native dependencies of the Python desktop app. Keep workflow,
robot/bin/tool settings, goals, backend selection, budgets and customization in
app projects/recipes and the trusted runtime profile.

## Available source

[opencv_dataflow_planning](packages/opencv_dataflow_planning/README.md) is an
experimental ament_python package with a **plan-only** MoveIt bridge and recorded
simulation-result summarizer. Existing ROS String JSON publishers/readers can
carry request/result envelopes. RViz uses standard DisplayTrajectory output.
The package contains no execution action/client, controller command publisher,
MoveIt server or simulator. No new planning block is registered by this folder.

The initial fixed-base joint-goal route checks scene/start agreement, keeps
non-group joints and attached bodies, bounds jobs and discards late replies after
timeout/cancel. Request/run/server/robot/scene identities travel with results.
Matching scene snapshots before/after a plan are not an atomic scene lock;
configured robot revision is an assertion rather than read-back model proof.
Status **preview** does not authorize motion or establish physical success.

The source package passed 22 core checks, a separate colcon overlay build and
native local DDS replay against analytic GetPlanningScene/GetMotionPlan fixtures.
No real planner, clutter scene, Gazebo grasp or physical motion was run by those
checks. The package's README gives build, request, benchmark and validation
commands. `package_sha256.json` pins the included package files.

## Build and runtime boundary

Build with colcon in a separate matching ROS development environment. Store
build/install/log directories in an ignored `.local` workspace or another local
directory. Do not add rclpy/ROS binary extensions, Open3D, PCL, cuRobo or a second
OpenCV wheel to the app environment merely to use this source folder.

The inspected Windows runtime has MoveIt interfaces but no preinstalled colcon;
the build was qualified with external build tooling, leaving runtime dependencies
unchanged. Deployment profiles select the ROS interpreter/overlay. Backend ROS
parameters supply endpoint names, robot revision and overall deadline; per-job
planning controls arrive from the app.

## Reuse and follow-up

Use existing `3d.cloud_filter`, `3d.cloud_voxel`, `3d.cloud_normals`,
`3d.cloud_transform` and native cloud views for the initial scene-preparation
workflow. Optional Open3D/PCL providers should add concrete missing operations
while retaining original/removed/unknown-point evidence. Filtering is not proof
of free space. Keep a conservative collision representation distinct from a
downsampled grasp-feature cloud.

Future work: app planning controls and trajectory/link-sweep review; real frozen
MoveIt empty/blocked/clearance tests; exact capture/model/state identity; pose-goal
IK and Cartesian approach/retreat; part attachments and correct jaw/TCP model;
independent collision/limit checks; physics/task outcome regression; isolated
cuRobo provider comparison. A geometric planning preview and a simulated grasp
outcome are separate measurements. Do not turn unknown/not-run episodes into
successful picks.

Primary references and licenses:
[MoveIt services](https://docs.ros.org/en/ros2_packages/rolling/api/moveit_msgs/srv/GetMotionPlan.html)
([BSD-3-Clause](https://github.com/moveit/moveit2/blob/main/LICENSE.txt));
[cuRobo](https://github.com/NVlabs/curobo) (Apache-2.0, current V2 API differs
from V1); [Open3D](https://www.open3d.org/) (MIT);
[PCL](https://pointclouds.org/) (BSD-3-Clause).
Russ Tedrake's [MIT bin-picking chapter](https://manipulation.csail.mit.edu/clutter.html)
informs the perception/planning/behavior separation; no course code or assets
are redistributed here. Dependency, robot mesh/model and dataset terms remain
separate. Original package code uses its included MIT license.
