# OpenCV hand-eye robot demo

Original `ament_python` source for an isolated, simulation-only UR5e wrist-camera
workcell. It renders a stationary metric ChArUco plate, plans a pose sweep in
MoveIt and captures original RGB/CameraInfo with exact-image-time robot TF.
Calibration and review run in the
[OpenCVDataflowGUI example](../../../../examples/industrial%20integration/ros2_hand_eye_calibration/README.md).

## Prerequisites and ownership

The allocated Windows runtime is `ros2-sim2` under `G:\dev`, with locked
RoboStack Lyrical / Gazebo Jetty / MoveIt. The ROS environment owner reserved
domain **44** and `opencv_hand_eye_` partitions for this example. Confirm the
allocation before reuse on another machine or concurrent session. Domain and
partition isolation are applied **after Pixi activation**. Leave the GUI Python
environment separate from native ROS extensions.

The runtime is read-only. Preparation copies the required native launch/config
helpers to a new private overlay, rebases generated output, adds the camera
there and records source/prepared hashes. It does not edit manifests, locks,
installed packages or existing robot environments. Use one new overlay and
unique partition for each session; no in-place world reset is supplied.

## Prepare, launch and capture

From this package directory, with the host application's Python for preparation:

```powershell
$devRoot = 'G:\dev'
$runtime = Join-Path $devRoot 'ros2-sim2'
$overlayPath = Join-Path $devRoot 'NEW_HAND_EYE_OVERLAY'
$scene = '../../../../examples/industrial integration/ros2_hand_eye_calibration/assets_diverse/scene.json'
python -m opencv_hand_eye_demo.prepare --runtime $runtime --config $scene --output $overlayPath
python -m opencv_hand_eye_demo.run --overlay $overlayPath --partition opencv_hand_eye_UNIQUE --plan-only
pixi run --locked --manifest-path (Join-Path $runtime 'pixi.toml') python -m opencv_hand_eye_demo.run --overlay $overlayPath --partition opencv_hand_eye_UNIQUE
```

The last command uses the activated native interpreter and remains running.
Default launch is headless. Add `--gui` for a requested Gazebo/RViz session.
In a second terminal, from the same package directory, set the same paths.

```powershell
$devRoot = 'G:\dev'
$runtime = Join-Path $devRoot 'ros2-sim2'
$overlayPath = Join-Path $devRoot 'NEW_HAND_EYE_OVERLAY'
$captures = Join-Path $devRoot 'NEW_CAPTURES'
pixi run --locked --manifest-path (Join-Path $runtime 'pixi.toml') python -m opencv_hand_eye_demo.capture --overlay $overlayPath --partition opencv_hand_eye_UNIQUE --output $captures
```

This sends **simulation robot motion**. It applies the target collision plate,
requests bounded MoveIt joint plans, checks interpolated post-retiming states,
executes each accepted trajectory and waits for stopped feedback before capture.
Planning/execution failures stop the run and retain the partial manifest.
Missing/stale inputs and clock rewind are held; uncertain execution is not
automatically retried. The package supplies no physical robot driver.

Read-only camera/TF preflight is available with `--observe-only`; its preview
capture is deliberately unsuitable for a complete calibration export.

Stop after acquisition with a generation-bound file request:

```powershell
$overlayPath = Join-Path 'G:\dev' 'NEW_HAND_EYE_OVERLAY'
$launchRecord = Get-Content -Raw -LiteralPath (Join-Path $overlayPath 'session-launch.json') | ConvertFrom-Json
@{action='stop';launch_id=$launchRecord.launch_id} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $overlayPath 'stop-request.json')
```

The runner asks its own ROS launch to stop. On Windows it also closes a leftover
Gazebo server only when its command line names this exact private world and its
process generation still matches. Verify `session-exit.json` and child completion
before starting the next session. The shared runtime is never reset or closed.

## Data and frames

- ROS RGB: `/hand_eye/camera/image`, original 1280×960 `rgb8`, 10 Hz.
- CameraInfo: `/hand_eye/camera/camera_info`, matching acquisition stamp/grid.
- Optical frame: `hand_eye_camera_optical_frame`, X right, Y down, Z forward.
- Robot pose: `base_link_from_tool0` queried from a persistent TF buffer at the
  image stamp, never latest TF substituted for an unavailable historical pose.
- Original bytes, ROS fields, joint feedback, run/launch identity, fixed
  training/held-out assignment and each motion receipt are retained.

The known scene/camera transforms are evaluation/design data. The host fitter
receives only detected board observations, intrinsics and acquired robot poses.
The scene has no contact/grasp assistance and performs no picks.

## Build and validation

Source execution above needs no installation into the shared runtime. A standard
`colcon build --packages-select opencv_hand_eye_demo` can install this ROS package
into a separately owned workspace with the listed `package.xml` dependencies.
That installed route requires its own overlay/runtime qualification.

With the app's Python, from the repository root:

```powershell
$env:PYTHONPATH = 'src;integrations/ros2/packages/opencv_hand_eye_demo'
python -m pytest -q tests/test_ros2_hand_eye_example.py integrations/ros2/packages/opencv_hand_eye_demo/test/test_scene.py
```

The metric scene generator additionally uses OpenCV, NumPy, SciPy and the
adjacent existing robot-FK review helper. Run it with the host interpreter only
when intentionally creating a **new** acquisition protocol:

```powershell
python -m opencv_hand_eye_demo.build_scene --robot-description C:/path/to/robot.urdf --output C:/path/to/NEW_ASSETS --diverse
```

The scene generator's numerical IK and field-of-view preflight aid acquisition
design; native MoveIt collision/constraint checks remain required. Synthetic
rendering and sampled collision checks do not establish physical accuracy or
continuous collision freedom.

## License

Original package code and generated primitive target geometry use the included
[MIT license](LICENSE). Native launch/config files are read from the separate
installed runtime and copied only into the user's private overlay, retaining
their notices. See the example's attribution for OpenCV, UR, ROS/MoveIt/Gazebo
and numerical dependencies. No vendor models or external dataset images are
bundled in this package.
