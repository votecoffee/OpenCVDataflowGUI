# Bin picking to an arranged table grid

A complete app-owned six-cell workflow for an isolated UR5e/Gazebo fixture.
Source data is the simulator's original first-return RGB-D camera, with original
CameraInfo and a known fixed optical transform. It detects visible box tops,
uses MoveIt for approach/lift/transfer/place trajectories, opens/closes the jaws,
waits for observed assisted attachment/release and checks settled grid placement.

## Install

Install the included `opencv-simple-bin-picking` source **0.1.0.dev2** from the
app repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .\integrations\simple_plugins\bin_picking
```

It uses the app's existing NumPy/SciPy/OpenCV; do not
install ROS binary extensions into that environment. Load this folder's
`plugin.py` explicitly using **Load Python plugin**, or CLI `--plugin`.

The separately built ROS packages are included in
[`integrations/ros2/packages`](../../../../../integrations/ros2/README.md):
`opencv_bin_picking_fixture` supplies the scene/camera/assistance;
`opencv_dataflow_pick_backend` supplies native planning/execution primitives.
Use a matched native RoboStack Lyrical/MoveIt/Gazebo environment and a trusted
local ROS runtime profile selecting its interpreter and installed backend overlay.
The saved project cannot choose an executable or launch a simulator.

## Usage

Open [project.json](project.json). **Enabled** defaults to false; Run once records
six disabled outcomes and a failed overall summary without connecting to ROS.
The graph is `Grid tasks -> Unbatch -> Pick one -> Batch(6) -> Summary`.

Prepare/launch the fixture using its package README. Use the allocated domain
130, localhost discovery and a unique `opencv_bin_pick_` Gazebo partition, applied
after Pixi activation. Set the same partition/domain and
`OPENCV_DATAFLOW_ROS2_PROFILE` for the app process. Do not share motion ownership
with another app/chat while running.

Set grid origin/pitch/rows/columns on **Bin to table grid** and detector ROI/floor/
part height/retract on **Simulated pick and grid place**. Planning time and
velocity/acceleration scales are separate robot execution controls. Coordinates are world
metres. The default grid is 3 columns by 2 rows, 100/140 mm pitch, on a 200 mm-high
table. The last cell parks the arm away from the camera's view of the grid.
If changing the number of cells, set **Batch serial entries → Size** to the same
rows-times-columns count so every outcome reaches the summary.
Enable simulation, then Run once. This deliberately performs real simulated
trajectory actions; it is not a preview. Opening/validation performs no I/O.

Each invocation recaptures the camera; no scene is broadcast across the six picks.
The app may repaint only after the whole graph completes. The activity log names
each primitive phase. Per-cell archives are written under the project-contained
`.local/bin-grid-runs/<run hash>/cell-NN`: source PNGs, original float depth NPY,
calibration/capture stamps, support overlays, colored world clouds and complete
planning/execution/attachment receipts. The **graphics** port uses the native
flyaround point-cloud thumbnail/popout. Pose axes mark measured candidates;
they are not an executed trajectory trace. Joint trajectories are retained in
the receipts. The new **motion** output adds phase-colored planned TCP lines and
pink measured-joint-feedback FK dots to the native flyaround scene;
**motion_summary** provides the image legend. The live project's thumbnail uses
**motion**. Both paths use the retained MoveIt model, not independent TCP metrology.

For command-line execution of this same app graph against an already running
isolated fixture (the script does not start/reset a simulator):

```powershell
.\.venv\Scripts\python.exe "examples/Simple Plugins/bin_picking/bin_to_grid/run_project.py" --output .local/episode-new --runtime-profile .local/ros2-runtime.json --partition opencv_bin_pick_UNIQUE --execute-simulation
```

One uncertain/failed operation holds the sequence. Every remaining cell emits a
skipped outcome, so the summary still accounts for all six. No uncertain effect
is retried automatically. Reset/restart the fixture explicitly only after
checking stationary/detached state; repeated Run once does not repopulate a bin.
In-place Reset is disabled after its native clock rewind interrupted joint/TF
feedback. Stop the fixture and prepare a fresh overlay/partition to repeat.

## Evidence and limits

The native app run completed **6/6 placements**, **37 successful arm
trajectories**, and six observed lifts/releases. Independent final evaluation
found maximum position error **1.2382 mm**, all six parts detached, no MoveIt
attachments and no remaining box-top candidates in the bin. Source candidate
counts were **3/3/3/3/2/1**, with the lower boxes revealed after top removal.
[acceptance.json](acceptance.json), [summary.json](summary.json) and the six
per-cell archives retain that evidence. This is one authored-box episode.

Open [review.json](review.json) for offline review with no ROS installation or
simulator. Change **Recorded bin-grid pick → Cell** from 1 through 6 to inspect
its images, outcome and native cloud. The other two sources show the completed
grid and the combined native world cloud, including green evaluation bounds.
[result.png](result.png) is a diagnostic overlay; [final-camera.png](final-camera.png)
preserves the unaltered final source image. Raw source depth remains float32
NPY, without quantization/filling. Pose/box graphics never enter detection.

The original 42 BinPicking/backend checks passed, including saved-graph reopening and a
held third pick with the remaining three cells explicitly skipped. Native source,
motion and independent state evidence are separate from those offline checks.
[Motion review](motion_review/README.md) adds two completed native episodes
(baseline and shifted/rotated stacks), an explicitly held historical transfer,
and offline native path clouds. The detector settings are the same in both new
completed episodes. The older frozen assets and acceptance above remain intact.

These original primitive boxes are a simulation task, not an external industrial
dataset. Object truth is used for conservative collision geometry, association,
assistance gates and evaluation; candidate estimation uses only the camera depth.
The assisted grasp is an acknowledged temporary fixed joint, with no set-pose
operation during picking/placing. It does not validate friction, force closure,
physical suction or a real hardware controller. No physical robot adapter is
provided. Runtime/source/dataset licenses remain separate; see
[ATTRIBUTION.md](ATTRIBUTION.md).

## Maintainers

The OpenCVDataflowGUI project maintainers.

## Contributing

Preserve source-camera/evaluation separation, full collision checks, run/request/
server/fixture identities, uncertain-outcome holds and recorded native evidence.

## License

Original example code and primitive scene: MIT. Referenced software and rendered
robot model retain their own notices.
