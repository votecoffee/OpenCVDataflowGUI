# Track a metric object silhouette

Open [silhouette_tracking.json](silhouette_tracking.json) and select **Run All**.
The graph tracks a 120 × 80 × 40 mm part across 12 frames. Select **Track metric
object silhouette** to inspect the projected mesh, matched image edges, pose,
and report. Expose **Result Graphics** to overlay the same drawings elsewhere
in the graph. Lean runs keep the pose and report and skip the drawings.

The **Camera** input provides undistorted pinhole intrinsics for the frame.
Set **Model** to a metric triangle mesh with consistently outward-wound faces
or connect a mesh. Set **Initial Pose** or connect a `pose_3d_v1` seed that maps
the object into the named camera frame. The example uses a seed offset from
the first frame by a few millimeters, and the tracker emits the refined pose
in meters with the current frame's ticket.

The report gives the fraction of search lines with a valid edge and the RMS
reprojection error perpendicular to the silhouette. If either check fails,
the pose output is withheld. Pulse **Reset** with a seed valid for the current
frame to restart. To inspect loss in this example, set **End Frame** in the
video source to `-1`. Frame 12 is blank and frame 13 shows the part again.
The tracking report should show `insufficient_edges` followed by
`reset_required`. The graph test checks recovery on frame 13 with a reset.

Use consecutive frames, a well-initialized pose, and a visible silhouette with
distinctive geometry. Symmetric objects can admit multiple poses. Large jumps,
occlusion, and strong background edges can make local tracking unreliable.

The 12 tracked frames have at most 0.99 mm of translation error and 0.34° of
rotation error against [ground truth](assets/ground_truth.json). Their largest
per-frame reprojection RMS is 0.13 px. The clip and mesh are original synthetic
assets by the OpenCVDataflowGUI contributors under the repository's MIT
license. [OpenCV RAPID API](https://docs.opencv.org/5.0/main_modules/rapid_8hpp.html)

```powershell
python "examples/opencv tools/silhouette_tracking/build_example.py" --output D:\temp\silhouette
opencv-dataflow-run "examples/opencv tools/silhouette_tracking/silhouette_tracking.json" --all-images --profile debug
```
