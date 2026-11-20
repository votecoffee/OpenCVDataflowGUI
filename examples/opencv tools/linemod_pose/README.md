# Find a metric pose from RGB-D templates

Open [linemod_pose.json](linemod_pose.json) and select **Run**. Two training
blocks teach a marked plate facing two directions. The search image shows
the quarter-turn view at a new position and under changed lighting.
Select **Find RGB-D template pose** to see the candidate box, pose axes,
native similarity, 3D anchor support and reprojection error.

**Train RGB-D pose template** needs an undistorted uint8 image, a registered
metric depth image, camera intrinsics, a training mask or region and a known
object-to-camera pose. Connect a previous model to add another named view.
The model is JSON-compatible and can be wired into the match block or saved
in its **Trained Model** parameter. Use matching intrinsics and pixel
dimensions for training and search.

By default the blocks check that color and depth share a registered optical
grid. The example uses **Pixel Grid** mode because its files were rendered
on the same grid and open as independent sources. The color and depth
files for each capture use the same ticket.

LINEMOD proposes candidates using gradients and depth normals. The match
block checks local appearance, estimates translation from corresponding
metric depth samples, and verifies reprojection. Rotation comes from the
selected training view. Teach additional views to cover more orientations.
The report retains candidate failure reasons, and the block emits a pose
only when at least one candidate passes the checks.

The saved search selects the quarter-turn view at **98.61%** native
similarity. All **41** anchor pairs agree, with **0.48 px** reprojection RMS
and **0.43 mm** translation error against the [analytic ground truth](assets/ground_truth.json).
The captures, calibration and plate design are original synthetic assets
by OpenCVDataflowGUI contributors under the repository's MIT license.
[OpenCV LINEMOD reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1linemod_1_1Detector.html)

```powershell
python "examples/opencv tools/linemod_pose/build_example.py" --output D:\temp\linemod
opencv-dataflow-run "examples/opencv tools/linemod_pose/linemod_pose.json" --profile debug
```
