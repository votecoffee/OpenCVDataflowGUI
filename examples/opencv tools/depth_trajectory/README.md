# Track and fuse a recorded depth trajectory

Open [depth_trajectory.json](depth_trajectory.json) and select **Run All**.
The batch block gathers ten ordered exposures. **Track depth trajectory with
TSDF** anchors the first usable exposure at the origin and estimates motion
from the last accepted depth frame. It raycasts the fused surface at each
proposed pose and checks support and metric depth residual before integration.

Select the tracker to inspect **Trajectory** and **Surface Depth**. Each
trajectory row keeps the recording index, timestamp, ticket and status.
Frame 4 has no valid depth and receives no pose. Frame 5 aligns against frame 3,
passes the surface check and resumes in the same world frame. Set **Maximum
Consecutive Losses** to 1 to see the tracker withhold the segment after frame 4.

The tracker compares only measured camera-Z pixels with a valid fused surface.
**Model Depth Tolerance** bounds which pixels count as supported. **Minimum
Model Overlap** checks how much usable depth is supported, and **Maximum Model
RMS** checks the residual of those pixels. Choose the voxel size and tolerance
together since the rendered surface has finite voxel resolution.

Each completed batch has a fresh world frame and a fixed CPU TSDF volume.
The **Frames** and **Poses** outputs contain the accepted depth packets and
matching camera-to-world transforms in meters. They can feed the existing
posed depth fusion and mesh tools. The final surface depth is in the last
accepted camera grid and carries that camera's fused pose.

The ten-frame synthetic control tracked **9 frames** and recovered after
one loss. Maximum position error was **1.11 mm**, and maximum rotation error
was **0.17°** against the [analytic path](assets/ground_truth.json).
The scene and recording are original synthetic assets by OpenCVDataflowGUI
contributors under the repository's MIT license. The recording's companion
color stream is a depth preview. Tracking uses the metric depth planes.
[OpenCV Volume and raycast reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1Volume.html)

```powershell
python "examples/opencv tools/depth_trajectory/build_example.py" --output D:\temp\depth-trajectory
opencv-dataflow-run "examples/opencv tools/depth_trajectory/depth_trajectory.json" --all-images --profile debug
```
