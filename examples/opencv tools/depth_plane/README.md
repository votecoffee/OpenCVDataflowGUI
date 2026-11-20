# Support plane and foreground separation

Open [depth_plane.json](depth_plane.json) and click **Run once**. The graph
loads a calibrated depth image, fits its tilted support surface, and sends
only points from the two raised objects to the foreground point cloud.
The graph uses original synthetic depth and independent reference labels.

The default run recovers the known plane at **0.78 mm RMS fit error**. The
foreground mask matches all **3,082** reference foreground pixels. A recess,
a shallow raised patch below the threshold, and missing depth stay out of
the cloud. The fitted camera-to-plane distance is 1.43999 m versus 1.44000 m
in the fixture.

## Use your own depth

Supply physical camera Z and the matching undistorted pinhole intrinsics.
For raw uint16 input, set units and units per sample. A mask or region can
limit which pixels teach the plane, for example an exposed part of a
conveyor. Classification still covers the whole frame.

The block uses a bounded three-point consensus search followed by an
orthogonal fit. It reports how many valid pixels in the fitting selection
support the plane. The reported equation is `n · XYZ + d = 0` in meters
with a unit normal. Positive signed distances lie on the camera side of
the plane and negative values lie behind it. The output is perpendicular
distance to the plane, measured independently of the camera's Z axis.
Missing depth is NaN in the distance image.

Set the plane tolerance for the sensor noise. Set the foreground cutoff
above the size of surface variation you want to ignore. Defaults are 4 mm
for plane support and 15 mm for foreground. The foreground-only depth
retains the source pixel grid, units and type, with all other samples
invalidated. When a plane fails the support checks, the block returns its
validity mask and fit report and withholds the plane and foreground outputs.

[Rebuild the fixture](build_example.py) using the installed app environment.
The asset labels are `0` missing, `1` plane, `2` foreground, `3` behind,
and `4` above the plane tolerance and below the foreground cutoff.

Original fixtures by OpenCVDataflowGUI contributors under
[MIT](../../../LICENSE). [OpenCV RGB-D depth backprojection](https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html).
