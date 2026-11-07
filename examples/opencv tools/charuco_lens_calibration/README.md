# ChArUco lens calibration

Open [charuco_lens_calibration.json](charuco_lens_calibration.json)
with **File → Open…** and click **Continuous** to process the eight
rendered views. **Detect ChArUco board** pairs the visible image corners
with their board coordinates. **Batch serial entries** collects all
eight observations before **Calibrate lens** fits the profile.

The images contribute 16 to 24 corners each. The fitted profile measures
about 0.142 px overall reprojection RMS. **Fit report** shows each image's
name, point count and error. The final block uses the fitted profile to
build a pair of lens correction maps.

For your own fit, use 3 to 64 views of one board on the same camera pixel
grid. Match the batch size to the number of views. The fitter accepts the
batch on **Observations** and uses the board's units. Leave **Image paths**
empty for this workflow. Each observation must pass detection and contain
distinct, noncollinear matched points. The calibration uses the corner
IDs present in each view, including when the visible sets differ.

```powershell
opencv-dataflow-run "examples/opencv tools/charuco_lens_calibration/charuco_lens_calibration.json" --all-images
```

[OpenCV board calibration](https://docs.opencv.org/5.0/tutorials/objdetect/aruco_calibration/aruco_calibration.html) · [All examples](../../README.md)
