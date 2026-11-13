# Validate a ChArUco lens calibration

Open [validate_charuco_calibration.json](validate_charuco_calibration.json)
with **File → Open…** and click **Run once**. The profile was fitted from
the eight images in the [ChArUco lens calibration example](<../charuco_lens_calibration/README.md>).
This graph uses a separate, cropped view of the same rendered target.
It identifies 17 of 24 corners and measures about 0.068 px RMS and
0.128 px maximum reprojection error.

Connect the validation image to both **Detect ChArUco board** and
**Validate lens calibration**. Wire the detector's **Observation** to
the validator, and supply the lens profile for that camera pixel grid.
The validator fits only the board pose and measures the error of every
observed corner. The pose translation uses the board's length units.

**Report** includes each corner's board ID, its measured and projected
pixel coordinates, and the worst corner ID. The overlay shows the
residual vectors. **Max RMS pixels** and **Max point error pixels**
apply separate limits. **Residual scale** magnifies the displayed
vectors while the reported errors retain their pixel units.

```powershell
opencv-dataflow-run "examples/opencv tools/validate_charuco_calibration/validate_charuco_calibration.json"
```

[OpenCV pose and projection reference](https://docs.opencv.org/5.0/main_modules/d_projection.html) · [All examples](../../README.md)
