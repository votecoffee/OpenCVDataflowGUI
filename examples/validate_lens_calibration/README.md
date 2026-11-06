# Validate lens calibration

Open [validate_lens_calibration.json](validate_lens_calibration.json)
with **File → Open…** and click **Run once**. The profile was fitted
with the six images in the [lens fit report example](../lens_fit_report/README.md).
This project checks it on a separate rendered circle-grid view.
The 24 points measure about 0.043 px RMS and 0.080 px maximum error.

Find **Validate lens calibration** under **Calibration → Lens distortion**.
Connect the pinhole calibration profile and a grid image from its camera
pixel grid. Configure the validation board’s pattern, rows, columns, and
spacing. The block fits the board-to-camera pose with fixed camera
intrinsics and distortion. Each point in **Report** includes the
observed and projected pixel coordinates and their separation.

**Max RMS pixels** limits the overall error. **Max point error pixels**
limits the worst point. Zero skips the point limit. The example overlays
observed points and residual vectors on the image. **Residual scale**
changes the displayed vector length while measurements stay in pixels.

```powershell
opencv-dataflow-run examples/validate_lens_calibration/validate_lens_calibration.json
```

[OpenCV pose and projection reference](https://docs.opencv.org/5.0/main_modules/d_projection.html) · [All examples](../README.md)
