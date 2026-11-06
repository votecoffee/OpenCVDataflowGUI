# Lens fit report

Open [lens_fit_report.json](lens_fit_report.json) with **File → Open…**
and click **Run once**. **Calibrate lens** fits a pinhole profile from
six rendered circle-grid views. The project builds correction maps
from the fitted profile and displays their displacement.

Inspect **Fit report** on the calibration block. The overall RMS is
about 0.196 pixels. The sixth image has deliberately displaced dots
and an RMS of about 0.469 pixels. Its zero-based view index is 5.
Set **Max view RMS pixels** to 0.3 to make the sixth image fail
the fit while the overall RMS stays within its limit.

Each view includes a Rodrigues rotation in radians, translation in the
configured board units, and estimated standard deviations for both.
The pose maps board coordinates to camera coordinates. OpenCV reports
zero standard deviation for parameters it holds fixed. Unavailable
estimates appear as null. The report contains fit statistics for the
input views. The calibration profile retains its original format.

```powershell
opencv-dataflow-run examples/calibration/lens_fit_report/lens_fit_report.json
```

[OpenCV calibration reference](https://docs.opencv.org/5.0/main_modules/calib.html#calibratecamera) · [All examples](../../README.md)
