# Calibration capture audit

Open [calibration_capture_audit.json](calibration_capture_audit.json) and run
once. Two independent sets of ten synthetic circle-grid views fit the same
pinhole lens. Both fits then validate a separate image. Review the audit report
and the two coverage plots. Each cell is labeled with the number of distinct
views that placed a detected point in that cell.

The sets cover all 48 cells. Their held-out RMS errors are about 0.057 and
0.058 pixels. The two fitted models differ by at most 1.52 pixels on the
63 sampled rays. The report also gives focal-length shifts, principal-point
shifts and distortion coefficient deltas.

[weak_capture.json](weak_capture.json) is a deliberate failure. Seven mostly
frontal views yield just 0.056 px fit RMS and cover only six center cells.
The audit flags the missing image and edge coverage and the limited tilt and
depth variation.

## Audit your captures

1. Connect **Fit report** from **Calibrate lens** to **Audit calibration
   capture set**. The fit report includes image coverage and board poses.
2. Optionally connect a second fit from an independent capture set. Both sets
   should use the same image grid and distortion coefficient layout.
3. Optionally validate each fit using **Validate lens calibration** on the same
   image kept out of both fitting sets. Connect the reports to the audit's
   holdout inputs. The audit verifies that the optics and measured points
   correspond, and flags exact reuse of an observed point set.

The coverage grid counts observed feature centers. Tilt is the span of
the target normal's X and Y angles relative to the camera axis. Relative
depth span uses the range of a common board center's depths divided by the
median depth. These checks describe capture diversity. The repeat comparison
samples the two lens models with the same rays. Held-out RMS uses fixed
intrinsics and estimates only the independent target's pose.

The circle grids are original synthetic fixtures under the repository's MIT
license. [Ground truth](assets/truth.json) includes the optical parameters,
target poses and projected circle centers. Rebuild both projects and their
assets into an empty directory.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/calibration_capture_audit/build_example.py" --output <scratch-directory>
```
