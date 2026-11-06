# Hand–eye calibration

Open [hand_eye_calibration.json](hand_eye_calibration.json) with
**File → Open…** and click **Run once**. The project fits the mounting
transform from eight synthetic pose pairs. It recovers a camera-to-gripper
translation of `[40, -25, 90]` mm. **Compose / invert 3D poses** uses a
different gripper pose to place the camera in base coordinates at about
`[252.55, 23.20, 351.66]` mm.

Find **Calibrate hand–eye** under **Calibration → Robot coordinates**.
Supply paired **Gripper poses** from gripper to base and **Target poses**
from the stationary target to camera. Two lists pair in order. Two ticketed
groups pair by identity. The samples must share frame names and units.

The Park–Martin solve uses OpenCV Rodrigues rotations and NumPy linear
algebra. The installed OpenCV 5 Python wheel lacks the hand–eye callable.
**Report** gives per-sample target consistency errors. Translation
thresholds are in millimeters and rotation thresholds are in degrees.
Setting a threshold to zero disables that check. When a limit is exceeded,
the report and RMS values remain available for review.

```powershell
opencv-dataflow-run examples/hand_eye_calibration/hand_eye_calibration.json
```

[OpenCV hand–eye frame conventions](https://docs.opencv.org/5.0/main_modules/calib.html#calibratehandeye) · [Park and Martin](https://doi.org/10.1109/70.326576)
