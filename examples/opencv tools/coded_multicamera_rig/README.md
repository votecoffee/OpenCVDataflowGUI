# Calibrate a rig from a custom coded target

Open [coded_multicamera_rig.json](coded_multicamera_rig.json) and select **Run**.
The graph decodes the photographed markers, calibrates a three-camera rig,
and checks it on six separate target poses. Select either **Load coded rig
captures** block to inspect each camera's first usable image and the report
of detected IDs, duplicate IDs and missing views.

The [custom target](assets/target.png) has ten ArUco IDs with different marker
sizes and rotations. Its geometry is defined by the measured XYZ of each
marker corner in decoded order. To use the example target as print artwork,
set the full image to **440 × 320 mm** and measure the printed marker positions
before using it for calibration.

[The training manifest](assets/train.json) demonstrates the fields for a
custom rig.

- `camera_frames` names cameras in rig order. The first defines the origin.
- `image_sizes` gives one full image size per camera.
- `capture_ids` identifies synchronized target poses. Fit and validation
  captures use separate IDs.
- `images` is a camera-by-capture table. Paths are relative to the manifest.
  Use `null` when a camera has no image for a capture.
- `target` holds a `fiducial_target_v1` board. Use `m` or `mm`, one OpenCV
  dictionary, and four ordered XYZ corners per marker ID. The board lies
  in the XY plane. The layout can also come from the **Target** connector.
- Optional `camera_models` holds the existing intrinsics in manifest camera
  order. Connect it to the fitter in **Fixed** mode. **Fit** mode estimates
  pinhole or fisheye intrinsics from the observed corners.

The graph uses **Register cameras** initialization. Camera B connects the
two training groups. Each synchronized pair sees complementary marker IDs,
so the optimizer fits the shared target pose using the corners available
in each camera. The held-out check predicts observations in a second camera
from a pose estimated in the first. It includes two captures seen by A and C
together.

The synthetic control produced **0.55 px** fit RMS and **0.59 px** held-out
RMS over 12 cross-camera predictions. Maximum rig translation error was
**0.55 mm** against the [analytic ground truth](assets/ground_truth.json).
The target and captures are original synthetic assets by OpenCVDataflowGUI
contributors under the repository's MIT license.
[OpenCV multi-view calibration tutorial](https://github.com/opencv/opencv/blob/5.0.0/doc/tutorials/calib3d/camera_multiview_calibration/multiview_calibration.markdown)

```powershell
python "examples/opencv tools/coded_multicamera_rig/build_example.py" --output D:\temp\coded-rig
opencv-dataflow-run "examples/opencv tools/coded_multicamera_rig/coded_multicamera_rig.json" --profile debug
```
