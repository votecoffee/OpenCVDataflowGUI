# Lens correction maps

Open [lens_correction_maps.json](lens_correction_maps.json) with **File → Open…**
and click **Run once**. Compare the bent lines in **Distorted grid** with the
straightened lines in **Correct the grid**. The project includes a synthetic
pinhole lens profile and a grid distorted with that profile.

Find **Build lens correction maps** under **Calibration → Lens distortion**.
Connect the pinhole profile from **Calibrate lens** or **Load calibration
profile**. The two float32 maps sample source X and Y coordinates for each
destination pixel. Wire both to **Remap image coordinates**. Reuse the pair
across image branches that use the same camera and calibrated pixel grid.
Remap checks the pair and the source dimensions and gives the corrected image
its own spatial frame.

Set width or height to zero to keep that calibrated dimension. **Scaled**
mode scales the original intrinsics to the output dimensions. **Optimal**
mode uses OpenCV's crop factor and optional centered principal point.
**Custom** mode accepts a 3×3 output camera matrix. An optional Rodrigues
rotation specifies rectification. **Rectified calibration** reports the
output intrinsics, zero distortion, and the rectification rotation.
**Valid ROI** reports OpenCV's `[x, y, width, height]` in optimal mode
with zero rotation.

```powershell
opencv-dataflow-run examples/calibration/lens_correction_maps/lens_correction_maps.json
```

[OpenCV map reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#initundistortrectifymap) · [All examples](../../README.md)
