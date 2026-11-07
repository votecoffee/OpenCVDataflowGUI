# Stabilize recorded camera motion

Open [recorded_camera_stabilization.json](recorded_camera_stabilization.json)
with **File → Open…** and run continuously. The graph replays the first
48 frames of the bundled 320 × 240 clip. Frame 0 is the anchor. The
**Status** output shows when the stabilizer is ready and how much of each
corrected frame has valid source support.

**Covered raw error vs anchor** and **Covered corrected error vs anchor**
use the same coverage mask. The raw comparison sets **Mask alignment** to
**Pixel grid** so it uses the corrected mask's pixel indices on the
equal-size raw raster. Both measurements wait for coverage before
reporting a value.

Across the 47 measured frames, mean absolute error averaged **66.49**
grayscale levels before correction and **12.76** after correction.
Coverage ranged from **65.0% to 97.9%**. The graph stops before frame 48,
where coverage drops below the configured 64% floor. The bundled clip
includes 12 more frames for the automated test, which verifies that the
block withholds the rejected frame and resumes with a new coordinate
anchor. The comparison graph uses a fixed reference from the first
anchor.

```powershell
opencv-dataflow-run "examples/custom tools/recorded_camera_stabilization/recorded_camera_stabilization.json" --all-images
```

## Fixture provenance

The clip and anchor image are derived from the RGB movie for
**TUM RGB-D benchmark, freiburg1_rpy**, by J. Sturm, N. Engelhard,
F. Endres, W. Burgard, and D. Cremers, *A Benchmark for the Evaluation
of RGB-D SLAM Systems* (IROS 2012). These fixture assets are available
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Dataset and license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

The published AVI was trimmed to zero-based frames 384–443, resized from
640 × 480 by area averaging, converted to 8-bit grayscale, and encoded
losslessly as FFV1 at 30 fps. The bundled PNG is the first trimmed frame.
The source AVI SHA-256 is
`23195e68b589a567101696bd7484dcde41bf58ce25165a326597bc1cf080c789`.
[Source recording](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy-rgb.avi)

[All examples](../../README.md)
