# Smooth a recorded camera path

Open [offline_camera_stabilization.json](offline_camera_stabilization.json)
with **File → Open…** and run all 48 frames. **Batch serial entries**
buffers the excerpt, **Stabilize recorded sequence** corrects it, and
**Unbatch serial entries** emits the corrected frames in source order
with their original tickets. This project uses the bundled
[TUM RGB-D clip](../recorded_camera_stabilization/README.md).

The stabilizer estimates translation or similarity motion between
consecutive frames. A centered Gaussian averages the observed camera
path using up to **Radius frames** on each side. **Sigma frames**
controls the weight of distant neighbors. When a motion fit is rejected,
the next frame starts a new segment with its own coordinate root.
Windows shorten at segment boundaries. An isolated frame uses its
original pose.

On the example clip, RMS second difference of the estimated image-center
path drops from **3.35 to 0.66 pixels per frame squared**. Each corrected
frame retains at least **94.1%** coverage.

- Batch **2–64** matching images from one recording. Corrected images and
  masks together are limited to 128 MiB. Each batch has its own path.
- Connect a static background **Mask** or **Region** to constrain the
  feature fit in raw-frame coordinates. If a frame falls below
  **Min coverage fraction**, the block reports the failed frame and
  withholds the batch.
- **Corrected**, **Coverage**, and **Transforms** are groups with matching
  order and per-frame tickets. Each transform maps the original pixels to
  its corrected grid. Within a segment, each corrected grid also maps to
  the segment's anchor.
- **Report** includes the fit result, segment index, smoothing window,
  observed pose, smoothed pose, and coverage for each frame. Window
  bounds are zero-based, with an exclusive end. Poses contain x and y
  displacement in pixels, angle in radians, and log scale.

```powershell
opencv-dataflow-run examples/flow_control/offline_camera_stabilization/offline_camera_stabilization.json --all-images
```

[All examples](../../README.md)
