# Stabilize camera motion

Open [camera_stabilization.json](camera_stabilization.json) with
**File → Open…**. Run continuously or step through the six PNG frames. The
first frame anchors the sequence. The remaining frames contain known shifts
of up to six pixels per axis and rotations of up to 1.6 degrees.

**Raw error vs anchor** measures mean absolute error across the incoming
frame. **Covered error vs anchor** measures the corrected frame using the
stabilizer's **Coverage** mask. On the bundled sequence the raw errors are
about 17–21 sample levels and the covered corrected errors are about 1.7–1.8.
The example includes every image it uses.

Find **Stabilize camera motion** under **Sequence → Temporal processing**.
Choose translation or similarity motion. Similarity includes rotation and
scale. **Follow** at zero holds the anchor. Intermediate values follow slow
movement while smoothing rapid changes. One follows the measured motion
without applying a correction.

- Use a mask or region to select stable background features, especially
  when parts move through the frame.
- **Coverage** marks corrected pixels with full input support. Wire it to
  downstream measurement masks to exclude uncovered pixels.
- **Transform** maps the current raw frame to the stabilized grid and back.
  The corrected image and coverage mask share the grid until the block
  re-anchors. They carry the current frame's ticket.
- **Status** reports readiness, track and inlier counts, and coverage.
  When motion or coverage checks fail, the block establishes a new anchor
  and waits for a usable estimate before publishing another corrected image.

The block uses past and current frames for causal correction. Registration
quality depends on stable, textured background features. The corrected
pixel grid provides image coordinates for inspection. Physical calibration
remains a separate step.
[OpenCV optical flow and tracking](https://docs.opencv.org/5.0/main_modules/video_track.html)

The [recorded camera example](../recorded_camera_stabilization/README.md)
uses the same coverage mask to compare raw and corrected error on a real clip.
The [two-plane validation example](../stabilization_validation/README.md)
checks whether the background fit also holds on a separate inspection area.
The [offline smoothing example](../offline_camera_stabilization/README.md)
uses frames on both sides of the current frame to smooth a recorded path.

```powershell
opencv-dataflow-run examples/flow_control/camera_stabilization/camera_stabilization.json --all-images
```

[All examples](../../README.md)
