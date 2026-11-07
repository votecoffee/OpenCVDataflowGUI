# Check stabilization on a second plane

Open [stabilization_validation.json](stabilization_validation.json)
with **File → Open…** and run the six synthetic frames. The upper
texture represents a far plane. The lower texture represents a near
plane that shifts faster for two frames, simulating parallax.

Both stabilizers fit motion from **Far plane for fitting**.
**Require near-plane agreement** also takes **Near plane for validation**.
On the fourth frame the far-plane fit still passes, but the near plane
has drifted about **4 pixels** from its predicted anchor position. This
exceeds the **2.5-pixel** per-track tolerance, so the checked branch
withholds its image, coverage, and transform. Inspect **Status →
validation** for the track count, inlier ratio, and median and 90th
percentile residual. The next frame resumes with a new anchor.

- Connect **Validation region**, **Validation mask**, or both to check
  a static inspection area. Their intersection is used when both are
  connected. The selected feature centers are excluded from fitting.
- The block seeds validation tracks when it establishes an anchor. It
  checks surviving tracks against the accumulated background motion in
  anchor pixels. This catches accumulated drift across smaller steps.
- Set **Minimum validation tracks**, **Max validation error pixels**, and
  **Minimum validation inlier ratio** to define how much support the check
  needs. Too few tracks or too much disagreement withholds the current
  output and establishes a new anchor.

The check measures where a single background-motion model breaks down.
Parallax, moving parts, or lost tracks can all trigger it. The correction
itself still uses one global translation or similarity transform.
[OpenCV sparse optical flow](https://docs.opencv.org/5.0/tutorials/others/optical_flow.html)

The [recorded validation example](../recorded_stabilization_validation/README.md)
uses the same check on near and far patches selected from recorded depth.

```powershell
opencv-dataflow-run "examples/custom tools/stabilization_validation/stabilization_validation.json" --all-images
```

[All examples](../../README.md)
