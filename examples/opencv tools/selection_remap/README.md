# Lens correction with whole selections

Open a project below and run once. No downloads, camera, optional plugin or
external service are required. The original RGB16 ramp makes color-channel
measurements easy to check. A circle minus a rectangle supplies a selection with
a hole. The image and selection pass through the same Brown5 lens correction;
the corrected selection is measured as a region and as a statistics mask.
Drawable boundaries overlay the final filmstrip image.

| Project | Selected target pixels | Real-source support | Expected |
| --- | --- | --- | --- |
| [Lens](lens.json) | 3,780 | 19,200 | PASS: complete linear footprint. |
| [Coverage](coverage.json) | 4,029 | 19,200 | PASS: at least half selection coverage. |
| [Packed](packed.json) | 3,820 | 19,200 | PASS: packed 1/32-phase map sampling. |
| [Wide](wide.json) | 1,382 | 12,663 of 43,200 | PASS: invalid border fill stays unselected. |
| [Wrong pair](wrong_pair.json) | No output | No output | Intentional contained exception: X/Y maps from different corrections. |

Floating maps on the qualified OpenCV 5 build retain finer interpolation than
packed maps, so their boundary counts differ. This is expected. All four working
projects use replicated image borders; support remains independent of that fill.
`wide.json` widens the field of view and changes the output size to 240×180.
Measurements are target pixels, not physical square millimeters. The calibration
is an authored control, not a camera qualification.

Try changing the circle, hole, selection rule or coverage threshold. Select the
remap node to inspect its Mask, Coverage, Valid mask, Report and corrected image.
Downstream Image statistics exposes the mean of each RGB channel. Result graphics
can be selected for overlay; Lean retains numerical results while skipping drawings.
The intentionally invalid project demonstrates failure containment without stale
measurement or filmstrip output.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/selection_remap/lens.json"
.\.venv\Scripts\python "examples/opencv tools/selection_remap/build_example.py"
```

## Reference and attribution

All images, camera parameters, graphs and generator are original work by
OpenCVDataflowGUI contributors under the repository MIT license. No customer or
third-party images are included. Every graph contains the same attribution as a
layout comment. The lens model follows the standard
[OpenCV pinhole/Brown distortion equations](https://docs.opencv.org/5.0/main_modules/calib.html).

`build_example.py` directly evaluates those equations and explicit four-tap
sampling, without executing remap or measurement blocks to create expected
results. `truth.json` records the source PNG SHA256, exact binary-mask SHA256,
support/selection counts, centroids and three channel means. Integer sampling can
round individual image values; channel means allow an absolute one-count
tolerance. Mask hashes and areas must match exactly. Tests regenerate all assets
and saved graphs in a fresh folder and compare exact bytes. The fixture uses
identity rectification and Brown5 distortion; arbitrary hardware optics are not
qualified by these examples.

[Sampling, metadata and limits](../../../docs/selection-remap.md) ·
[All OpenCV examples](../README.md)
