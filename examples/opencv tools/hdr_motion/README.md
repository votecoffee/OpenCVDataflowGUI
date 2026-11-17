# HDR motion and measured support

Open a project and run once. Three image sources form an exposure bracket at
0.25, 0.5 and 1.0 seconds. **Merge HDR with motion rejection** produces measured
radiance and an explicit Valid mask. Image statistics uses that mask. A separate
reference-filled Display is tone-mapped for the filmstrip, with rejected regions
outlined. **Conventional merge** supplies an ordinary native Debevec comparison.
No camera, plugin, download or external service is needed.

| Project | Measured pixels | Inconsistent pixels | Reference fills | Expected |
| --- | --- | --- | --- | --- |
| [Static](static.json) | 15,360 | 0 | 0 | PASS; exact linear radiance, including highlights clipped in the long exposure. |
| [Moving](moving.json) | 15,000 | 360 | 360 | PASS; three 12×10 part positions are excluded. |
| [Strict coverage](strict.json) | 15,000 | 360 | 360 | Intentional FAIL at 100% measured-coverage requirement; current diagnostics retained. |
| [Ignore clipping](ignore_clipping.json) | 15,120 | 240 | 240 | PASS; deliberately weaker evidence misses the third, long-exposure-clipped part. |
| [Missing support](missing.json) | 15,296 | 0 | 64 | PASS; an 8×8 all-clipped patch has no measured HDR. |
| [Native alignment](aligned.json) | 7,676 of 7,680 | 0 | 4 | PASS; MTB removes known shifts; padding is excluded from evidence. |

The moving part appears at a different location in each exposure. The reference
contains only the middle part. Default clipping bounds still detect a contradiction
at the third location, although that exposure is saturated and cannot contribute
radiance. Ignoring clipping loses that evidence. Touch the Clipping evidence
dropdown to compare both behaviors. Other tolerances are zero in these exact controls.

All pixels outside Valid mask are zero in Image, including displayed fills. Those
zeros are missing data, not measured darkness. Display/filmstrip fills may be clipped
reference values and are review-only. The native comparison produces ghost detail
at the first two moving-part positions; it has no validity output. Inspect the
HDR Report for contributor counts, clipping, source tickets and coverage.

The shifted control applies source shifts `[2,1]`, `[0,0]`, `[-2,-1]`. Native MTB
finds their inverses. Its support metadata keeps padded borders from becoming
photometric evidence. Four corner pixels have only one available exposure and
are excluded. All other controls explicitly assert their authored pixel alignment.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/hdr_motion/moving.json"
.\.venv\Scripts\python "examples/opencv tools/hdr_motion/build_example.py"
```

## Attribution and independent truth

All images, radiance scenes, generator, graphs and expected masks/measurements
are original work by OpenCVDataflowGUI contributors under the repository MIT
license. No customer or third-party images are included. Each project contains
an attribution comment. Stored codes are generated as `radiance*128*seconds`,
with independently specified clipping, object positions and camera shifts.
`truth.json` pins 18 PNG hashes, exact binary-mask hashes, coverage, selected
mean radiance and expected alignment. Tests regenerate assets and graphs in a
fresh folder and compare exact bytes, then verify Debug/Lean numerical results.

These are synthetic linear-response controls, not a calibrated camera or proof
that every moving object is detectable. Lighting, response error, pose error,
clipping and lookalike content affect the proxy. See
[contracts and limits](../../../docs/hdr-motion.md) · [All OpenCV examples](../README.md).
