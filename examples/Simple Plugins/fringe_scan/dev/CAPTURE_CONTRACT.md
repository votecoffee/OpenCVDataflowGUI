# SimpleFringeScan capture and recipe contract

The execution block consumes one complete registered capture stack. Sequence
names and metadata describe caller provenance; they do not verify physical
camera/projector synchronization, exposure calibration or a stationary object.

## Offline capture

Import a JSON manifest using `simple_fringe_capture/v1` (complementary Gray plus
phase) or `simple_fringe_phase/v1` (wrapped phase only). Each `frames` entry has
an observed `id` and relative `file`. Files resolve against the manifest folder.
See the retained [metric](studies/analytic_clean/capture.json) and
[phase-only](studies/warsaw_crown_phase/capture.json) examples.

Declare `capture_id`, `pixel_grid`, `exposure_s`, `source_encoding` (`linear` or
`srgb`) and `input_scale`. Import applies the declared normalization/response
once. A full plan specifies original `projector_size` [width, height], positive
X/Y `period_px` and `phase_steps`. Its observed IDs must match the generated plan:
black/white, complementary X/Y Gray bits, then equally spaced X/Y phase frames.
A phase-only manifest declares explicit `phase_angles` in radians and ordered
`phase:0`, `phase:1`, etc. frames. No metric calibration is allowed for phase-only.

## Connected capture

Use the block's array-preserving `capture` port, with a Python dictionary:

```python
{
    "schema": "simple_fringe_stack/v1",
    "images": images,          # float32/64 N x H x W, normalized linear codes
    "mask": mask,              # optional H x W nonnegative integer/bool mask
    "frame_ids": observed_ids, # exact IDs and order of the retained sequence
    "capture_id": "actual-capture-0001",
    "pixel_grid": "original-camera-grid",
    "response": "linear",
    "exposure_s": 0.01,
    "sequence": {
        "schema": "simple_fringe_capture/v1",
        "plan": plan_dict,
        "phase_angles": None,
    },
    "calibration": calibration_dict,  # None for uncalibrated/phase-only
    "metadata": {},                   # optional bounded JSON provenance
}
```

For phase-only, use its schema, `plan: None` and the explicit angle list inside
`sequence`. `capture_from_recipe(recipe)` builds this dictionary from a retained
reference; `retain_capture(capture)` makes a private reference copy. A connected
runtime source must match the retained dimensions, grid, sequence, angles,
exposure and calibration. Input is already linear; no implicit conversion or
exposure compensation occurs. Out-of-range, clipped, weak or nonfinite samples
remain unavailable. A runtime mask can restrict the trained mask, never expand it.

## Known calibration

Full scans may include `calibration` with separate `camera` and `projector`
records: full 3x3 `matrix`, original `[width, height]` `image_size`, OpenCV
`distortion` coefficients and printable `frame`. Supply a proper rigid
`rotation_cp` and `translation_cp_m` satisfying:

`X_projector = rotation_cp @ X_camera + translation_cp_m`.

Translation and output XYZ are meters, in the named camera frame. Calibration
is imported/reviewed, not estimated from the teaching images. The native
calibration import accepts these fields at top level with
`schema: "simple_fringe_calibration/v1"` and optional `units: "m"`.
Editing values invalidates previews and requires a fresh reference fit.

## Retention, results and bounds

Sealed `.sfs` recipes retain normalized reference arrays, exact mask, sequence,
known calibration, settings and provenance. Checksummed numeric members never
use pickle. Saving creates a new asset and never overwrites an earlier recipe.
Stage updates the inspector draft; Apply commits it. Independent test previews
do not replace the reference. Abandoned staged files may remain unused on disk.

The execution block requires an explicit connected stack or capture manifest.
It never silently processes training images. Output `image` is wrapped phase
in radians; `maps` contains original-grid numeric evidence; `report` describes
availability/rejections. `cloud` contains accepted finite metric points only,
in meters, with source pixels and optional ray-gap colors. It is absent for
phase-only or zero accepted geometry. `overview` and named graphics are review
images. Coverage PASS means availability only, not gauge or defect acceptance.

Limits: 16 million total capture samples; 160 MiB sealed recipe; one million
prepared camera rays for metric reconstruction. Pattern metadata can describe
larger projectors without allocating generated commands; actual generation has
its own 16-million-sample limit. Tiled/streamed high-resolution capture is future
work. Compiled phase basis/camera rays are reused across unchanged recipes;
capture data, projector rays and validity are recomputed every run.
