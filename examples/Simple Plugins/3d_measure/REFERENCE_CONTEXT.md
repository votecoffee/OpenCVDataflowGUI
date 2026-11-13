# Retained source context

Simple3DMeasure **0.1.0.dev4** keeps optional reference RGB, source image pixel
rows/size, and descriptive capture/calibration metadata beside reference XYZ.
Reopening restores the appearance and context without needing original paths.
Context never changes fitting, measurements, tolerances or runtime reports.

## Training and review

Import a colored PLY/PCD or an extended cloud NPZ, or capture a native connected
cloud. **Show source RGB** switches the teaching point colors while keeping
region handles and datum axes. Switch it off to inspect fit inliers/exclusions.
**Reference context…** shows the retained source evidence as read-only text.
Older recipes show that only XYZ is retained; missing colors or calibration
are never synthesized.

**Completed scene + fixture** keeps the original row order, RGB, pixel rows
and source metadata while transforming XYZ into the current teaching frame.
It retains the reference-to-source transform, original source frame, input
payloads and the host's snapshot/revision/execution identifiers when available.
Those identifiers describe a completed execution; they are not camera exposure
timestamps. Supplied timestamps/exposure/calibration remain upstream metadata.
Replacing a source invalidates training, including replacement of context on
identical XYZ. Stale background jobs cannot activate that reference.

The review popup offers **Fit evidence** and **Source appearance / measured
points**. These reuse frozen results and do not refit. In source mode the viewer
contains measured source rows only, in that result's source frame. Fit layers
remain available in the evidence mode. **Source context…** follows the selected
reference/completed run. Layer/appearance changes do not modify the recipe.

An export with context includes `source.npz` with the full selected capture's
XYZ/RGB/pixels/metadata, its SHA-256 and context in `report.json`, the exact
visible PLY, and the retained recipe. Reimport `source.npz` to reuse the frozen
source. PLY/PCD import currently preserves XYZ/RGB through the host reader;
it does not restore arbitrary calibration comments or source pixel rows.
Use NPZ or a native cloud input for that context.

## API and CLI

```python
from opencv_simple_3d_measure import ReferenceEvidence, train
from opencv_simple_3d_measure.cli import read_npz, write_npz

evidence = ReferenceEvidence(len(reference_xyz), colors=rgb, pixels=image_xy,
    source_size=(width, height), metadata={"camera": camera_record,
                                         "acquisition": capture_record})
recipe = train(reference_xyz, planes, measurements, units="mm", frame="part",
               reference_evidence=evidence)
write_npz("reference.npz", reference_xyz, units="mm", frame="part", evidence=evidence)
xyz, units, frame, evidence = read_npz("reference.npz", with_evidence=True)
```

The CLI `train` command retains this NPZ evidence. The original three-value
`read_npz()` return remains compatible. Base NPZ entries stay `points`, `units`
and `frame`; optional `colors` is Nx3 uint8 RGB, `pixels` is Nx2 int32 XY, and
`evidence` is a Unicode scalar containing JSON with `source_size` and `metadata`.
Use the writer to preserve these conventions. Source size is width/height;
pixels must lie inside it. RGB/pixel rows must match XYZ exactly. No normals,
full source image bytes or unobserved points are invented or retained here.

## Archive and evidence limits

Context uses archive **version 4**, with checksums for optional NPY arrays and
the combined canonical metadata/appearance. Existing XYZ-only recipes retain
archive versions 1/2/3 and their numerical report structure. Version 4 supports
plane, line and minimum-zone measurements together. Arrays and metadata are
copied at capture; metadata access returns detached JSON values.

The existing 64 MiB compressed/expanded archive limit still applies. Oversized
saves fail before replacing an existing recipe. Source reduction is explicit
upstream; the tool never drops context or downsamples measurements to fit.
Metadata accepts plain finite JSON: at most 256 KiB, 8,192 nodes, 12 nested
levels, 32,768 characters per string and 256 characters per key. Integer IDs
are preserved without float conversion. Unsupported arrays/custom Python
objects are rejected with a training error rather than silently coerced.

Metadata is descriptive evidence, not an automatically applied calibration
model. The numerical path still requires explicit units/frame and fixture.
No image registration, calibration compatibility, timestamp synchronization,
sensor uncertainty or physical accuracy is certified by retaining these fields.
Those stronger acquisition/qualification contracts remain roadmap work.

`examples/build_reference_context.py` uses pinned T-LESS RGB-D/calibration and
supplied poses with the same regions/bands as the earlier measurement example.
Its frame-13 numerical report must remain equal to that earlier report. Source
RGB/pixel rows come from frame 3. Dataset material retains CC BY 4.0 and the
published author attribution; generated previews are derived views.
