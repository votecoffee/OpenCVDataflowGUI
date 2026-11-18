# Completed surface-map review

**Powered by Poly Haven.** Material controls use Charlotte Baglioni's
[Anti Slip Concrete](https://polyhaven.com/a/anti_slip_concrete), licensed CC0.
These lighting stacks are rendered diffuse controls, not acquired camera stacks.

Load [plugin.py](plugin.py) explicitly, open a project below, run in **Debug**, then
use **Review completed surface maps…** in the tool's inspector. The dev4 wheel is
SHA-256 pinned and loaded without installing/updating the host environment. Restart
before switching an already loaded package version. Manifest review needs a host
supporting explicit empty paired-input selections; no wired input is fabricated.

| Project | Fixed-setting evidence |
| --- | --- |
| [Analytic clean](projects/analytic_clean.json) | Generated diffuse geometry, 100% availability. |
| [Analytic contaminated](projects/analytic_contaminated.json) | Injected disagreement, missing and ceiling observations; original fit decisions retained. |
| [Four-light ambiguity](projects/four_light_ambiguity.json) | Competing normal fits remain unavailable; original coverage failure retained. |
| [Material clean](projects/material_clean.json) | Rendered from the attributed diffuse/normal maps; no real sensor claim. |
| [Material contaminated](projects/material_contaminated.json) | Rendered material with injected non-diffuse/missing observations. |
| [Connected material](projects/connected_material.json) | The same actual material capture transported through the array-bearing capture port, independently of training. |
| [Declared exposure change](projects/exposure_changed.json) | Half exposure with corresponding black-relative source-code change; imported lighting declarations stay explicit. |
| [All missing](projects/missing.json) | Zero source codes, no valid normal/albedo/relighting values, no stale result. |
| [Graphics off](projects/graphics_off.json) | Same raw maps/report; no emitted overview/views. Review-only colors use already stored arrays. |

Every case uses the same generic fit settings and **95% availability** gate. No
per-image tuning or retraining is performed. The first five use original source
declarations with only file paths repackaged; the connected variants carry explicitly
recorded capture archives. Reference and inspection captures remain separate.

The independent Results window links source lights, source-aligned maps and selected
pixel evidence. Shift-click either source-grid image or enter X/Y. Zoom/pan,
Focus/Overview, map/light/reference selection and per-light charts/tables change the
view only. Inspect raw codes, prepared observations, flags, fit usage and stored
residuals. Any displayed prediction is observation minus stored residual for an
accepted fit; it is not a new solve. Normals are directions, not measured XYZ/height.

Six [frozen exports](snapshots/) retain exact model/consumed manifest bytes, native
capture/prepared/map arrays, array-bearing lighting declarations, full JSON and
readable PNGs. Native precision, mask labels and unavailable NaNs are preserved.
Source/model files are not loaded during review/export, and preparation, fitting,
gradient/relighting and calibration are not repeated. Training drafts and Apply/Cancel
remain independent. New publications mark a frozen review stale; recapture explicitly.
Active/errored/Lean runs cannot supply a new capture.

See [attribution and scope](SOURCES.md), [workflow](COMPLETED_REVIEW.md),
[verification](verification.json), [wheel proof](wheel-verification.json) and
[file pins](integrity.json). Capture/grid/calibration identifiers are caller
declarations, not physical registration/synchronization or material/gauge qualification.

Replay through the existing host environment without installing anything:

```powershell
python examples.py verify . C:\path\to\new-surface-check --native
```

This validates pins, every numerical map and emitted raster, native review controls,
six frozen exports and byte-identical model Save As/reopen. Wired capture archives
are sealed assets. Manifest captures use the host's ordinary path-input rebasing
semantics when saved elsewhere; the frozen export is independent of those files.
