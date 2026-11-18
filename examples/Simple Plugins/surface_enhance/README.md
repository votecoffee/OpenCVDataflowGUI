# SimpleSurfaceEnhance setup and execution

[Completed-results reviews](completed_runtime/README.md) add the pinned dev4 plugin,
nine projects and six frozen exports. Load that folder's `plugin.py` for independent
source/reference/light/map/pixel selection, zoom/pan, prepared observations, flags,
fit usage and stored residuals. Review/export does not prepare, estimate, relight,
recalibrate or reload source/model files. Manifest review requires the host's explicit
empty paired-input support (`da6de1a4`); no wired input or training substitution is
invented. Examples retain their rendered-control and CC0 attribution scope. No
environment installation is needed; the older dev3 loader/projects below remain.

**Powered by Poly Haven.** The public material maps are **Anti Slip Concrete**
by **Charlotte Baglioni / Poly Haven**, under **CC0**. Exact original files,
hashes and transformations are retained. See [dataset credits](DATASET_CREDITS.md).
All example light stacks are **rendered diffuse controls**, not acquired camera
stacks or evidence of measured surface accuracy. No metric height/cloud is emitted.

## Run the tool

In OpenCVDataflowGUI choose **Load Python plugin** and explicitly select this
folder's `plugin.py`. It verifies/loads the bundled `0.1.0.dev3` wheel without
installing anything into the host environment. Then open one of these projects:

| Execution project | What to assess | Availability judgment |
| --- | --- | --- |
| [Analytic clean](execution/analytic_clean.json) | Normal/albedo separation with a frozen reference. | Pass, 100%. |
| [Analytic contaminated](execution/analytic_contaminated.json) | Bright/dark disagreement and missing/ceiling samples. | Pass, 98.97%. |
| [Four-light ambiguity](execution/four_light_ambiguity.json) | Competing normals stay unavailable; raw evidence retained. | Fail, 61.79%. |
| [Concrete material clean](execution/polyhaven_clean.json) | Attributed material-map normal/albedo inputs. | Pass, 100%. |
| [Concrete contaminated](execution/polyhaven_contaminated.json) | Rejected observations and missing source pixels. | Pass, 99.69%. |

Every case uses the same generic fit settings and **95% available coverage**
threshold. This judgment does not decide defects, physical accuracy or surface
quality. Clean references are reused for the contaminated analytic/material
captures; no test-image parameter tuning or retraining occurs. The four-light
example retains its ambiguous reference and shows the missing evidence openly.

Run the graph and select the tool's **overview** result. The node thumbnail
shows combined source, normal/albedo, relighting, residual and availability
panels with a reason legend. Eight named maps appear in **Image results** with
result thumbnails enabled. Use a profile retaining previews, such as Debug.
The raw `image` output is floating orientation-only relighting with NaN gaps;
`maps` retains all 14 numerical arrays. Graphics can be disabled independently.

## Inspect or teach a reference

1. Save the project, select SimpleSurfaceEnhance and open **Set up lighting and
   reference**. Existing source images, lighting, source mask and results restore.
2. For a new setup, import an ordered capture manifest or copy a connected
   illumination bundle. A known lighting rig and linear, registered source stack
   are required. The tool does not infer light directions or camera gamma.
3. Inspect each retained source light. Review ID, unit X/Y/Z direction, relative
   power and exposure seconds in the lighting table. Untouched values retain
   full precision. Changing values requires a new reference preview.
4. Import an exact source mask, draw a rectangle, or move/resize its handles.
   Scroll/buttons zoom; Pan moves the view. Imported nonrectangular masks remain
   exact unless replaced. ROI contours show which source area is inspected.
5. **Preview reference fit**. Review normals, albedo, virtual light, fit residual,
   availability and detail tabs. Health reports valid/active coverage, rejected
   observations, no-redundancy pixels and reason counts; coverage is not confidence.
6. **Preview another capture** or **Preview connected capture** to test the frozen
   draft. Preview inputs never replace training. Stale edited/publication results
   are discarded and workers cancel on close.
7. **Stage recipe**, then **Apply configuration** in the host. Stage creates a new
   sealed `.sse` asset and changes only the inspector draft. Cancel leaves the
   applied recipe unchanged. Save As preserves sealed training bytes.

The editor requires a valid reference preview before Stage. It retains every
normalized reference image, not just a screenshot or trained coefficients.
No source file is moved, overwritten or deleted by training.

## Connect another application

The `capture` connector uses the host's array-capable **any** port, semantic
type `surface_illumination_stack`; `maps` uses `surface_maps`. The full schema
and examples are in [CAPTURE_CONTRACT.md](CAPTURE_CONTRACT.md). Arrays remain
numerical rather than becoming large JSON lists. Use **connected capture**
for actual pipeline values; **capture manifest** explicitly replays offline
files. No runtime source silently falls back to the training images.

Declared exposure seconds may change while compiled lighting geometry is reused.
Light order/directions/power/frame, source grid/dimensions and black/ceiling/linear
response must match. Runtime masks intersect the retained inspection region.
Grid/capture/editor-copy IDs are provenance, not proof of registration/synchrony.
Fixed camera/object pose through the stack is the supported scope.

## Review without loading a plugin

The original `studies/<case>/project.json` files load eight retained RGB views
with ordinary file sources. No estimation runs and file-source PASS means
loading only. These remain useful for quick map/coverage review.

`run_capture.py` explicitly selects a wheel for standalone CLI replay without
installation. `verify_package.py` checks retained numerical map parity;
`verify_execution.py` reopens/runs the actual plugin projects and checks judgments,
map parity, graphics-off behavior and compiled-estimator reuse. All result
exports/receipts require a new destination.

See [validation](E2_VALIDATION.md) for the actual checks and
[future work](FUTURE_WORK.md) for acquired-camera/material/calibration gaps.
Original code uses [MIT](LICENSE); upstream assets/dependencies retain their terms.
