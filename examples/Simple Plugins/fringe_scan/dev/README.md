# SimpleFringeScan setup and execution

These examples run the estimator through one **SimpleFringeScan** block. Its
native setup editor retains all reference images, sequence, known calibration
and inspection mask. Reopening restores the source and fit layers. Metric
results use colored **fly-around cloud thumbnails**; wrapped phase has image
results and no XYZ cloud.

The new [completed-runtime reviews](completed_runtime/README.md) provide an
independent Results action for exact consumed patterns, executed recipe/calibration,
raw phase/coding/rejection/XYZ maps and native colored fly-around clouds. Source/
pixel/point selection and frozen exports never decode or triangulate again.
Eight projects and six exports include coding conflict, ray-gap rejection,
missing input, graphics-off and the attributed external phase example. Load that
folder's pinned `plugin.py` first in a fresh process for the new workflow.

## Open and run

Explicitly load the bundled `plugin.py` using the host's plugin loader, then
open an execution project below. No environment installation, network access,
camera or projector activation is needed. Loading trusted plugins runs Python.

| Execution project | Accepted samples | Expected availability judgment |
| --- | --- | --- |
| [Analytic clean](execution/analytic_clean.json) | 6,912/6,912 metric points | PASS |
| [Analytic missing](execution/analytic_missing.json) | 6,415/6,912 metric points | FAIL, 92.81% below 95% |
| [External crown phase](execution/warsaw_crown_phase.json) | 56,985/559,872 phase pixels | FAIL, 10.18% below 95%; no metric cloud |

All three use the same generic fit settings and **95% coverage** gate. The two
analytic captures share one retained clean training recipe. A coverage failure
keeps its maps/cloud available for diagnosis. This judgment is availability,
not accuracy or defect acceptance. Open the block's result list to compare
source-aligned phase, amplitude, fit, decoded/final quality and calibrated depth.
Accepted cloud colors run green toward orange as ray gap approaches its limit;
rejected pixels remain gaps. Phase-only has six review images; metric has eight.

## Teach a new setup

1. Save a project and add SimpleFringeScan. Open **Set up sequence and
   calibration**. Import an ordered capture manifest, a retained `.sfs`, or
   capture a private snapshot from the connected stack.
2. Select a reference frame, zoom/pan and draw or adjust the inspection rectangle.
   Imported exact masks remain intact until replaced by a rectangle.
3. Review sequence identity, exposure and fit limits. Review/import known camera
   and projector intrinsics, distortion and rigid pose in meters. Calibration
   estimation from images is future work. Phase-only remains explicitly nonmetric.
4. Preview the reference. Inspect availability/reasons, fit/depth layers and the
   native cloud. A fresh preview with accepted evidence is required before Stage.
   Edits clear stale results. Independent manifest/connected previews never
   replace the saved reference or its health assessment.
5. **Stage recipe** creates a new sealed source/recipe asset and updates the
   inspector draft. **Apply** commits it. Cancel preserves applied training.
   Reopen to restore the retained images, calibration and mask.
6. Choose **connected capture** or **capture manifest** explicitly for execution.
   A connected source must match the taught order, grid, exposure and calibration.
   Runtime never silently falls back to the training images.

`image` is unpainted wrapped phase in radians. `maps` retains full numeric reasons
and metric evidence where calibrated; `report` describes availability/identity.
`cloud` is native accepted XYZ in camera meters with source pixel indices.
Graphics-off execution removes review rasters/colors while preserving maps,
accepted points, report and judgment. Unchanged recipes reuse the compiled phase
basis and calibrated camera rays; each capture is decoded independently.

## Data, scope and verification

The clean/missing controls are analytic rendered tilted planes. The external
crown frames are **simulated PBRT images**, not acquired-camera evidence.
Credits: **Michał Własiuk**, **Robert Sitnik**, Warsaw University of Technology,
*Simulated structured light 3D scanner datasets*, v1.0.0,
[Zenodo DOI](https://doi.org/10.5281/zenodo.17826191), **CC BY 4.0**. Exact original
members, hashes and transformations are retained in [DATASET_CREDITS.md](DATASET_CREDITS.md).
No metric calibration was imported for the crown. Its low coverage remains
visible; generic settings were not tuned to make it pass.

[E1 file-only reviews](MAP_REVIEWS.md) remain separate from execution. The new
`verify_execution.py` explicitly loads the plugin, reopens/runs all three
execution graphs and checks raw-map parity, judgments, caches and native clouds.
`verify_package.py` replays the CLI from the selected wheel. All numerical
results are software evidence; acquired-camera accuracy remains unqualified.

From a published examples folder, using the host's Python:

```powershell
python verify_execution.py . --plugin plugin.py --receipt new-execution-checks.json
```

See [capture/recipe contract](CAPTURE_CONTRACT.md),
[validation](E2_VALIDATION.md), [remaining work](FUTURE_WORK.md) and
[code license](LICENSE). Keep runtime manifest/image folders intact when
moving a project; retained training assets travel with host Save As.
