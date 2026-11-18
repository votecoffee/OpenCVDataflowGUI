# Completed fringe scan reviews

Eight saved projects and six frozen reviews explain the actual pattern source,
coding/phase diagnostics and accepted calibrated geometry of one completed run.
All use unchanged generic settings and the same **95% availability** gate.
These are software controls, not acquired-camera or physical accuracy evidence.

## Install

Use OpenCVDataflowGUI with the independent inspector Results action and paired
snapshot SDK, including empty-input manifest support (da6de1a4 or later). In a
fresh app process explicitly load this folder's `plugin.py`, then open a project
in `projects/`. The loader pins its adjacent dev3 wheel; no pip installation or
environment changes are required. Load this plugin before other FringeScan loaders.

## Usage

Run in Debug, select **fringe**, and open **Review completed coding, phase and
cloud…**. Choose an observed pattern and stored map/component. Shift-click source
or map, or use X/Y, to inspect the exact source pixel and all recorded codes.
Zoom/pan and Focus pixel support fine review. Reference/source switching never
substitutes teaching images for inspection. New publications only mark the frozen
review stale; **Capture latest completed run** explicitly replaces it.

Calibrated results use native **colored fly-around** cloud thumbnails and review.
The separate native point-selection tab links accepted rows back to camera pixels.
Yellow selection and optional camera/projector sightlines are display overlays;
rejected pixels stay unavailable and never become points. The external crown is
phase-only and intentionally has no metric cloud.

| Project | Purpose | Available / judgment |
| --- | --- | --- |
| analytic_clean | Independently constructed first-surface tilted plane | 100% / PASS |
| analytic_missing | Nonfinite, clipped, unilluminated and weak-modulation regions | 92.81% / FAIL |
| gray_phase_conflict | X phase shifted 2px while Gray codes stay fixed | 78.73% / FAIL |
| geometry_rejected | Coherent Y codes/phase shifted 2px; inconsistent calibrated rays | 78.73% / FAIL |
| connected_missing | Missing control through the native connected stack port | 92.81% / FAIL |
| missing | Zero observations; no replacement cloud | 0% / FAIL |
| graphics_off | Missing-control maps/XYZ without emitted graphics/colors | 92.81% / FAIL |
| warsaw_crown_phase | Attributed external simulated six-step scan | 10.18% / FAIL; phase only |

`snapshots/` retains exact executed recipe/manifest, native consumed arrays,
all maps, emitted rasters and optional exact native cloud data. Source/reference,
selected-map, observation-chart, overview and assessment PNGs help inspect them.
PNG ranges and projections are display only. See [review workflow](COMPLETED_REVIEW.md).

Model/capture `.sfs` assets are sealed. Manifest sources and their image paths
are ordinary file references; Save As carries the model and rebases those
references, rather than sealing a camera acquisition. Keep this capture folder
intact. Frozen exports remain independent of those files.

## Verify

From this folder in the host Python environment:

```powershell
python examples.py verify . new-checks --native
```

Verification checks all published file pins, exact source/maps/clouds/rasters,
native inspector browsing without analysis, and exact-model Save As/reopen.
The numerical package's focused suite has 71 passing checks, including file
deletion, missing geometry, worker handoff, stale/error/Lean and teaching isolation.
`wheel-verification.json` pins the exact source members and license. Timing
experiments compare the same saved recipes with dev2; no universal speed claim.

## Attribution and limits

Michał Własiuk and Robert Sitnik, Warsaw University of Technology,
[*Simulated structured light 3D scanner datasets*](https://doi.org/10.5281/zenodo.17826191),
version 1.0.0, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The unchanged six crown PNGs, projector commands and scene declarations are in
`sources/`. The explicit green-channel, inferred sRGB inverse, 3×3 averaging,
ceiling propagation and phase-order transformation is in `sources/transform.json`.
This remains simulated phase evidence; no source calibration or shape truth is
imported. No endorsement is implied. See [full attribution](SOURCES.md).

Original analytic controls/code use MIT. Low sinusoidal/ray residuals, caller
identity and software parity do
not establish absolute correspondence correctness or physical gauge accuracy.

## Maintainers

votecoffee.

## Contributing

Preserve unknown states, fixed settings, native source precision and honest scope.

## License

Original code/controls: [MIT](LICENSE). External data: CC BY 4.0 as credited above.
