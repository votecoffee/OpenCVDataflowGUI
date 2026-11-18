# Completed reflected-pattern reviews

Nine saved projects and six frozen exports bind actual inspection patterns to
the executed reference recipe, radiometry, phase/change maps and judgment.
All use unchanged generic phase settings, **95% comparability** and an illustrative
**0.45-radian phase-vector limit**. These are phase criteria, not physical defect
or dimensional acceptance limits.

## Install

Use OpenCVDataflowGUI with independent inspector Results and paired snapshots,
including empty-input manifest support (da6de1a4 or later). In a fresh process,
explicitly load this folder's `plugin.py`, then open a project in `projects/`.
The loader pins both adjacent dev3 wheels; no environment installation or hardware
activation is required. Load this plugin before earlier Deflectometry loaders.

## Usage

Run in Debug, select **deflect**, and open **Review completed patterns and phase
changes…**. Select an observed X/Y pattern, actual source or executed reference,
stored map and map component. Shift-click source/map or enter X/Y for exact pixel
evidence. Wheel zoom, drag pan and Focus pixel support fine review. Cyan marks
the reference ROI used as the comparability denominator.

The table/chart shows reference and inspection codes for every observed frame,
ceiling/nonfinite states and the stored circular X/Y change vector. Phase maps
use radians; the vector chart has fixed -pi..pi axes and the configured limit.
Comparison and individual-axis phase reasons remain separate. No per-frame
prediction/residual is invented. New publications only mark the snapshot stale;
**Capture latest completed run** explicitly replaces it.

| Project | Evidence | Comparable / configured judgment |
| --- | --- | --- |
| analytic_clean | Unchanged independent sinusoidal control | 6,912 / PASS |
| analytic_change | Localized X/Y change; 293 pixels exceed the same limit | 6,912 / FAIL |
| analytic_missing | Nonfinite/clipped patterns | 6,212 (89.87%) / FAIL |
| connected_change | Changed control through the native connected-stack port | 6,912 / FAIL |
| whole_period_shift | Independently generated 2pi shift is indistinguishable | 6,912 / PASS; blind to 2pi |
| test_mask_restricted | Runtime mask reduces coverage against the reference ROI | 5,912 (85.53%) / FAIL |
| missing | Zero observations; changes unavailable | 0 / FAIL |
| graphics_off | Missing-control maps/judgment without emitted graphics | 6,212 / FAIL |
| sofast_self_review | Real code self-comparison; unknown exposure/response | 114,356/114,359 / PASS; self-case |

**Whole-period ambiguity remains.** Phase-only processing has no absolute screen
coordinate, measured slope, height or 3D cloud. The SOFAST self-case has zero change
by construction; it establishes import/fit/replay behavior, not held-out detection
or physical accuracy. It uses explicit `uncalibrated_codes` diagnostics.

`snapshots/` retains original executed recipe/manifest, native inspection and
reference arrays/masks, every map and emitted raster, JSON evidence and readable
source/reference/map/chart/assessment PNGs. Colors/ranges are display only. See
[completed review](COMPLETED_REVIEW.md). Model and recorded-stack `.sdef` assets are
sealed. Manifest sources remain ordinary file references; Save As copies the model
and rebases those references, rather than sealing a camera acquisition. Keep this
capture folder intact. Frozen exports remain independent of it.

## Verify

In the host Python environment, from this folder:

```powershell
python examples.py verify . new-checks --native
```

Verification checks every file pin, numerical maps/emitted rasters, native
inspector review without analysis, and exact-model Save As/reopen. All 66 package
checks pass, including source-file deletion, disabled limits, unknown radiometry,
float64/label-mask retention, worker handoff, stale/active/error/Lean and independent
teaching. `wheel-verification.json` pins source members and original MIT licenses.
Timing comparisons use identical recipes/data/settings with the shared dependency
held fixed; they do not establish a universal latency or speedup.

## Attribution and limits

Courtesy **OpenCSP**. **OpenCSP Team**, Sandia National Laboratories NSTTF,
[*OpenCSP: An Environment for Collaborative CSP Optical Technology Development*](https://opencsp.sandia.gov/).
SOFAST Landscape originals and derived data/results retain
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
No endorsement is implied. The exact official directory-tree license, original
camera/command/mask images and README are retained in `sources/originals/`.
Provenance, original hashes and code-value/mask reduction are documented in
[SOURCES.md](SOURCES.md) and `sources/provenance.json`.

The retained official 145-byte license matches its publisher-page SHA1. No new
camera capture, HDF5 processing, upstream script execution or image download is
needed for these reviews. The supplied mask is an inspection region, not defect
ground truth. Unknown screen pitch, response, exposure and physical truth are
not inferred from the pattern illustrations.

## Maintainers

votecoffee.

## Contributing

Preserve exact source data, unknown states, fixed settings and honest scope.

## License

Original code/analytic controls: [MIT](LICENSE). SOFAST retained/transformed data,
recipes and derived maps/images: CC BY-SA 4.0 with the attribution above.
