# Completed surface-map review

Run **SimpleSurfaceEnhance** in **Debug**, then use **Review completed surface maps…**
in the inspector. The independent, read-only Results popup binds one finished run's
actual consumed capture, executed `.sse`, effective lighting/region declarations,
prepared observations and all stored numerical maps. It does not reload source or
model files, normalize again, estimate normals/albedo, relight or recalibrate.

Choose an inspection source light or the executed training reference on the left,
and a completed map on the right. These are distinct sources; reference selection
does not replace inspection input. The reference caption shows its retained exposure,
while the light selector/table shows the actual capture exposure. Source views draw
the effective inspection ROI in cyan. All source-grid maps retain original X/Y.
The combined overview has separate display coordinates; select pixels on the source.

Shift-click either source-grid image, or enable **Inspect original pixel** and enter
X/Y. Source/map selection and the per-light table refer to the same recorded pixel.
Wheel zoom and drag pan are available; selection preserves zoom. **Focus pixel**
zooms both source and map, while **Overview** clears selection and restores full views.

Inspect raw normalized source codes, exposure/power-normalized prepared values,
availability flags, fit usage, signed stored residuals and the original fit reason.
The observation chart draws only available values. A plotted prediction is derived
as prepared observation minus stored residual, only for an accepted fit. It is
display evidence, not a new fit. Nonfinite/dark/input-ceiling/outside-ROI flags can
coexist; the table and native arrays retain exact combined bits. Reason-map classes
describe fit availability/conditioning/consensus, not material or defect labels.

Normal RGB encodes `(NXYZ+1)/2`; purple denotes unavailable values. Normal directions
are estimated in the declared lighting frame, not measured XYZ or metric height.
Albedo/radiometry are relative; residuals are model disagreement, not physical
measurement uncertainty. Scalar color ranges/percentiles are display-only. Counts,
singular ratios, fit RMS and per-light residual/usage/flag maps remain native arrays.

The exact emitted overview is available when runtime graphics were enabled. With
graphics off, emitted views remain absent; review may color the existing maps and
compose display-only panels. Stored relighting/detail/residual arrays are reused;
no relighting, gradient calculation or solver call occurs during review. Debug holds
references to the prepared arrays already calculated during execution, without an
extra preparation/copy pass. Full freezing, hashing and display/export occur on
explicit capture. Lean retains neither the review record nor extra prepared values.

**Export frozen review…** creates a new folder containing the byte-identical executed
recipe, original consumed manifest bytes when applicable, normalized native capture
images/mask/array-bearing lighting declarations, all prepared/maps/emitted-view NPY
arrays, complete JSON evidence and source/reference/map/overview/assessment PNGs.
`capture_data.json` maps array declarations to their NPY files and records dtype,
shape and SHA-256. Native float precision, mask labels and unavailable NaNs are
preserved. PNGs show display contrast/markers rather than replacing native data.
Original encoded camera-image bytes are not retained by this interface; the actual
normalized arrays consumed by the algorithm are retained. Extra top-level capture
fields outside the contract are listed as ignored. Existing export folders are
not overwritten; exports work after original files disappear or the graph closes.

Manifest mode requires a host that supports paired snapshots with an explicit empty
input selection: it legitimately consumes no wired inputs. The actual loaded capture
is retained privately from the finished execution, never substituted with training.
New publications mark an open review stale; recapture explicitly. Active, errored,
unexecuted and Lean runs cannot replace it. Failed recapture leaves the previous
frozen run visible with an explanation. Unapplied setup/region/calibration drafts
and Stage/Apply/Cancel remain independent.

Coverage is availability, not accuracy or confidence. Caller capture/grid/lighting
declarations do not establish physical registration, synchronization or calibration.
See [completed examples](README.md) for explicitly rendered
analytic/material controls and retained acquired-data/lighting-calibration limits.
