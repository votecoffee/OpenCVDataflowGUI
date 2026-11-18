# Completed laser-profile review

Run **SimpleLaserProfile** in **Debug**, then use the inspector's
**Review completed laser profile…** action. The independent, read-only popup uses
one paired finished input/output snapshot, the original `.slp` bytes and the
recorded stripe/validity/profile evidence. It does not execute extraction, pixel
undistortion, ray/plane intersection, calibration or current-file loading.

The source overlay, trend and sample table select the same original source bin.
Shift-click the source or click the trend, choose a table row or use the bin control.
Filters show accepted, unavailable/rejected or competing samples without changing
their decisions. Missing/rejected pixels and XYZ remain unavailable. ROI-excluded
bins stay distinct from active rejected bins. The selected-bin summary includes
reasons, candidates, contrast, support width and available incidence/XYZ.

Wheel zoom and drag pan work on the source, exact-input preview, executed training
reference and trend. **Focus source sample** zooms around a recorded source location;
selection preserves the zoom. **Overview** clears selection and restores whole-frame
views. Display contrast, colors and selection markers do not change native data.
The original dtype/encoding and consumed gray/RGB encoding are retained separately.

When valid calibrated points exist, the **Native calibrated cloud** tab uses the
host's native fly-around viewer. Overview keeps emitted colors, including the host
default when graphics were disabled. Review layers use green for the profile and
yellow for a selected accepted sample. Optional blue origin-to-sample segments, purple configured
laser plane and RGB camera axes are derived display geometry. They are not additional
sensor returns, new intersections or measured hidden surfaces. An invalid selection
cannot create a metric point or a ray segment. Pixel-only and empty profiles have
no native cloud. Unstored undistorted rays and reprojection errors remain unavailable.

The default overview reproduces emitted overlay/chart/cloud values. When execution
graphics were disabled, the emitted rasters remain absent and a calibrated cloud
keeps its original numeric points. Opening review may draw display-only images from
the frozen source and decisions. It does not force any extraction or diagnostic work
onto the execution path. Full array freezing, rendering and export occur on request;
execution retains a binding, serialized numeric report, existing result references
and the recipe bytes already read on a cold load. Lean retains no review record.

**Export frozen review…** creates a new folder containing the original `.slp`, exact
source/consumed/emitted NPY images, all stored stripe/profile NPY arrays, complete
sample/candidate/calibration JSON, source/overlay/trend/assessment PNGs and optional
native/selected PLYs plus a labeled cloud projection. NPY preserves dtype, precision
and unavailable NaNs; PLY/display follows the host's floating-point representation.
PNG contrast/projection is for assessment only. Existing folders are not overwritten.
Exports work after the original model file is deleted or the graph closes.

New editor output marks an existing review stale. Only explicit **Capture latest
completed run** replaces it. Active, errored, unexecuted or Lean executions cannot
supply a new capture. A failed recapture retains the previous frozen view with an
explanation. Unapplied training/calibration drafts and Apply/Cancel stay independent.

Coverage and pass/fail concern sample availability, not stripe identity, confidence,
dimensional acceptance or gauge accuracy. A declared calibration/ticket is software
evidence; it does not prove physical calibration or acquisition synchronization.
See [completed examples](README.md) for licensed real
pixel-only data and generated known-calibration controls with unchanged settings.
