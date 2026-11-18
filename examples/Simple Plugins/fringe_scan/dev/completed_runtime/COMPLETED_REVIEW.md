# Completed coding, phase and cloud review

Run a configured SimpleFringeScan block in Debug, then choose **Review completed
coding, phase and cloud…** in the inspector. This opens a read-only review independent
of sequence/calibration teaching. An unexecuted, active, exception or unretained
Lean result cannot supply it. A failed availability inspection can.

The popup freezes the actual consumed native pattern stack and its caller-declared
IDs, order, source grid, common exposure, response and calibration; the exact
executed `.sfs` bytes; effective decode/phase/geometry settings; full report;
and all emitted numerical maps/cloud rows. It never falls back to training images
or current upstream inputs. File-driven runs use an explicit empty paired-input
selection; the host needs the corresponding SDK fix (da6de1a4 or later).

Choose an observed frame and switch between actual source and executed training
reference. Shift-click a source-grid image or use X/Y to inspect one original
camera pixel. Wheel zoom, drag pan, Focus pixel and Overview help review fit.
Vector maps have X/U, Y/V and Z component controls. Overview panels use display
coordinates and cannot be picked as camera pixels. Cyan marks the effective ROI;
the yellow cross is the selected pixel. Map values keep their original types,
NaNs and units; color scales and ranges only affect display.

The selected-pixel table/chart shows every recorded normalized linear pattern
code, phase shift where declared, ceiling and nonfinite states. Wrapped phase,
modulation, Gray diagnostics, phase disagreement and relative fit errors are
stored evidence. Diagnostic coarse codes can exist at rejected pixels; only
valid correspondence can lead to accepted geometry. Per-frame fitted predictions
and residuals are not stored and remain unavailable. Review never fits new ones.

Calibrated results use the host's native **colored fly-around** viewer. A separate
native point-selection view provides orbit/zoom/pan and links an accepted point
back to its exact source pixel. The selected row is yellow with a derived cross.
Optional rig origins/baseline/sightlines derive only from the declared rigid rig
and stored accepted XYZ; they are overlays, never extra source rows. Missing,
rejected, phase-only or zero-valid results never create replacement geometry.
Cloud output is float32 per the host contract; raw XYZ maps retain full precision.

New editor publications mark the snapshot stale without replacing it. **Capture
latest completed run** explicitly replaces it only after a successful paired
capture; an unavailable attempt leaves the earlier frozen run visible. Pending
teaching, calibration edits and Stage/Apply/Cancel remain independent. Closing
the popup releases arrays, pixmaps, point-projection state and fly-around timers.

**Export frozen review…** creates a new folder and refuses overwrite. It writes
byte-identical executed recipe and consumed manifest (when used); native array
files plus dtype/shape/hash declarations in `capture_data.json`; every raw map
and emitted raster; source/reference/selected-map/overview/observation/assessment
PNGs; and exact emitted native cloud rows/colors/source pixels with a PLY when
available. `review.json` retains settings, identities, reasons and scope. PNGs
and cloud projection are display artifacts; the native arrays are the evidence.
Original encoded source-image bytes are not retained, only the actual normalized
consumed arrays. Extra unknown capture fields are listed as ignored.

Capture, browsing and export do not reopen files, decode, unwrap, triangulate,
calibrate, verify new correspondence, change acceptance or edit training. Debug
execution retains references and compact correlation metadata; deep array copies
occur only on explicit review capture. Lean adds no review record. Graphics-off
keeps raw maps/XYZ and omits emitted rasters/colors; review may color stored maps
for display without reconstructing geometry.

Caller IDs/tickets and low fit/ray residuals do not prove synchronization,
registration, absolute coding correctness or physical accuracy. Known-calibration
XYZ is a reconstruction, not an independent measurement of ground truth. Acquired
sequence/calibration/gauge qualification remains a separate roadmap item.
