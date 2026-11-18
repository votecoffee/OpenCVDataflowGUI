# Review completed 3D measurements

Version 0.1.0.dev6 adds **Review completed 3D measurements…** to the inspector's
Results page, independently of teaching and its draft. Use a current host with
paired completed snapshots and Debug input/output retention.

1. Run a saved measurement recipe, select the block and open Results review.
   The consumed typed cloud, optional fixture input, effective fixture, original
   recipe archive and recorded measurements/fits/residuals are frozen together.
2. Read every measurement's value, units, actual limits, status and reason.
   Unavailable values remain null and show **Unavailable**; they are never zero.
   Select a row to focus its region; paired measurements show both regions.
3. Switch between fit/datums/dimensions, recorded residuals and exact consumed
   cloud. Choose a fit region and toggle layers. Native RGB orbit/Flyaround,
   yaw/pitch and point-size controls use the host CloudViewer. The node's primary
   output remains its native fit-cloud thumbnail.
   **Focus measurement region** hides global scene/dimension context to enlarge
   the selected fit; **Overview** restores all fit layers. These are view changes.
4. In residual mode, select one region for its histogram, physical fit band and
   tail counts. Plane residuals are signed distances to the current fitted plane;
   line residuals are unsigned radial distances to the fitted line. The scale
   stays at three taught fit bands. Full values beyond the color range survive.
5. Export frozen review to a new folder. Close without changing source labels,
   teaching settings, staged training or inspector Apply/Cancel state.
6. Later publications mark the review frozen. Capture latest explicitly replaces
   it. Active/error/unretained capture refuses to mix runs and keeps the previous
   frame clearly labeled. Before any completion, export is disabled.

Gray measured points, green inliers, red exclusions and magenta unavailable
observations remain distinct from derived datum grids, ROI boxes, dimension
lines, axes and minimum-zone support geometry. Fit axes are expressed in the
measurement frame; display geometry uses the **recorded effective** fixture to
map into the scene frame. The raw fixture declaration is retained separately,
including any small rotation normalization accepted during execution. Source
RGB/normals/pixels/size/metadata are preserved on the exact-input view/export.

Residual arrays and fit selections already exist in completed execution evidence.
Review only summarizes/colors those values and transforms derived geometry; it
never selects new source points, fits a plane/line, reruns minimum-zone flatness,
measures, retrains or loads the current recipe. Histograms are cached on first
request. When execution graphics were off, the outputs stay absent and review
displays are explicitly derived on request from frozen fits. Empty/unavailable
fits have no invented residual arrays, summary values or histogram.

Execution retains a small Debug binding and references to existing evidence.
Original recipe bytes are retained from the loader's existing verified read.
No new source-cloud copy, fitting, residual analysis or model serialization is
added to the execution path. Explicit capture privately freezes arrays and copies
metadata; Lean skips the extra completion record. Software pairing does not
establish physically synchronized acquisition.

Exports contain `review.json`, original `executed_recipe.s3m`, binary input/emitted/
selected PLY clouds, exact NPY source attributes, double-precision measurement
scene/local points/effective fixture, per-region source row IDs/inlier masks/
available residuals, and full `all_residuals.npz`. PLY preserves XYZ/RGB/normals/
units/frame; native arrays and JSON retain source pixels/size/metadata. The
assessment PNG labels selected values, limits, status, frames and display angles.
Single-region residual export from the popup also includes its histogram PNG.
Existing folders are never overwritten. Save As preserves the sealed recipe
bytes/checksum and retained reference context.

Fit residuals describe the current fit, not deviation from the taught reference
or measurement uncertainty. A pass without configured limits means measurements
were available. Independent dimensional/gauge accuracy, hands-on acceptance,
cross-frame comparison and richer native picking/focus remain follow-ups.
