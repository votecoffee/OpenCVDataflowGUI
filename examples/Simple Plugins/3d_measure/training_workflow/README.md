# Simple3DMeasure training workflow

Load **plugin.py first**, then open a project below. It selects the included,
checksum-pinned dev7 wheel without installing anything. Restart if another
measurement version is already imported. Use a current host with native cloud
teaching, paired completed snapshots and independent Results review.

| Project | What to inspect |
| --- | --- |
| [Real capture 13](projects/real13.json) | Retained frame-3 reference, two trained plane volumes, signed datum axes, learned physical fit bands; separate first-return capture. |
| [Real capture 24](projects/real24.json) | Same training and supplied annotation fixture, different measured capture. Nonplanar/noisy support remains visible. |
| [Missing surface](projects/missing.json) | Generated 5 mm step reference; independent scene lacks the target. Three measurements remain unavailable while the datum probe is available. |

1. Open **Train 3D measurements…**. The retained reference, named regions,
   fitting evidence, units/frame and measurement definitions appear immediately.
   Green/red are reference fit inliers/exclusions; cyan is an untrained selection.
2. Select a measurement row to highlight its datum and read its target/probe and
   limits. Real examples have no acceptance limits. Numerical reference values
   describe training support; they are not independent gauge truth.
3. Select a region, use **Edit XYZ bounds…**, and update its minimum/maximum
   coordinates. Units and measurement frame are explicit; no conversion occurs.
   Invalid volumes cannot apply. Cancel preserves the old region; untouched
   floating-point coordinates retain their full precision.
4. Undo/Redo uses the same definition history as native XYZ face handles. Any
   definition change requires **Train / preview reference** before staging,
   including after Undo. Training uses a frozen background job. The persistent
   guide explains the next step and unavailable reference values.
5. Orbit/zoom/pan, focus selected, change source appearance or inspect reference
   context. Regions are complete axis-aligned XYZ volumes, including points
   hidden in the current view. Source-row picking is distinct from derived axes.
6. **Stage recipe**, then inspector **Apply**. Stage alone does not activate it.
   Cancel from either editor keeps the active recipe. Reopen or Save As retains
   reference XYZ, regions, measurements, fit bands and settings.
7. Run in Debug, open **Review completed 3D measurements…** on Results and use
   its native fly-around view. Source/fixture/model/fit evidence is captured from
   one completed execution. Review and exports do not refit or retrain.

The main window follows the available screen size. Setup controls scroll while
Train/Review/Stage/Cancel stay in a fixed footer. At narrow widths, scroll the
outer content horizontally to reach the full native toolbar. A compact shared
host toolbar, oriented regions, more primitives, background reference restoration,
finer fit cancellation and hands-on operator acceptance remain follow-ups.

[Native PLY evidence](evidence) includes colored reference source rows and completed
runtime graphics. PNGs are documentary point renders, not desktop screenshots;
the interactive native cloud is the primary output. Derived region boxes, grids,
axes and dimensions are not measured source points.

The real cases are T-LESS data with supplied annotation fixtures, not detected
poses. They have no independent dimensional gauge truth. Generated controls
provide finite-point geometry and missing-support behavior, not sensor accuracy.
See [sources and license](SOURCES.md). All prior runtime algorithms/settings remain
unchanged; [qualification](qualification.json) distinguishes exact original-recipe
execution from retraining comparisons.

Run `python check_training.py --output C:\path\to\new-output` using the host's
Python. This exercises the actual inspector offscreen, without desktop input or
window capture. The output directory must be new. The three receipts cover
restore, numeric edit/Undo/Redo, background train, Stage/Apply/Cancel, native
Results and exact recipe copying through Save As. Native teaching remains a
definition editor; it does not infer a calibrated frame or validate the datum.
