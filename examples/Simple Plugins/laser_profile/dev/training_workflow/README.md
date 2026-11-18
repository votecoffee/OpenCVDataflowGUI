# LaserProfile calibration and training workflows

Load **plugin.py first**, then open a project below. The included checksum-pinned
dev4 wheel loads without installing or replacing the host package. Restart if
another LaserProfile version is already imported. Use a current host with native
image/cloud viewers, connected input snapshots and independent Results review.

| Project | Retained outcome |
| --- | --- |
| [Calibrated step](../../training/calibrated_step.json) | Generated known geometry: 480/480 bins and a nominal 50 mm step. |
| [Unavailable bins](../../training/calibrated_unavailable.json) | Generated missing/competing/input-ceiling samples: 390/480, failure at the unchanged 90% gate. |
| [Real WLI](../../training/real_wli1.json) | Unmodified licensed laser photograph: 72/120, pixel-only failure remains. |

Finished staged versions are also included:
[calibrated step](../../training/calibrated_step_trained.json),
[unavailable bins](../../training/calibrated_unavailable_trained.json), and
[real WLI](../../training/real_wli1_trained.json). These retain the newly taught model
and calibration evidence in exact checksummed archives.

Open **Train laser profile…** in Configuration. The saved reference/ROI,
background, signal, frozen stripe settings and calibration reopen immediately.
Persistent guidance explains whether learning is missing/edited, running,
unavailable or ready. The physical summary labels meters, camera axes/frame,
source dimensions, intrinsics/distortion, plane equation and axial range.
Pixel-only mode explicitly has no metric points or native cloud.

**Review retained calibration…** opens read-only coefficients and raw evidence.
Enter an original pixel X/Y and **Check pixel ray** to inspect declared geometric
intersection or an unavailable reason. This does not detect a stripe or establish
accuracy. Opening/closing review changes no training.

For the generated metric cases, **Fit plane from observations…** can load
[observations.json](observations.json). It supplies the known camera, two training
target planes at 0.4/0.6 m and two separate validation targets at 0.5/0.55 m.
The existing all-point least-squares kernel fits only training observations.
The table and raw record distinguish training residuals/conditioning from
validation target agreement, counts and invalid-ray reasons. Both use the same
declared camera; observation IDs do not prove independent physical acquisitions.
There is no automatic acceptance threshold or target-pose/board detector.

**Use calibration in training draft** keeps the reviewed fit privately and
invalidates prior learning; **Close / keep draft** preserves the old calibration.
Learn the reference again, then preview an independent file or connected image.
Taught-reference health remains separate from the last preview's availability.
Changing ROI, signal, background or calibration clears old results and closes
stale child snapshots. Original input/recipe bytes remain intact.

The controls/content scroll on smaller screens while Learn, independent preview,
Stage and Cancel remain in a fixed footer. **Stage recipe** updates only the
inspector draft; inspector **Apply** activates it. Cancel preserves the active
model. Reopen and Save As retain reference pixels, mask, settings, calibration
coefficients, observation roles and fit/validation summaries exactly.

Run in Debug and open **Review completed laser profile…** on Results for frozen
source/stripe/trend/bin evidence. Calibrated profile output uses the native colored
fly-around cloud. The real WLI case has no supplied camera/laser calibration and
cannot acquire metric output through review. [Saved PNG/PLY evidence](evidence)
contains diagnostic renders, not desktop screenshots. Rejected/missing bins stay
unavailable; coverage and imported calibration do not establish gauge accuracy.

Run `python check_training.py --output C:\path\to\new-output` using the host
Python. The output must be new. The actual offscreen inspector checks restore,
read-only review/probe, fit Cancel/Use, reversible ROI edits, unchanged-reference
learning, independent preview, native metric cloud, Stage without Apply,
Apply/reopen/Cancel, completed Results and exact recipe Save As. All three staged
numeric reports currently match their original outcomes exactly; the checker
also bounds possible calibration roundoff to 1e-10 in declared numeric units.
Nine earlier cases retain exact reports/images/clouds and six exports verify.
See [qualification](qualification.json) and [source attribution](SOURCES.md).

Image-derived target/stripe collection, camera/target-pose estimation, graphical
observation editing, calibration-image associations, draft Undo/history,
background restoration and finer cancellation remain future work. Real WLI
extraction failures, hands-on acceptance and independent physical gauge checks
remain open.
