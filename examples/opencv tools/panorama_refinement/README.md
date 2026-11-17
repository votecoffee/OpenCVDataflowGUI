# Panorama camera refinement and reserved validation

Open a project with **File → Open…**, then Run. All use the explicit native
camera-fitting path in the rotating-panorama block. Select that block and inspect
Report → `camera_fit` for fit/reserved counts, native matching evidence and
initial/fitted intrinsics. Report → Pairs checks only reserved observations.
The normal result image overlays source boundaries and mapped source corners.

| Project | Expected result | Demonstrates |
| --- | --- | --- |
| [reprojection.json](reprojection.json) | Pass | Estimate focal from fit homographies, then refine focal and rotations. |
| [ray.json](ray.json) | Pass | Native ray bundle adjustment with centered square work intrinsics. |
| [fixed_intrinsics.json](fixed_intrinsics.json) | Pass | Keep known K fixed, refine rotations and render original uint16 pixels onto a plane. |
| [initial_only.json](initial_only.json) | Pass | Known K and native initial rotation estimates without bundle adjustment. |
| [refine_prior.json](refine_prior.json) | Pass | Start from 390-pixel focal estimates and refine toward the known 420-pixel focal. |
| [validation_rejected.json](validation_rejected.json) | Fail, no panorama | Wrong fixed 200-pixel focal fails reserved camera-model checks. |
| [disconnected.json](disconnected.json) | Fail, no panorama | An unrelated view is outside the anchor's connected component. |
| [wave_corrected.json](wave_corrected.json) | Pass | Fixed known intrinsics, refined rotations and horizontal wave correction. |
| [insufficient_validation.json](insufficient_validation.json) | Fail, no panorama | Require more reserved matches than available; fit observations are never substituted. |

In Debug, failed fits still provide the separate connectivity chart to the
filmstrip. Lean omits optional charts and drawings; the same numerical judgments
apply. The nine projects have six passing and three intentionally failed cases.

## Controlled reference

These projects reuse the original MIT captures and independent truth from
[rotating_panorama](../rotating_panorama/README.md). There are no copied or new
third-party images. Known **480×360** sources have focal **420 pixels**, centered
principal points and yaw **−36°, 0°, +36°** from one optical center. The first two
inputs have no direct overlap; the center view bridges them.

Tests compare fitted cameras to known focal/rotation and independently projected
world landmarks: **10%** focal error, **3°** relative rotation error and **5-pixel**
landmark RMS limits. Fixed-K cases retain original-source K within **1e-9**,
including a reduced work-grid check. uint16 source samples remain above 255.
Native matching, initial estimation and bundle adjustment are instrumented to
verify they never receive reserved keypoint centers. Validation failure is
separate from an unexpected native exception; recovery and preflight controls
are checked. Save/reopen, XML, offscreen controls and graphics-disabled runs are
covered without opening a desktop window.

The rounded-center partition uses seed 104 and reserves about 35% of keypoints
before native matching. Validation requires 12 inliers, a 0.7 inlier fraction
and at most 3-source-pixel camera-model RMS. The test data has abundant texture;
the fraction is not a guarantee of sufficient paired reserved observations.
Changing solver settings does not search seeds or re-fit using validation data.

These captures provide controlled panorama geometry, not general physical camera
calibration. Shared-image features are correlated. Particularly with yaw-only
views, freeing aspect and principal point can be underconstrained even when
overlap residuals are small. The examples keep them fixed. See
[solver and validation contracts](../../../docs/rotating-panorama.md).

Run or regenerate from the repository environment:

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/panorama_refinement/reprojection.json" --profile debug --summary
python "examples/opencv tools/panorama_refinement/build_example.py"
```

Generation loads the shared source projects without executing them and saves
rebased paths to the same source assets. Each project contains an attribution
comment. Images, generators and projects are original OpenCVDataflowGUI
contributor work under the repository's [MIT license](../../../LICENSE).
