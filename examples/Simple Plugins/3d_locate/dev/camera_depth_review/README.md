# Simple3DLocate camera-depth evidence

These six real T-LESS projects load **native RGB flyaround clouds** showing
camera-ray evidence at supplied annotated poses. They do not run localization,
and no annotated pose counts as a detection. Colored points are expected CAD
surfaces; gray points are measured depth returns.

- [Object 6 / capture 3](object000006_scene000002_frame000003_isolated/project.json)
- [Object 6 / capture 255](object000006_scene000002_frame000255_isolated/project.json)
- [Object 6 / capture 494](object000006_scene000002_frame000494_isolated/project.json)
- [Object 8 / capture 5](object000008_scene000003_frame000005_isolated/project.json)
- [Object 8 / capture 261](object000008_scene000003_frame000261_isolated/project.json)
- [Object 8 / capture 493](object000008_scene000003_frame000493_isolated/project.json)

Run a project and double-click its cloud preview to orbit/fly around it.
Units are millimeters in the camera frame. Green is consistent ray depth,
red is a farther return, orange is a nearer return, magenta is missing exact
depth, blue-gray is grazing incidence, and RGB axes mark the supplied pose.
Nearer/farther do not establish a cause: sensor residuals, CAD or pose errors
and occlusion can produce them. Green alone does not establish normal agreement
or an accepted localization.

## Using the new trainer review

Install **Simple3DLocate 0.1.0.dev17** and reopen the application. Open an
existing locator project, then **Train 3D locator…**. Capture/load its matching
test scene, camera and optional full-frame verification depth. Run **Locate in
test scene**, select one instance or rejected candidate, and choose **Review
camera-depth evidence…**. Rejected candidates remain labeled as rejected.

The popup provides native flyaround, category/scene/pose layers, counts and
physical limits. Layer and orbit changes reuse cached maps without repeating
localization or depth rendering. Input/training changes close stale reviews.
Export creates a new folder containing an RGB cloud, full native NPZ maps,
camera, pose, counts, limits and input/model hashes. When available, use
**Capture last processed inputs** for a completed-run bundle; this does not
prove simultaneous sensor acquisition.

## Recorded findings

[summary.json](summary.json) retains exact counts and signed residual
percentiles. Object 8 captures 5 and 493 have 18.3% and 12.6% missing eligible
pixel rays. Fully observed object 6 / capture 255 has median residual -2.73 mm;
capture 3 has +2.32 mm, against the same 2.91 mm tolerance. These findings need
separate normal/correspondence and sensor/CAD investigation. The twelve-positive
novel-model matching gap remains unresolved. No default gate was loosened.

Each case has expected/measured optical-Z maps, signed normal-distance
residuals, state codes, camera calibration and the supplied pose. Missing depth
and unmeasurable residuals remain NaN. No neighboring pixel fills a hole.
The cloud uses category-balanced display sampling; raw maps retain every
rendered ray. Bounds match the existing renderer: at most 640 by 640 cells and
four million triangle/pixel work pairs. This review executes only on explicit
request and does not add work to ordinary localization.

## Rebuild and attribution

From this folder, use the app environment and a fresh output folder:

```powershell
python build_review.py ../novel_models/source ../camera_depth_review_rebuilt
```

Source hashes are verified before processing. Generated evidence hashes and
byte-preserving Git attributes accompany the results. T-LESS by Hodan et al.,
WACV 2017, and derived data retain **CC BY 4.0**. See [complete attribution and
modifications](ATTRIBUTION.md). Implementation code is MIT. No endorsement
is implied.
