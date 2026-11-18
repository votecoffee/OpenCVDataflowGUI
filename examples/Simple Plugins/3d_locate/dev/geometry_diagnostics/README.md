# Simple3DLocate supplied-pose geometry diagnostics

These six real T-LESS examples explain why supplied annotated poses fail
geometry verification. **They are not locator detections.** The saved graphs
load stored native RGB point clouds and execute no pattern search or ICP.

Open either project, run once, and double-click a cloud preview for native
orbit/flyaround review:

- [Object 6 / capture 3](object000006_scene000002_frame000003_isolated/project.json)
- [Object 8 / capture 261](object000008_scene000003_frame000261_isolated/project.json)

All six case folders include a project, baseline and all-triangle visibility
clouds, secondary photographic overlays, and a full seven-variant report.
Coordinates are in millimeters in the camera frame. RGB axes mark the supplied
model pose. Green means supported, red means beyond the distance tolerance,
orange means a rejected normal, blue-gray means hidden, magenta means unresolved,
and gray points are the measured scene. No missing scene surfaces are invented.

## Recorded findings

Five controls each pass 0/6 supplied poses: runtime preparation, full model
voxels before budget reduction, normals estimated before scene reduction,
six-neighbor normals, and a dense unreduced scene. All-triangle self-visibility
passes 1/6 with either the retained or full-voxel model. Model 8 / capture 261
changes from 41.2% to 57.1% support with the unchanged 50% requirement. The
other five cases still fail. These are geometry-feasibility results only;
the separate twelve-positive unseeded matching failures remain unresolved.

Four baseline cases have less than 50% distance-only support, so changing
normal estimation alone cannot make those pass. This study retains the same
physical thresholds and model settings throughout. These captures informed
development and are not independent production qualification.

[summary.json](summary.json) records pass counts; per-case reports contain
all gate values, limits, pose matrices and evidence counts. The source/member
and generated-artifact hashes are retained separately. [overview.jpg](overview.jpg)
is secondary photographic documentation; native clouds are the primary review.

## Rebuild

Requires Simple3DLocate **0.1.0.dev16** and this app's Python environment.
From this folder, run:

```powershell
python build_study.py ../novel_models/source ../geometry_diagnostics_rebuilt --host-projects
```

The builder verifies pinned source hashes before processing and writes to a
fresh folder to preserve existing evidence. Triangle-ray diagnostics are
bounded offline calculations and do not change runtime defaults,
matching latency, trained recipes or acceptance gates. Excess diagnostic work
raises an explicit error; unresolved intersections are never assumed visible.

## Attribution

T-LESS by Hodan, Haluza, Obdrzalek, Matas, Lourakis and Zabulis, WACV 2017,
**CC BY 4.0**. See [complete attribution and modifications](ATTRIBUTION.md).
Diagnostic code is MIT; the data license applies separately. No endorsement
is implied.
