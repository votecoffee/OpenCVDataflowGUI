# Completed bin-picking review

Load [plugin.py](plugin.py) explicitly in OpenCVDataflowGUI, open a project below,
run in **Debug**, then use the picker's inspector button
**Review completed bin-picking evidence…**. The bundled dev13 wheel is SHA-256
pinned and loaded without installing or updating the host environment. Restart the
app before loading this plugin if an older SimpleBinPicking version is loaded.

The independent Results window shows every recorded candidate, geometric score,
contact fraction, clearance decision and reasons. Select a candidate, use
Focus/Overview, toggle measured scene, CAD/tool, TCP, sweep and witness layers,
and use the native fly-around controls. Review captured input images with zoom/pan.
Export a frozen recipe/input/NPY/PLY/JSON/PNG assessment without rerunning planning,
contact tests, collision checks, depth projection or training.

| Project | Purpose |
| --- | --- |
| [Suction good](../../inspection/suction_good.json) | Eligible alternatives on a generated first-return surface. |
| [Suction holes](../../inspection/suction_holes.json) | Missing rays; conservative unknown/rejection evidence remains. |
| [Suction missing](../../inspection/suction_missing.json) | No observed points, eligible picks or stale TCP output. |
| [Suction real frame 13](../../inspection/suction_real13.json) | Real T-LESS RGB/depth, calibrated bin pose, unchanged zero-pick result. |
| [Jaw clear](../../inspection/jaw_clear.json) | One eligible axial grasp; another alternative stays unknown. |
| [Jaw obstacle](../../inspection/jaw_obstacle.json) | Observed obstacle intersects the configured palm sweep. |
| [Jaw unknown](../../inspection/jaw_unknown.json) | Missing rays under the palm remain unknown despite accepted part pose. |
| [Jaw real frame 3](../../inspection/jaw_real3.json) | Real T-LESS first returns and supplied object pose; current pose check rejects. |
| [Jaw real frame 13](../../inspection/jaw_real13.json) | A second real rejected pose-support case. |
| [Jaw retained mesh](../../inspection/jaw_mesh.json) | Exact retained sample/triangle first-surface diagnostics. |
| [Jaw graphics off](../../inspection/jaw_graphics_off.json) | Same numeric decisions; no emitted clouds or new fit diagnostics. Review derives its plan view from recorded decisions. |

The six [snapshots](snapshots) include byte-identical executed models, exact native
input arrays and full candidate reports. PLY/display precision follows the host;
NPY files preserve native arrays. Select a recorded row to inspect alternatives
omitted from the bounded overview. A new publication marks the view stale; only
explicit recapture replaces it. Lean/active/errored runs do not supply a new capture.
Training remains separate with independent Apply/Cancel.

Colors: green eligible, red rejected, orange unknown, magenta observed collisions,
RGB TCP axes. Configured cups/jaws/CAD/sweeps are derived geometry. Unknown query
points are not free-space observations. Geometric eligibility and scores do not
qualify physical grasp success, hidden contacts, robot reachability or a motion.
The real poses are explicit dataset assistance, not independent localization.
There are no robot, ROS, gripper or motion-execution blocks in these projects.

See [source attribution and limitations](SOURCES.md), [review behavior](COMPLETED_REVIEW.md),
[verification](verification.json) and [file pins](integrity.json).
Project assets are shared by content hash, without duplicate model copies per case.

Replay with the host's existing Python environment, without installing anything:

```powershell
python examples.py verify . C:\path\to\new-review-check --native
```

The check validates file pins, numeric/default-cloud parity, native Results controls,
all six frozen exports and Save As/reopen with exact recipe bytes. This is offline
software evidence, not physical grasp or robot qualification.
