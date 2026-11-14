# Bin-grid motion review

Offline review of two completed native simulation episodes and an earlier held
transfer. No ROS installation or running simulator is required to review them.
Install the included plugin **0.1.0.dev2** and load [../plugin.py](../plugin.py)
using the [parent example instructions](../README.md#install).

## Usage

Open [shifted_stacks/project.json](shifted_stacks/project.json) and Run once.
The **Recorded bin-grid pick** thumbnail is a native colored point cloud: open
its flyaround view, then orbit/zoom around the source scene and TCP paths.
Select **Cell** 1..6 to review a different pick. The result image list includes
**motion_summary**, with phase colors, sample counts and durations, as well as
before/after source overlays. The **motion** output is the primary cloud.

| Graphic | Meaning |
| --- | --- |
| Blue/orange/cyan/purple/yellow/green lines | Planned pregrasp/approach/lift/transfer/place/retreat TCP positions. |
| Gray lines | Initial observation or final park move. |
| Pink dots | Discrete TCP positions reconstructed from measured joint feedback. Gaps are left unfilled. |
| Red/green/blue pose axes | Camera-derived grasp candidates, not path samples. |

Both planned and feedback positions use the actual MoveIt robot-description
kinematic chain. Three planned states were checked against `GetPositionFK`.
This is model-based motion review, not independent TCP metrology or a continuous
collision certificate. The colored scene is a source capture at its own retained
timestamp; historical paths are not simultaneous collision geometry.

## Native evidence

| Episode | Result | Arm actions | Max final grid error |
| --- | --- | --- | --- |
| [Baseline](baseline/project.json) | 6/6 released, empty bin | 37 | 1.2381 mm |
| [Shifted and rotated stacks](shifted_stacks/project.json) | 6/6 released, empty bin | 38 | 0.7544 mm |
| [Earlier held transfer](held_transfer/project.json) | 2 placed, cell 3 held, cells 4..6 skipped | Partial paths only | No success claim |

Detector ROI/contact settings were unchanged between the two completed runs.
The variant shifts the three stacks by up to 12 mm in X and 6 mm in Y, and
rotates them by 5, -6 and 8 degrees. Both use fresh fixtures; neither uses an
in-place clock reset. The baseline started from an existing clear observation
pose after prior zero-pick planning holds; its startup move was correctly skipped.
The variant started from the normal fixture arm position.

Open-space pose requests try a complete collision-checked Cartesian candidate,
then bounded global search if necessary. Required Cartesian contact moves must
be complete. The variant exercised three global fallback plans. All accepted
retimed paths received additional whole-robot sampled validity checks: 2,519
states in baseline, 3,651 in the variant. Feedback records contain 2,401 and
3,047 measured-joint samples respectively.

The historical held case preserves a MoveIt `ValidateSolution` rejection of a
forearm/gripper self-collision in a global path. No transfer was executed after
that rejection; the part remained attached after lift and later cells were
skipped. [Planner diagnostic](held_transfer/planner-diagnostic.txt) retains the
native error excerpt. Newer successful runs do not relabel this failure.

[acceptance.json](acceptance.json) pins episode identities and counts. Each
completed case includes its scene, final evaluation and final unaltered camera
image. Per-cell archives retain correlated plan/action receipts, source PNGs,
calibration/capture stamps, native RGB clouds and model-bound motion samples.
Original float32 depth is losslessly stored in **source-depths.npz**, keys
`before`/`after`; bytes were checked against the original NPY arrays. The parent
example's older frozen episode remains unchanged.

These are authored-box simulations with privileged collision/evaluation geometry
and pose-gated temporary fixed-joint grasp assistance. Friction grasp, arbitrary
clutter, broad repeatability and physical robot accuracy remain unqualified.

## License

Original example/scene code: MIT. Robot rendering and dependencies retain their
own licenses and [attribution](ATTRIBUTION.md). No external dataset or private
customer images are included. [sha256.json](sha256.json) pins the review assets.
