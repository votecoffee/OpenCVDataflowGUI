# SimpleBinPicking suction planning

The separate [CAD-taught jaw examples](known_part_jaw/README.md) add a pinned
explicit-load plugin, clear/obstacle/unknown controls and two real T-LESS
pose-assisted replays. They use native colored fly-around contact/sweep graphics.
The initial jaw core retains authored frames through API/CLI; native jaw teaching
and robot execution integration remain next. The suction projects below keep
their existing installed package and recipes.

Install the included `opencv-simple-bin-picking` **0.1.0.dev3** in the Python
environment running this app, from the application repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .\integrations\simple_plugins\bin_picking
```

This adds **Simple Plugins → Bin picking → SimpleBinPicking**. The numerical
core needs NumPy and SciPy; native training uses the host's Qt/OpenCV. Restart
the app after installation. These projects use stable type `3d.simple_bin_picking`.

| Project | Frozen result |
| --- | --- |
| [Bin to table grid](bin_to_grid/README.md) | Actual six-pick simulation, 37 MoveIt arm trajectories, native colored cloud/recorded review; assisted grasp explicitly labeled. |
| [Motion review](bin_to_grid/motion_review/README.md) | Two further 6/6 native episodes, phase-colored planned paths and measured-joint FK dots; historical held transfer retained. |
| [T-LESS mesh picking](tless_mesh_grid/README.md) | Six real CAD mesh instances, retained source pose labels, fixed taught annular top and native 6/6 assisted placements. |
| [Visible raised patch](generated/good/project.json) | 10 eligible alternatives from 64 candidates. |
| [Empty floor](generated/empty/project.json) | No candidates above the minimum pick height. |
| [Missing depth](generated/missing/project.json) | No candidates; no fabricated surface or pose. |
| [Depth holes](generated/holes/project.json) | 64 evaluated, no eligible picks; unknown contact remains explicit. |
| [Narrow configured bin](generated/narrow_bin/project.json) | 16 evaluated, no eligible picks; tool/bin clearance fails. |
| [T-LESS reference frame 3](real_tless/frame000003/project.json) | 256 evaluated, no eligible picks. |
| [T-LESS held-out frame 13](real_tless/frame000013/project.json) | 256 evaluated, no eligible picks. |
| [T-LESS held-out frame 24](real_tless/frame000024/project.json) | 256 evaluated, no eligible picks. |

Run a project and select **Cloud / graphics** for the native RGB flyaround
thumbnail/popout. Green cups are eligible, red rejected, orange unknown and
yellow selected. RGB TCP axes, blue tool sweeps, violet bin boundaries and
magenta/orange witnesses explain the result. The overview draws a subset and
one envelope to reduce clutter; all candidates remain in the report/table.
`result.ply` is an inspectable native output and `result.png` is a computed
preview. Graphic points never enter the contact calculation.

Open **Train bin picking…** to restore retained reference depth/RGB, camera,
bin and tool dimensions. Teach floor picks and edit the bin in the native
cloud, then preview and select candidates. Capture connected source inputs
before the picker is configured, or freeze a completed execution for review.
**Stage recipe** updates the inspector draft; inspector **Apply** activates it.
Cancel preserves active parameters. [Training workflow](TRAINING.md).

All real cases use the same frozen recipe and 1 mm surface tolerance. They
retain noisy/missing contact and tilt rejections, rather than relaxing limits
per image. They are real first-return tabletop-clutter captures, not bulk-bin
or successful suction-pick demonstrations. Camera world extrinsics are supplied
calibration, not estimated object poses. No grasp labels or physical pick
outcomes are present. [Attribution and transformations](real_tless/ATTRIBUTION.md).

Ranked outputs are alternatives and can lie on the same object. Reobserve
after every physical pick. Geometry scores are not success probabilities.
Missing/occluded sweep samples remain unknown; finite sampled clearance does
not certify the entire continuous volume, robot arm, IK or suction seal.

Each project has sealed `assets/recipe.sbp`, raw depth, expected numerical
evidence and RGB result cloud. `sha256.json` pins public assets; original
dataset members retain separate publisher provenance. [Application plan](APPLICATION_PLAN.md).
The builder uses `ProjectSession` for static validation, save and normal graph
execution. Generated controls and builder code use [MIT](LICENSE); T-LESS
source and derived assets retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
