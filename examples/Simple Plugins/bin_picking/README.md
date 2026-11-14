# SimpleBinPicking suction planning

Install `opencv-simple-bin-picking` **0.1.0.dev0** in the Python environment
running this app, using your supplied wheel or checkout:

```powershell
.\.venv\Scripts\python.exe -m pip install "C:\path\to\opencv_simple_bin_picking-0.1.0.dev0-py3-none-any.whl"
```

This adds **Simple Plugins → Bin picking → SimpleBinPicking**. The numerical
core needs NumPy and SciPy; native training uses the host's Qt/OpenCV. Restart
the app after installation. These projects use stable type `3d.simple_bin_picking`.

| Project | Frozen result |
| --- | --- |
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
