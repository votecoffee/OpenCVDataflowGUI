# Simple3DLocate result review

Requires **Simple3DLocate 0.1.0.dev14**. These saved views demonstrate clearer
assessment with unchanged matching settings and model assets.

| Case | Result image | Rejected-pose view | Accepted count |
| --- | --- | --- | --- |
| Generated workflow | [Result](generated.png) | [Rejected candidate](generated_rejected.png) | 2 |
| Measured T-LESS scene | [Result](real_positive.png) | [Rejected candidate](real_positive_rejected.png) | 1 |
| Known novel-model failure | [Result](real_no_match.png) | [Rejected candidate](real_no_match_rejected.png) | 0 |

The status, legend and measurement panel are outside the cloud viewport.
The failed novel-model example shows all 24 tested candidates rejected at
geometric verification. Its selected candidate has 30.9% coverage against the
unchanged 50% requirement. It is a failed localization, not a successful
empty inspection. The remaining novel-model accuracy gap is unchanged.

Open the corresponding saved project and **Train 3D locator…**. Capture/load
its independent test scene, locate, then use **Result selection** to select
an instance or retained rejected candidate. Hide **Scene** to enlarge the
selected model; toggle axes and supported/missing/hidden/free-space layers.
Orange rejected poses are not detections. These choices do not change
training, poses, thresholds or measurements.

## Point clouds and reports

Each `*_accepted_fit.ply` contains measured gray scene samples and only
accepted fitted model points, with supported/missing/hidden/free-space colors
and pose axes. For the zero-match case this cloud contains only the measured
scene. `*_REJECTED_candidate.ply` is a separate, explicitly rejected-pose
display with the transformed full model in orange. These model samples are
expected geometry, not additional sensor returns. Units are millimeters;
the frame is `camera`. PLY does not retain the unit/frame fields itself.

The corresponding JSON report retains the model-to-camera poses, stage
counts, gate measurements and limits. [receipt.json](receipt.json) records
input hashes and accepted counts. At most eight rejected candidates are
retained, preferring later verification stages and then the existing score
within each stage. A rejected pose is not known to be physically correct.

Rebuild from the host repository root with its Python environment:

```powershell
python "examples/Simple Plugins/3d_locate/result_review/build_views.py"
```

## Attribution

Generated data and builder code use this repository's MIT license. Real
T-LESS data and all derived images, reports and clouds retain **CC BY 4.0**.
Authors: Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis
Lourakis and Xenophon Zabulis. *T-LESS: An RGB-D Dataset for 6D Pose Estimation
of Texture-less Objects*, WACV 2017.
[Original project/license](https://cmp.felk.cvut.cz/t-less/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[original-scene attribution and changes](../real_tless/source/ATTRIBUTION.md) ·
[novel-model attribution and changes](../novel_models/source/ATTRIBUTION.md).
This example adds display layout, candidate/gate annotations and colorized
PLY export to existing measured-depth and CAD derivatives. No endorsement is
implied. Dataset licenses do not replace the separately documented OpenCV
surface-matching patent notice.
