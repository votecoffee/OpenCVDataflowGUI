# Simple3DLocate · novel-model training and validation

Requires **Simple3DLocate 0.1.0.dev13**. These examples demonstrate automatic
bounded CAD training and reproduce **known matching failures** on new models.
Both saved first-view projects currently return zero matches:

- [Object 6 project](object_000006/project.json)
- [Object 8 project](object_000008/project.json)

Open **Train 3D locator…** to restore the full reference and CAD mesh. The new
training sampler keeps at most 2,000 learned points without changing the user's
sampling distance or fit parameters. Object 6 has 2,482 occupied voxels and
object 8 has 2,662; both retain all 6,000 reference points and their meshes.
The training/status display reports **Model budget**, including on restoration.

## Training budget

Models already within the budget retain the previous sample arrays exactly.
For larger references, training selects at most 10,000 uniform candidate voxels
with extrema retained, estimates normals when needed, then uses geometry-aware
selection to retain 2,000 samples. Fine features can still be lost under a finite
budget. Runtime model and candidate limits are unchanged; no speedup is claimed.

[training_budget.json](training_budget.json) records default training across
all thirty T-LESS CAD models. All thirty now train; fourteen previously exceeded
the point limit. The other sixteen retain exact legacy samples. This measures
training behavior. Matching results are recorded separately below.

## Frozen validation cases

Each model has nine cases. Three positive RGB-D captures are tested both with
their supplied visible masks and with all measured depth pixels. Three full
captures from the other scene are target-absent negatives.

| Model | Positive scene/captures | Target-absent scene/captures |
| --- | --- | --- |
| 6 | 2: 3, 255, 494 | 3: 5, 261, 493 |
| 8 | 3: 5, 261, 493 | 2: 3, 255, 494 |

IDs and views were selected using metadata before matching. The first/middle/
last available depth filenames and cross-scene target absence determine the
selection. CAD alone trains each recipe. All models use the same settings,
four workers and projective scoring; poses and exact discrete symmetries enter
evaluation only. No annotated boxes or poses guide search, and no workspace is
manually clipped. Full-frame measured depth remains available to every case.

Load [object 6 validation](validation_object000006.json) or
[object 8 validation](validation_object000008.json) through **Optimize execution…
→ Load validation manifest…**. These portable manifests include cameras, depth,
pose truth and symmetry. They share measured clouds in `validation/`.

## Retained results

Correct / missed / extra per repeat, two repeats per scene:

| Model | Global PPF | Planes | Features | Regions |
| --- | --- | --- | --- | --- |
| 6 | 0 / 6 / 0 | 0 / 6 / 1 | 0 / 6 / 0 | 0 / 6 / 2 |
| 8 | 0 / 6 / 0 | 0 / 6 / 0 | 0 / 6 / 0 | 0 / 6 / 0 |

See [summary.json](summary.json), [object 6 comparison](object_000006/comparison.json)
and [object 8 comparison](object_000008/comparison.json). All twelve positives
remain missed. Optional plane/regional routes also produce false positives
on a model-6 negative. Both repeats retain the same counts. No plan qualifies,
and the optimizer recommends none. All predictions and overlays retain failures.

[annotated_pose_diagnostic.json](annotated_pose_diagnostic.json) is a separate
**annotation-only evidence diagnostic**, with no candidate search. At exact
annotated transforms all six isolated cases fail geometric acceptance; sampled
coverage is about 10–41%, below the unchanged 50% minimum. Those diagnostic poses
never count as detections and are not supplied to the saved runtime projects.
Verification visibility and scene-normal evidence need further work alongside
candidate recall. Successful training currently does not provide general CAD
matching robustness.

The selected cases exposed a training-budget failure before matching. After
the generic sampler fix, all profiles were evaluated without per-model tuning.
This is development validation, with a small set of views and one sensor.
It does not establish production or official BOP benchmark performance.

Reproduce with both Python packages installed:

```powershell
python build_novel_validation.py source output --repeats 2 --host-project
python diagnose_novel_geometry.py source annotated_pose_diagnostic.json
python benchmark_training_budget.py path/to/tless_models.zip training_budget.json
```

`build_tless_demo.py` provides shared mesh/overlay helpers for this builder.

## Attribution and license

Real T-LESS data and derived clouds, recipes, reports and overlays retain
**CC BY 4.0**. Authors: Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas,
Manolis Lourakis and Xenophon Zabulis. *T-LESS: An RGB-D Dataset for 6D Pose
Estimation of Texture-less Objects*, WACV 2017.

[Original project/license](https://cmp.felk.cvut.cz/t-less/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[full attribution/changes](source/ATTRIBUTION.md),
[pinned files and hashes](source/provenance.json),
[frozen selection](source/selection.json).
Changes comprise measured depth-to-XYZ conversion, supplied-mask isolation,
bounded CAD training, retained camera/full-frame depth context, validation and
pose/fit overlays. No hidden scene returns are invented. No endorsement is
implied. Builder code uses this repository's MIT license; the existing OpenCV
surface-matching patent notice still applies.
