# Real model-source comparison

Requires Simple3DLocate **0.1.0.dev18**. These seven runnable projects compare
CAD, publisher reconstructed meshes and a captured depth reference. They retain
successes, misses and accepted wrong poses from a fixed development comparison.
They demonstrate how to assess results; general 3D accuracy remains unresolved.

## Open and review

Open a `project_*.json`, run it and select **Simple3DLocate → Cloud / fit**.
The result thumbnail and enlarged viewer use the native colored flyaround
cloud with RGB pose axes. Bitmap rendering is disabled. Green means measured
geometric support, not ground-truth pose correctness. The two `false_positive`
projects intentionally retain accepted poses in scenes where the target is
absent. Read each project's comment and the expected counts in `summary.json`.

| Project | Expected outcome |
| --- | --- |
| [CAD, object 6](project_cad_object000006__missed.json) | Missed positive |
| [Reconstructed, object 6](project_reconstructed_object000006__correct.json) | One correct isolated pose |
| [Reconstructed, object 6 nearby view](project_reconstructed_object000006__missed.json) | Missed positive |
| [Reconstructed, object 8 absent](project_reconstructed_object000008__false_positive.json) | Five accepted wrong poses |
| [Captured, object 8](project_captured_object000008__correct.json) | One correct isolated pose |
| [Captured, object 6](project_captured_object000006__missed.json) | Missed positive |
| [Captured, object 8 absent](project_captured_object000008__false_positive.json) | Five accepted wrong poses |

Open **Train 3D locator…** to review the retained reference and trained points.
Mesh recipes also retain canonical triangles for resampling. With dev18,
**Import surface mesh…** accepts STL and triangle PLY, with explicit units,
scale and winding. The captured reference contains one measured view and has
no complete hidden-surface geometry. Stage and Apply remain explicit.
Existing model-source settings are preserved; no new default is selected.

## Complete fixed comparison

Two T-LESS objects each teach from one capture and test the next two available
capture IDs plus two distant views, both isolated with supplied masks and as
full scenes. Each source therefore has 16 positive cases from eight captures,
plus six full-scene target-absent cases. The isolated/full pairs are correlated.
The near IDs were frozen before acquisition and matching; adjacent available
IDs need not imply a small angular change. This development set has already
been examined; it is not an unseen qualification set or official BOP benchmark.

All sources share the teaching viewpoint, millimeter units, four workers and
otherwise default settings, including geometric scoring and existing gates.
Only teaching annotations transform captured references into the object frame.
No test pose is connected to a runtime port. Publisher finite symmetries and
5 mm / 5 degree tolerances are used for evaluation only. Even isolated tests
receive the full measured depth as a separate input; no hidden returns are added.

| Source | Correct / 16 positives | Missed | Extra poses, all cases | Extras on absent targets |
| --- | --- | --- | --- | --- |
| CAD | 0 | 16 | 0 | 0 |
| Reconstructed mesh | 2 | 14 | 18 | 7 |
| Captured surface | 2 | 14 | 25 | 10 |

All full-scene positives are missed. Reconstructed successes are the two
distant isolated object-6 frames; captured successes are the two nearby
isolated object-8 frames. Captured teaching replays succeed and are excluded
from these totals. All 66 cases ran twice with identical accepted poses and
scores. Changing the source alone does not solve the general accuracy gap.
No algorithm or acceptance threshold was changed for these examples.

Gallery selection is the first correct, missed-positive and target-absent
false-positive outcome per source, when available, scanning objects 6 then 8.
All cases, including those not selected for projects, remain in `reports/`.
`summary.json` records the complete matrix, settings, controls and selection.
`artifact_hashes.json` pins generated assets; `source_hashes.json` pins raw
inputs. Keep `.gitattributes` files to preserve those exact bytes.

## Rebuild

Install the plugin and host, then run into a fresh folder:

```powershell
python benchmark_reference_sources.py source comparison_output --repeats 2 --host-projects
```

The builder checks every pinned source hash before processing and verifies each
saved graph's actual outcome. It does not download data. `result_*.ply` files
are saved numerical result clouds, not substitutes for interactive host output.

## Attribution and license

T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects.
Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis,
and Xenophon Zabulis. WACV 2017.
[Publisher project and CC BY 4.0 license](https://cmp.felk.cvut.cz/t-less/).
See [full source attribution](source/ATTRIBUTION.md) for the pinned official
mirror revision, raw members, archive verification and modifications.
Raw and derived data retain CC BY 4.0; example implementation code uses MIT.
No endorsement is implied. Preserve these notices on redistribution.
