# Experimental boundary verification

Requires Simple3DLocate **0.1.0.dev19**. These projects enable an experimental
measured-surface continuation check around a mesh's predicted silhouette.
The normal default remains **off**. This comparison retains the same detection
counts as boundary-off: some wrong candidates are rejected, then replaced by
other wrong poses. It does not resolve the general accuracy gap.

## Start with the native evidence views

Open and run one of these projects, then select its **Cloud / cloud** output.
The thumbnail and enlarged viewer use native RGB flyaround clouds with pose
axes. These load computed evidence for a cached runtime pose; no localization
or annotation-seeded detection runs in these three review projects.

| Review | What it shows |
| --- | --- |
| [Rejected false candidate](pose_reviews/rejected_false_pose/project.json) | 327/500 sampled outline pairs continue on a larger measured surface; at most 34.6% could provide distinctive support, below the unchanged 50% requirement. |
| [Accepted wrong replacement](pose_reviews/retained_false_pose/project.json) | 254/461 pairs are unknown. The optimistic support bound is 55.1%, so this check cannot reject the pose safely. The target is absent. |
| [Accepted correct isolated pose](pose_reviews/retained_correct_pose/project.json) | 169 changed-depth pairs, 13 continuations and 88 unknowns; the 95.2% possible-support bound does not reject it. This remains a limited isolated success. |

Orange means measured surface continuation; green means changed depth outside
the expected outline; magenta is unknown. Colored inner points are expected
geometry, while available outer points are actual measured returns. Gray is
the measured scene; RGB axes mark the cached pose. Unknown depth is never
filled or treated as proof of a clean outline. The percentages are bounds on
sampled geometric evidence, not probabilities of pose correctness.

## Run the actual matcher

The seven root `project_*.json` files run actual unseeded localization with
`boundary_verification=auto`, preserve native **Cloud / fit** output and keep
the original model-source gallery selection. The supplied masks are used only
for documented isolated cases; full measured depth is a separate input.
Test poses/symmetries are evaluation only, never graph input seeds.

All 66 cases ran twice with identical accepted poses/scores within this
configuration. There are 16 positive cases from eight captures per source
(each isolated/full pair is correlated), plus six full-scene target-absent
cases. Teaching replays remain excluded from test totals. This is an examined
development set, not unseen qualification or an official BOP benchmark.

| Source | Correct / 16 | Missed | Extra poses | Extras on absent targets |
| --- | --- | --- | --- | --- |
| CAD | 0 | 16 | 0 | 0 |
| Reconstructed | 2 | 14 | 18 | 7 |
| Captured | 2 | 14 | 25 | 10 |

These are the same totals as the [boundary-off comparison](../reference_sources/README.md).
All full-scene positives remain missed. Captured templates lack complete mesh
geometry, so boundary verification reports unavailable for them. Remaining
accepted wrong poses and all rejected candidates are retained in `reports/`.
`artifact_hashes.json` and `source_hashes.json` pin derived and raw assets;
`pose_reviews/artifact_hashes.json` pins the separate native evidence views.
Keep the `.gitattributes` files to preserve the pinned bytes.

## Train and review

Open **Train 3D locator…**, run a test scene and choose one accepted instance
or rejected candidate. **Review boundary evidence…** opens the native layered
popup. Orbit and layer changes reuse calculated evidence; editing training,
camera or input invalidates it. Export writes a fresh folder with the visible
PLY, raw NPZ pairs/states/residuals and JSON provenance.

`off` preserves existing results. `report` measures without rejecting.
Experimental `auto` rejects only sufficiently observed continuation evidence.
A mesh, calibrated camera, surface mode and enabled depth verification are
required. Unknown pairs remain possible support, and insufficient evidence
does not create an additional rejection.

The physical probe distance is one model sampling cell plus two endpoint
distance tolerances. Three measured interior points define a plane; an outside
depth return on that plane proves surface continuation. At most 8,192 evenly
selected boundary/direction pairs are inspected. A candidate is rejected only
when `1 - continued_pairs / sampled_pairs < min_coverage`, with enough distinct
observed outer rays. No example-specific parameter was introduced or tuned.
Thin/curved silhouettes, sensor noise, occlusion, clipped views and neighboring
surfaces still require broader validation. This route adds work when enabled.

## Rebuild

Install the plugin and host, and use fresh output folders:

```powershell
python benchmark_reference_sources.py source comparison_output --boundary-verification auto --repeats 2 --host-projects
python build_boundary_reviews.py source comparison_output review_output
```

Builders verify input hashes before processing. The pose-review builder
selects saved runtime accepted/rejected poses and never runs localization.

## Attribution and license

T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects.
Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis,
and Xenophon Zabulis. WACV 2017.
[Publisher project and CC BY 4.0 license](https://cmp.felk.cvut.cz/t-less/).
See [full source attribution](source/ATTRIBUTION.md) for pinned raw members,
official mirror revision, archive verification and modifications. Raw and
derived data retain CC BY 4.0; implementation code uses MIT. Preserve these
notices on redistribution. No endorsement is implied.
