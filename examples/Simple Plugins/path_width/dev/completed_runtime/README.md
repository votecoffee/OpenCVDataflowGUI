# SimplePathWidth completed-run review

Review the finished runtime frame directly from **Results → Review completed
path width…**. The image, pixel/physical trends, selected scan, native intensity
profile and limits table are linked. This action is independent of training.

## Run

1. Use a host with paired execution snapshots and the independent read-only
   plugin Results hook (`f2eff46`/`70b5242` or later).
2. Restart if another PathWidth package is loaded. Load **plugin.py first**.
   It checks/imports the bundled dev13 wheel without installing into the host.
3. Open a project below, run in Debug with input/output previews retained,
   select the PathWidth block and use its Results review action.
4. Click the image/trends or step through problem stations. Zoom/pan, Focus
   station and 100% help assess the edges. Export station… writes a new folder.
5. The view stays frozen as the editor runs. Capture latest completed run
   explicitly replaces it. Unavailable capture keeps the prior view labeled.

| Project | Frames | Expected path judgments |
| --- | --- | --- |
| [Generated bead](generated/project.json) | Reference; gap/narrow/competing-edge control | Pass; fail |
| [Calibrated bead](calibrated/project.json) | Nominal; narrow; off-path | Pass; fail; fail |
| [Real weld](weld/project.json) | Training photo; separate photo; failing photo | Pass; pass; fail |

All recipes, limits and input preparation are copied unchanged. Generated
examples are controls. Real weld photographs and supplied masks have their own
MIT attribution in [SOURCES.md](SOURCES.md). These are review examples, not
independently measured dimensional ground truth or recognition qualification.
Weak/wrong edges in the failing weld remain visible.

## Assess and export

Green = passing measured station; red = measured violation or missing pair;
amber = unresolved edge ambiguity; gray = unobserved masked/clipped/corner scan;
cyan = selected scan and edges. Only missing pairs contribute to gaps.
Unavailable widths are null, not zero. The full-path verdict considers stations
outside the selected crop; the summary shows group counts, coverage and reasons.
Physical units and image pixels stay distinct.

The [snapshots](snapshots) select the middle of the longest run of each problem
type present, or a measured station when none occur. This is a display choice,
not an acceptance change. Exports include overlay/profile/assessment and trends,
native exact input/measurement NPY arrays, inspected-mask PNG, original executed
manifest bytes and review JSON with full measurements/limits/pose/calibration
and ticket/spatial/pyramid metadata. The manifest is not a full training archive;
the training reference is not loaded or replaced by the current image.

Selected assessments: [separate real weld](snapshots/weld_1_measured/assessment.png),
[real missing pair](snapshots/weld_2_missing/assessment.png),
[calibrated width violation](snapshots/calibrated_1_out_of_tolerance/assessment.png)
and [generated ambiguity](snapshots/generated_1_ambiguous/assessment.png).

The core does not retain every sampled profile. Review resamples only the
selected diagnostic profile from frozen pixels/settings/calibration and caches
64 stations. It never reruns edge pairing, width measurement or the verdict.
No extra profile sampling or source-image copy is added during execution;
Lean retains no extra review record. See [full guide](COMPLETED_REVIEW.md).

## Evidence and rebuild

[wheel-verification.json](wheel-verification.json) pins the tested source and
wheel hash; the loader checks it. The actual wheel passed **52 tests**, including
CAD import with an isolated optional dependency, float/native-depth input,
unavailable/masked profiles, changing/deleted models, effective calibration,
worker/display transfer and an independent unapplied training draft. No host
environment package was installed or replaced.

[native-verification.json](native-verification.json) records the native Results
action and exports across three projects/eight frames. Measurement/model reload
is patched to fail during review; primary values, default graphics and parameters
stay unchanged. [baseline-comparison.json](baseline-comparison.json) compares
every station/scalar/physical report and default raster to the prior core.
[native-replay.json](native-replay.json) and
[host-native-replay.json](host-native-replay.json) record fresh native replay
and Save As/reopen from source and final host locations. Recipes copy exactly;
ordinary input images rebase to the original assets on Save As. These are Qt
offscreen checks, not a hands-on operator acceptance record.

From an SDK-capable environment with no other PathWidth package imported:

```powershell
python verify_examples.py . C:\path\to\new-verification --native
python build_examples.py C:\path\to\existing\path_width C:\path\to\new-examples
```

Both outputs must be new directories. The builder reuses existing public assets
and copies published replay/baseline receipts when present; its new run writes
native-verification.json/expected.json. The verifier writes a separate checks.json.

## Attribution

Hyun Soo Kim (2024), *Weld bead images and masks*, Figshare v1, **MIT License**.
[Dataset](https://doi.org/10.6084/m9.figshare.25904014.v1) ·
[retained MIT copyright/terms](WELD-LICENSE-data.txt) · [full sources](SOURCES.md).
Generated controls and authored code use the supplied [MIT license](LICENSE).
