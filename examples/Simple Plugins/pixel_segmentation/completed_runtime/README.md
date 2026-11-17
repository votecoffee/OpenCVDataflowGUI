# SimpleSegmentation completed-run region review

Inspect exact completed source pixels, pinned model/region assets, source masks,
inspected coverage, retained forest votes and reported component statistics from
**Results → Review completed segmentation…**. Annotation/training stays independent.

## Run

1. Use a host with paired completed snapshots and the independent plugin Results
   hook (`f2eff46`/`70b5242` or later). A compatible OpenCV/ML build is required.
2. Restart if another SimpleSegmentation package is loaded, then load **plugin.py
   first**. It checks/imports the bundled dev2 wheel without installing packages.
3. Open a project below, run in Debug with input/output previews retained, select
   **segment** and open its Results review.
4. Choose a part/image and component row, or click its projected bounds. Focus
   selected, scroll zoom and drag pan help assess it. Other tabs show fixed-scale
   votes, coverage, native input and assessment. Export frozen review… writes a
   new folder; Capture latest explicitly changes a frozen frame.

| Project | Frames | What it demonstrates |
| --- | --- | --- |
| [Tote](tote/project.json) | Loaded; empty | Held-out real inventory; observed zero foreground differs from unknown |
| [PCB](pcb/project.json) | Training photo; independent photo | Exposed laminate as a foreground material, not a defect |
| [Votes unavailable](votes_unavailable/project.json) | First PCB photo | No emitted score/heatmap: review does not infer missing votes |
| [Heatmap-only scores](heatmap_only/project.json) | Loaded; empty tote | Retains already computed votes even without emitting the score port |
| [Unobserved control](unobserved/project.json) | Loaded tote | Deliberate source-aligned all-zero valid mask: unknown, not clean background |

Models, inspection regions, cutoffs, minimum areas, sampling and worker settings
are unchanged. Controls change only score-output/graphics retention or deliberately
remove observation validity. They do not tune the classifier. Labels/masks were
drawn for the existing examples and are not publisher-provided object ground truth.
Touching inventory can form one component. These demonstrate review and current
recipe behavior, not broad inspection or dimensional qualification.

## Interpret and export

Red is foreground; green inspected background; gray checks excluded/unobserved.
Cyan shows projected sampled-grid component bounds and center, not an isolated
member mask. Source overlap is a union/max-vote projection; component area and
local mean votes belong to the sampled part grid. Forest fractions use fixed
0..1 Viridis and are not calibrated probabilities. Outside zeros are sentinels;
the inspected mask determines availability. No observations gives a null fraction.
Block pass means some observations were processed, not a quality acceptance gate.

[snapshots](snapshots) has one export per frame. The largest reported component
is selected for presentation when present; empty cases retain the part footprint.
Each export includes exact original/measurement NPY pixels, mask/inspected arrays,
available votes, original executed YAML/region PNG bytes, full settings/report/
selection/provenance JSON and source/coverage/votes/assessment PNGs. The assessment
is a display-only crop; original/source/sampled coordinates stay explicit.
[Full workflow and limitations](COMPLETED_REVIEW.md).

Selected views: [loaded inventory](snapshots/tote_0/assessment.png),
[observed empty tote](snapshots/tote_1/assessment.png),
[unobserved control](snapshots/unobserved_0/assessment.png) and
[votes unavailable](snapshots/votes_unavailable_0/assessment.png).

## Verification and rebuild

The pinned wheel passes **58 tests and 12 subtests**, including native Results
selection/export, RGB16/pyramid metadata, matched sampling/overlap geometry,
unavailable votes/observations, active/error/stale states and an independent
unapplied training setting. A regression deletes original model/region files
after Save As and successfully reopens the copied trained assets.
[wheel-verification.json](wheel-verification.json) pins source/module bytes/license.

[native-verification.json](native-verification.json) records five projects/eight
frames/eight exports with inference/model loading/decoding patched to fail during
review. Primary reports and default arrays/graphics/parameters stay unchanged.
[baseline-comparison.json](baseline-comparison.json) compares the previous core;
[native-replay.json](native-replay.json) and [host-native-replay.json](host-native-replay.json)
record actual native offscreen replay and portable Save As/reopen. Model/region
pins copy exactly; ordinary input image references rebase. Training/validation
manifests remain separate source material. No host package was installed/replaced.
These checks do not constitute hands-on operator acceptance or accuracy claims.

```powershell
python verify_examples.py . C:\path\to\new-verification --native
python build_examples.py C:\path\to\existing\pixel_segmentation C:\path\to\new-examples
```

Use an SDK-capable environment with no older package imported. Both outputs must
be new directories. The builder copies published replay/baseline evidence when
present and creates fresh native-verification.json/expected.json; the verifier
writes its separate checks.json receipt.

## Attribution

VisA PCB photographs: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and
Onkar Dabeer, ECCV 2022, CC BY 4.0. ARMBench tote photographs: Amazon.com, Inc.,
CC BY 4.0. [Sources/preparation](SOURCES.md) ·
[retained ARMBench license](ARMBENCH-License-CC-BY.txt). Authored code uses [MIT](LICENSE).
