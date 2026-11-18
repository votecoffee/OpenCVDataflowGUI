# SimpleReferenceInspect completed-run review

Review the finished frame/model, residual/deviation maps, support and actual
gates from **Results → Review completed reference inspection…**. Synchronized
reference/current comparisons and native exports share the existing viewer while
training drafts stay independent. No new inspection or exposure fitting occurs.

## Run

1. Use a host with paired completed snapshots and the independent plugin Results
   hook (`f2eff46`/`70b5242` or later).
2. Restart if another ReferenceInspect/PatternMatch package is loaded. Load
   **pattern_plugin.py**, then **plugin.py**, before opening a project. These
   checksum-pinned loaders import the bundled PatternMatch dev2/ReferenceInspect
   dev4 wheels without installing or replacing host environment packages.
3. Open a project below, run in Debug with input/output previews retained, select
   the inspection block and open its Results review.
4. Change comparisons, zoom/pan together, use 100%/Fit comparisons, then inspect
   Input overlay/Assessment or the outer Exact consumed input tab. Export review…
   writes a new frozen-review folder. Capture latest explicitly changes the frame.

| Project | Frames | Expected inspection state |
| --- | --- | --- |
| [Generated](generated/project.json) | Independent nominal; material change | Pass; fail |
| [Fixture](fixture/project.json) | Reference; translated nominal; translated change | Pass; pass; fail |
| [Limited PCB control](visa_limited/project.json) | Self-reference; independent normal photo | Pass; not evaluated |
| [Posed real PCB hard case](visa_posed/project.json) | First held-out normal; first held-out anomaly | False alarm; flagged anomaly |

The recipes, masks, limits and existing preparation are unchanged. The posed
case selects fixed first normal/anomaly frames from the publisher split, without
using prediction scores. Ground-truth defect masks are evaluation-only. Existing
false alarms and incomplete defect coverage remain visible. The limited recipe
has one reference/no pose alignment; its independent normal image fails prerequisite
gates and is not evaluated for changes. Neither case is qualified PCB inspection.

The selected anomalous frame has **3.1% pixel precision / 24.8% recall** under
the unchanged model. [Evaluation graphic](snapshots/visa_posed_1/ground_truth_comparison.png)
shows green true positives, red false positives and amber missed labeled pixels.
[Normal-frame false alarm](snapshots/visa_posed_0/ground_truth_comparison.png)
and [failed prerequisite assessment](snapshots/visa_limited_1/assessment.png)
make the current limitations visible. Dataset labels/masks never drive inference.

## Assess

Reference/current share a grid and intensity scale. Signed residual uses native
intensity; deviation is residual/tolerance, not probability (one is the boundary).
Gray maps were uninspected. Coverage: green inspected, red accepted changes,
amber expected but not inspected, dark excluded. Input overlay and native source
maps retain input coordinates; the assessment is display-only. Component areas
use reference pixels. Failed prerequisites show Not evaluated and null metrics,
not a passing zero-area measurement.

[snapshots](snapshots) contains one export per frame: comparison PNGs/assessment,
native reference and source maps, exact consumed/measurement pixels, optional
normalized current/support, executed model and JSON provenance. The model is
re-encoded with retained training references; original/exported archive hashes
are separate. Normalized current resamples frozen pixels with recorded exposure;
it does not refit or evaluate. Software completion IDs do not prove exposure identity.
[Full workflow and retention scope](COMPLETED_REVIEW.md).

## Verification and rebuild

The actual pinned wheel passed **24 tests**. [wheel-verification.json](wheel-verification.json)
and [pattern-wheel-verification.json](pattern-wheel-verification.json) pin source,
module bytes, license and wheel hashes. Review/model copies occur on explicit
capture; execution only retains its existing reference buffers and a small record.
Lean skips that additional record; a frozen model is cached per loaded revision.

[native-verification.json](native-verification.json) records native Results review
over four projects/nine frames and nine exports. Inspection, exposure fitting and
model loading are patched to fail during review. Primary reports/maps/default
graphics and parameters remain unchanged. [baseline-comparison.json](baseline-comparison.json)
compares the previous ReferenceInspect core. [native-replay.json](native-replay.json)
and [host-native-replay.json](host-native-replay.json) record native replay and
portable Save As/reopen. Trained archives copy exactly; ordinary input paths rebase
to the original assets. These are Qt offscreen checks, not operator acceptance.

```powershell
python verify_examples.py . C:\path\to\new-verification --native
python build_examples.py C:\path\to\existing\reference_inspect C:\path\to\new-examples
```

Use an SDK-capable environment with no other copies of the tool modules imported.
Both output folders must be new. The builder copies published baseline/replay
receipts when present; it creates a fresh native-verification.json/expected.json.
The verifier writes a separate checks.json. No training or threshold tuning occurs.

## Attribution

Real **VisA** photographs, masks, retained training pixels and derived reviews:
Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation*, ECCV 2022, **CC BY 4.0**. [Dataset](https://registry.opendata.aws/visa/) ·
[license](https://creativecommons.org/licenses/by/4.0/) · [source details](SOURCES.md).
Generated controls and authored code use the supplied [MIT license](LICENSE).
