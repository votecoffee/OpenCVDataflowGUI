# SimplePatternMatch completed-run review

Review the exact completed input, executed model and cached fits from the
matcher's **Results** section. Zoom/pan, focus a pose, filter accepted/rejected
candidates and toggle found/missing/unmeasured edge layers without another
search. This action is separate from **Train geometric recipe…** and leaves
training drafts and acceptance settings unchanged.

## Run

1. Use a host with paired execution snapshots and the read-only plugin Results
   dialog hook (host commits `f2eff46` and `70b5242`, or later).
2. Restart the app if another PatternMatch package is already loaded. Load this
   folder's **plugin.py first**; it verifies and imports the bundled
   **0.1.0.dev2** wheel without installing it into the environment.
3. Open a project below and run in **Debug** with input/output previews retained.
   Select **matcher**, then **Results → Review completed pattern match…**.
4. Choose a row or **All returned instances**. The second tab shows the exact
   consumed input; the fit tab uses working grayscale pixels. Use **Capture
   latest completed run** to replace a frozen view explicitly.
5. **Export frozen review…** writes a new folder with the exact input array,
   executed model and review graphics. Close leaves the project unchanged.

| Project | Input frames | Returned instances | Purpose |
| --- | --- | --- | --- |
| [Brackets](brackets/project.json) | Tray; partial tray with decoy | 6; 4 | Multiple poses and incomplete fits |
| [Wired model](wired/project.json) | Same two trays | 6; 4 | Exact model packet from a separate loader |
| [Real PCB2](visa_pcb2/project.json) | Reference photo; separate photo | 1; 1 | Real VisA photographs and edge-fit evidence |
| [Blank control](blank/project.json) | Constant generated image | 0 | Failed inspection with no invented pose |

The existing recipes, training and search parameters are unchanged. These
examples demonstrate review and provenance; they do not establish pose accuracy.
The synthetic tray/blank controls are identified as generated. Rejected poses
remain diagnostics and are never returned as accepted detections.

## Saved reviews

There are thirteen exports under [snapshots](snapshots): the lowest-score
accepted instance and first retained score rejection for each populated frame,
plus a no-match overview. For example:

- [Separate PCB accepted fit](snapshots/visa_pcb2_1_accepted/assessment.png)
- [Separate PCB rejected fit](snapshots/visa_pcb2_1_rejected/assessment.png)
- [No-match overview](snapshots/blank_0_no_match/fit.png)

Green = found, red = missing, amber = outside/unmeasured and cyan = pose.
Expected probes and residuals can be toggled. Limits shown are the executed
recipe's actual limits; score is a geometric fit fraction, not a probability.
Disabled correspondences/diagnostics stay unavailable rather than being
recomputed. Lean execution does not retain the review cache.

Every export includes **consumed_input.npy** (exact dtype/channels/pixels,
no pickle), **working.png**, **inspected_mask.png**, **inspected.png**,
**fit.png**, **review.json** and **executed_recipe.npz**. Selected reviews also
include **assessment.png**. The executed recipe is re-encoded; its archive hash
can differ from the original pin. Both hashes are recorded. Source/working
coordinates and upstream ticket/spatial/pyramid metadata remain explicit.
Software snapshot identifiers do not prove physical exposure identity.

## Verification and rebuild

[native-verification.json](native-verification.json) records the actual native
Results action over four projects/seven frames, with matching and model reload
patched to raise during review. The captured source and fit are visible;
numerical outputs, default overlays and parameters stay unchanged within each
completion. [native-replay.json](native-replay.json) records a fresh pinned-wheel
native replay and portable Save As/reopen. Saved recipes copy byte-for-byte;
ordinary input image paths rebase to their original assets on Save As.
[host-native-replay.json](host-native-replay.json) repeats these checks from
the final OpenCVDataflowGUI example location, including all thirteen exports.

[baseline-comparison.json](baseline-comparison.json) compares the prior matcher
core with these frozen expected values. All seven default overlays match
exactly; the largest numeric difference is about `1.75e-7` (tolerance `1e-5`).
The full actual-wheel suite passed **114 tests and 46 subtests**. These are
programmatic Qt offscreen checks, not a hands-on operator acceptance record.
[wheel-verification.json](wheel-verification.json) pins the wheel to its tested
source. The checksum is checked by the loader.

From an SDK-capable environment, with no other PatternMatch package imported:

```powershell
python verify_examples.py . C:\path\to\new-verification --native
python build_examples.py C:\path\to\existing\pattern_matching C:\path\to\new-examples
```

Both output folders must be new. The builder reuses the existing sibling
`asymmetric_brackets` and `visa_pcb2` public assets; it does not retrain or tune
them. It copies the published baseline/replay evidence for reference; the fresh
builder produces its own `native-verification.json` and `expected.json`, and
the verifier produces a separate `checks.json` receipt. Full behavior,
retention and export details: [guide](COMPLETED_REVIEW.md).

## Attribution

Real VisA images and their derived reviews are **CC BY 4.0**, by Yang Zou,
Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer. Generated bracket
images, controls and authored code use the supplied MIT license. See
[SOURCES.md](SOURCES.md) for exact image members, preparation and source links.
