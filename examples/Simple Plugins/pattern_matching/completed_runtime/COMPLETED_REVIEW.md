# Inspect the exact completed pattern match

Dev2 adds **Review completed pattern match…** in OpenCVDataflowGUI's Results
inspector for both the saved-file and wired-model matchers. It is separate from
training: opening the review never loads the configured recipe file, edits the
training draft or runs another search.

## Workflow

1. Execute the graph in **Debug**, retaining processed inputs and output values.
2. Select the matcher, open **Results**, then **Review completed pattern match…**.
3. Inspect returned instances or retained rejected candidates by reason. Select a
   row for the exact pose, found/missing/unmeasured edges and captured limits.
   Zoom/pan, use **100%**, focus the selected footprint or show all returned poses.
4. The **Exact consumed input** tab retains original pixels, depth and encoding.
   The fit tab is the matcher's working grayscale grid. Table coordinates map to
   the consumed input grid; upstream spatial/pyramid transforms stay in metadata.
5. Toggle fit layers and inspection-area shading. Dark pixels are excluded by the
   captured ROI/mask; green is found, red missing, amber outside or unmeasured.
6. **Export frozen review…** writes a new folder. Close review leaves configuration,
   training and the completed result unchanged.

The captured frame is frozen. A newer editor publication only marks it as old;
**Capture latest completed run** explicitly replaces it. An active run, missing
retention or processing exception refuses a new capture, leaving an existing
frozen frame clearly labeled. A completed failed inspection is reviewable;
zero matches does not invent a pose or reuse an earlier successful fit.

## Evidence and scope

An immutable per-completion record retains the actual executed model and cached
poses/correspondences/rejection counts. The host atomically snapshots matching
input/output packets, and a namespaced output-payload token binds them to that
record. IDs/counters identify software completion/copies, not physical exposures.
Each input's own tickets, timestamps, source frame and pyramid mapping are kept.

Changed/deleted model files, changed training drafts and newer upstream images
cannot silently replace the captured model/frame. Wired models remain the exact
packet used at execution. Worker/display graph transfer preserves the immutable
record through the standard runtime-copy hook. No source-image copy or new
correspondence fitting is added during execution; source pixels are privately
copied only on explicit review capture by the host API.

The record is retained only when both input and output previews are retained.
Lean execution skips it. Existing rendering/rejection settings still decide
whether edge correspondences and rejected candidates are captured. If those
were disabled, the review reports them unavailable instead of recomputing them.
Scores/limits and all accepted/rejected numerical outputs remain unchanged;
the execution binding lives in wire metadata, not the primary JSON values.

## Export

Selected review exports include the working-grid `fit.png`, display-only
`assessment.png` and `review.json`, with exact completed-frame provenance added.
Overview/no-match exports retain a fit image and explicit null selected candidate.
Every export also includes:

- `consumed_input.npy`: exact original consumed array/depth/channels, no pickle.
- `working.png`: grayscale working pixels used by the matcher.
- `inspected_mask.png` and `inspected.png`: captured ROI/mask coverage.
- `executed_recipe.npz`: re-encoded immutable executed model/search snapshot.

The exported recipe's bytes may differ from an original archive pin; both the
original pinned hash/model ID and exported archive hash are recorded. Review
exports refuse an existing folder and never modify the active training asset.

The fit/assessment grid is distinct from source pixels. Neither a score nor a
colored edge is a new accuracy gate or a probability. Rejected candidates stay
diagnostics; the source image is not a ground-truth pose annotation.
