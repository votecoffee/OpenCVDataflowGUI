# Review a completed classification

SimpleClassify 0.3.2 adds **Review completed classification…** on the inspector's
Results page, independent of **Train image classifier…**.

1. Run in Debug with input/output previews retained. Select the classifier and
   open Results review. A paired snapshot freezes the exact completed input,
   executed manifest/model bytes, settings and all candidate scores.
2. Read the winning label/percentage, runner-up, percentage-point gap and
   confusion. The table keeps every class, full Unicode labels and original
   score ordering. The unchanged raster assessment shows at most 20 classes.
3. Switch between exact consumed input and ranked assessment. Scroll to zoom,
   drag to pan, or choose Fit / 100%. The input keeps original channels/depth;
   the assessment uses a separate display grid and a converted BGR thumbnail.
4. Select a candidate and export to a new folder. Closing leaves assignments,
   staged training, inspector drafts and completed outputs unchanged.
5. Editor publications mark the view frozen. Capture latest explicitly replaces
   it. Active/error/unretained capture keeps a previous frame clearly labeled;
   before any completion, export is disabled.

The probabilities output sums to 100%. Both profiles are **uncalibrated**:
Color and layout uses normalized tree votes; Surface texture uses normalized
nearest-exemplar distance evidence. Confusion is `100 * runner-up / winner`:
100% is a tie; 0% means no runner-up evidence. It is not an error probability.
Even an unrelated image receives a winner; there is no inferred unknown class,
acceptance threshold, per-pixel localization or defect judgment. Training health
belongs to the saved held-out training report, not this image's confidence.

Execution retains a small Debug-only completion record and immutable model bytes
already read once at model load. It adds no source-image copy, model serialization
or second inference. Explicit capture privately freezes native arrays and copies
ticket/spatial/pyramid metadata. Completion IDs prove software pairing, not
physical exposure synchronization. Lean omits the extra record.

Exports contain `review.json`, exact native `consumed_input.npy`, the classifier's
converted `classifier_input.npy`, input/assessment PNGs, and the original
`executed_recipe/manifest.json` plus `forest.yml` or `surface.npz`. The manifest's
training-image paths are historical provenance; exports do not read/copy those
images and are not portable training bundles. **Save As** preserves the complete
model and labeled images separately, with unchanged bytes and assignment paths.
Existing files/folders are never overwritten by review export.

Calibration/unknown rejection, cross-frame comparisons, training-example
explanations, independent dataset qualification and hands-on operator acceptance
remain future work. Published examples keep their original model settings and
include incorrect predictions rather than tuning them away.
