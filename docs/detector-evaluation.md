# Image detector evaluation

Evaluate class-aware, axis-aligned source-image boxes with **Results > Validation
image sets…**, `ProjectSession.run_validation()` or `opencv-dataflow-api run-validation`.
This extends the existing [classifier evaluation](classifier-evaluation.md) workflow
and completes the bounded classifier/detector baseline in S2-23. No model downloads
or additional dependencies are needed.

## Label and run images

1. Create a [validation image set](production-validation.md) and choose its root
   image source. Set ordinary pass/fail expectations to **Unknown** to evaluate
   detection independently of inspection judgments.
2. On **Detector evaluation**, enable **Evaluate detector object boxes**, choose
   a root detector block and its collection output. Native `ml.decode_detections`
   uses `collection`. Set the matching IoU, operating score cutoff and minimum
   precision/recall.
3. Double-click each image's **Object labels** cell. Enter its source-image width
   and height, then add object IDs, exact class labels and left/top/width/height.
   The editor initializes size from saved labels or the file header when available.
   It validates finite positive boxes inside the declared image bounds.
4. Turn **Unlabeled image** off for a fully labeled image. With no objects, this
   explicitly declares a negative image: any detection counts as a false positive.
   Unlabeled images remain visible but do not contribute object metrics.
5. Save the set/project, run it and inspect **Detector report**. Its tabs show
   classes, per-image status/counts, score sweeps and IoU cutoffs. Hover a failing
   image's status for its captured diagnostic. Export retains complete records,
   matched IDs, matched IoUs, missed IDs and precision/recall curves.

Advanced settings expose class display order, declared training capture groups,
extra IoU cutoffs, score sweeps and AP interpolation. Capture-group and pinned-hash
columns are shared with classifier evaluation. Classifier and detector checks can
both be enabled; all configured checks must pass. Editing labels stages them in
the set; Save set and Save project retain them. Cancel preserves the original labels.

```powershell
opencv-dataflow-api run-validation "examples/opencv tools/detector_evaluation/correct.json" --set objects --report new-detector-report.json --summary
```

The report must be a new file. Exit 0 means passing validation, 1 means a failed or
incomplete report, and 2 means an invalid command/configuration. The versioned JSON
response's `data.detection` contains the compact summary with `--summary`.
`--profile lean` suppresses optional graphics without changing numerical evaluation.
Python callers use `project.run_validation("objects")`; optional progress/cancellation
callbacks are the same as for ordinary image-set validation. The run uses isolated
program copies, leaving the open job/session's processing state and production
counters intact. Trusted program processing/lifecycle behavior still executes;
output blocks are skipped by default.

## Coordinate and event contracts

Each case may contain this optional field:

```json
{
  "detection_truth": {
    "size": [160, 80],
    "objects": [{"id": "part-1", "label": "part", "bbox": [25, 20, 50, 20]}]
  }
}
```

Boxes use continuous **source pixel edges**, XYWH, without an extra inclusive
pixel. Truth must be inside its declared image; predictions retain their supplied
extents and are not silently clipped by evaluation. Use the detector's border
policy and labels consistently. Class labels are exact strings, including case.
Object IDs must be unique strings per image/side; image IDs must be unique per set.
Missing/null truth means unlabeled; `objects: []` means fully labeled negative.

A set's optional `detection` configuration selects `block`, `output`, `classes`,
`training_groups` and the metric parameters below. It requires exactly one current
collection event and one source-image event with the validation ticket. A valid
empty box collection is an observed empty prediction. Absent emissions are Missing;
exceptions, malformed data, unrelated tickets/grids, wrong image sizes or failed
image preparation are Error. Retained last results never count as new detections.

The collection must be a zero-origin `vision_collection_v1` **boxes** collection.
Every item supplies string `label`, finite scalar `score` and `geometry.bbox`;
native item IDs are converted to strings. Collection, packet and source payloads
must identify the same known spatial frame and current image grid. Sizes must match
the actual source image and truth. Rotated geometry is rejected: converting an
oriented box to an axis-aligned envelope is a different evaluation contract.
Crop/ROI collections must be explicitly mapped back to the source grid first.

The validation runner checks optional image SHA-256 against the frozen actual
bytes before processing. Declared training groups cannot overlap evaluation groups,
but group declarations alone do not prove an arbitrary model's training provenance.
JSON/XML preserve annotations/configuration; portable Save As rebases image paths
and uses the existing model-asset transport.

## Matching, operating metrics and AP

For each IoU cutoff, predictions are processed per image in descending score order.
Each prediction claims the unused same-class truth with the largest IoU meeting
`>= cutoff`. Truth-ID order resolves equal overlaps; prediction-ID order resolves
equal scores within an image. One truth can be matched once. This is score-greedy
matching, rather than a maximum-cardinality assignment. A wrong class contributes
a false positive to its predicted class and leaves the true class missed. Duplicate
predictions count as false positives.

`score_threshold` keeps predictions with score **>=** cutoff for operating metrics:

- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN), where FN includes objects in missing/error/unprocessed images.
- F1 = 2TP / (2TP + FP + FN).

Undefined ratios are JSON null and shown as **unscored**. A clean fully labeled
negative-only set can pass without numeric precision/recall; any false positive
fails the default precision limit. At least one labeled image is required. Missing,
error or incomplete execution always fails, including on negative/unlabeled images.
Ordinary execution/cleanup failures also fail the combined validation report.

Per-class curves accumulate predictions across labeled images by descending score,
merging equal-score predictions into a single cutoff point. AP uses the monotonically
nonincreasing precision envelope over these points. Choose **all_points** (area under
the envelope over attained recall; default) or **101_points** (mean precision at
recall 0, .01, …, 1; unattained recall contributes zero). No truth for a class gives
null AP; its false positives still count in operating precision. mAP averages AP
over classes with truth. Reports state that contributing class count explicitly.

AP uses all predictions supplied to evaluation, independently of the operating
score cutoff. `score_thresholds` produces operating precision/recall sweeps using
the same ranked matches. `iou_thresholds` reports each additional IoU independently;
`mean_ap_over_iou` averages mAP over the distinct main/additional IoUs. Sweeps cannot
restore detections already removed by upstream thresholds, selections or NMS.
Detector scores are not assumed to be calibrated probabilities.

These are explicit local inspection metrics, not the complete official COCO
benchmark protocol. COCO also specifies crowd/ignore behavior, area ranges and
detection-count policies; its [evaluation reference](https://cocodataset.org/dataset/detection-eval.htm)
and [official evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
describe those details. This baseline has no crowd/ignore annotations, segmentation
IoU, oriented boxes or benchmark-format importer. Equal-score grouping is explicit
and can differ from evaluators that order tied predictions individually.

## Reusable evaluation block and bounds

**Evaluate image detections** (`ml.evaluate_detections`, Custom > ML > Evaluation)
accepts record lists, a FlowGroup of records or a validation report. Each record
has `id`, `size`, `truth` (the object above or null), `predictions` (ID/label/XYWH/score),
and `status` (`predicted`, `missing`, `error`). Optional `group` records capture identity.
An unobserved/error image may have null `size`; no substitute dimensions are invented.
`expected_cases` verifies image IDs/truth/groups and materializes absent expected
images as Missing; without it a record list cannot prove every image was observed.
The saved-set runner always supplies the complete expected set.

The block emits a full report and optional PR plot. The plot has its own diagnostic
coordinate frame, a legend for the first 16 classes and full Unicode names in the
report/table. Its node judgment follows minimum precision/recall and execution status;
its optional score is operating F1, or the clean-negative judgment when F1 is undefined.
Graphics-off/Lean skips the plot without changing counts. Use a validation run or
batch/API collector to gather serial events; this block does not accumulate a stream.

Limits: 10,000 image records, 128 classes, 256 objects per image/side, 20,000 total
truth/prediction objects, 2,000,000 image-local IoU pairs and 16 MiB of input records.
There are up to ten additional IoUs and 32 score sweep cutoffs. Image sizes follow
the 16,384-per-axis/64M-pixel collection budget; finite score/coordinate ranges and
distinct IDs/cutoffs are checked. Exceeding a budget fails before overlap allocation.

Try [the eight public projects](<../examples/opencv tools/detector_evaluation/README.md>).
They use a pinned original ONNX wiring fixture with fixed proposals and image-dependent
scores, not a trained detector or production accuracy benchmark. Independent geometry,
counts, AP, cancellation, identity and persistence checks qualify the software baseline.

The projects explicitly select OpenCV 5's new CPU engine. CPU inference avoids
unrequested accelerator discovery, including on the validation worker; cold-process
worker shutdown is checked separately from successful per-image assertions.
