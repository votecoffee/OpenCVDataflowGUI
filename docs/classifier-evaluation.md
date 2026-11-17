# Image classifier evaluation

Evaluate one recognized class per image using **Results > Validation image sets…**,
the saved-project API or the command line. The existing validation runner substitutes
one root image source and processes the declared cases on separate program copies.
Production counters, previews and learned state in the open job are preserved.
This is the classifier portion of S2-23; [detector evaluation](detector-evaluation.md)
provides its source-box IoU matching and precision/recall/AP counterpart.

## Build and run a labeled set

1. Create an image set and select the source as in [production validation](production-validation.md).
2. On **Classifier evaluation**, enable **Evaluate classifier labels**, select a root
   classifier block and its recognized-label output. Native KNN, SVM and random forest
   blocks use `labels` with value path `0`. A custom scalar string output uses a blank path.
3. Enter each image's **Class label** and **Capture group**. A blank class is unlabeled:
   it remains in the report but contributes no accuracy denominator. Keep correlated
   frames, crops or views from the same acquisition in one group.
4. Set minimum accuracy and coverage. Advanced settings expose class display order,
   declared training groups, pinned image SHA-256 and optional scalar score sweeps.
5. Save the set/project, run the image set and inspect **Classifier report**. The
   table has true classes down the rows and predicted classes across the columns,
   followed by Rejected, Missing and Error. Per-class support, precision, recall and
   F1 appear beneath it. Export retains the full report, records, settings and sweep.

Use `unknown` for the ordinary pass/fail expectation when testing recognition alone.
An incorrect recognized class can still be a passing inspection result in a classifier
whose accepted-class list allows it. Class recognition and inspection judgment are
independent; combined validation passes only when both configured checks pass.

```powershell
opencv-dataflow-api run-validation "examples/opencv tools/classifier_evaluation/correct.json" --set heldout --report new-report.json --summary
```

The report destination must be new. Exit status is 0 for a passing report, 1 for
failed validation and 2 for an invalid command/configuration. The response uses the
usual versioned JSON envelope. `--profile lean` suppresses optional diagnostics while
retaining numeric evaluation. Ordinary `python -m opencv_dataflow.validation` also
supports classifier-enabled sets. Trusted program code and lifecycle hooks execute
under the same rules as an ordinary validation run; output blocks are skipped by default.

```python
from opencv_dataflow.assistant_api import ProjectSession

with ProjectSession.open("program.json") as project:
    report = project.run_validation("heldout")
    print(report["classification"]["summary"])
```

The API accepts a set ID or unique name and optional `cancelled` and `progress`
callbacks. Cancellation retains observed rows, marks the report incomplete and adds
unobserved expected IDs as Missing in classifier metrics. An incomplete run fails.

## Saved fields and current-event contract

An optional set `classification` object selects `block`, `output`, `value_path`,
`classes`, `min_accuracy`, `min_coverage`, `training_groups`, `score_output`,
`score_path`, `score_kind`, `thresholds` and `score_direction`. Defaults are recognized
`labels[0]`, 100% minimum accuracy/coverage, inferred class order and no sweep.
Each case can add `class_label`, `capture_group` and `image_sha256` to its existing
ID/path/expectation. Existing projects without classifier configuration retain their
old summary contract. JSON/XML and portable Save As retain these fields; validation
image paths are rebased with the project.

The selected port must emit exactly one current event matching the validation image's
ticket. Default list lookup additionally requires exactly one label. A string is
recognized; `None` is Rejected. A tool that suppresses its unknown-label event should
expose a separate nullable recognized-label output for evaluation; absent events are
Missing. Stale last results never count. A failed block, invalid output/path, unrelated
ticket, malformed score or failed image preparation is Error. Arbitrary expressions
are not evaluated: dotted paths traverse dictionary keys or list indices only.

An optional scalar score must also have a current matching ticket. Examples are
KNN `report` / `vote_fractions.0` or forest `report` / `winning_vote_fractions.0`.
These are agreement fractions, not calibrated probabilities. Give custom scores an
explicit meaning in `score_kind`. Sweeps require comparable finite scores and keep
`>= cutoff` for `higher` or `<= cutoff` for `lower`. They can further reject existing
recognized predictions, but cannot recover candidates already rejected upstream.
Cutoffs should be selected on validation groups, then checked on untouched test groups.

## Metrics and failure accounting

- Accuracy = correct / all labeled images, including rejection, missing and errors.
- Coverage = recognized / all labeled images, including wrong recognized labels.
- Selective accuracy = correct / recognized labeled images; null when none is recognized.
- Per-class precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
  Rejected/missing/error rows add false negatives for their true class.
- Undefined ratios are JSON null and displayed as **unscored**. Macro means omit
  undefined values and report their contributing class counts. Balanced accuracy is
  macro recall; weighted F1 uses true-class support.

Unlabeled rows have separate counts. Predicted labels absent from the declared class
order are appended and retained. Zero labeled cases, any missing/error row or incomplete
execution fails even when minimum cutoffs are zero. Confusion counts include all labeled
cases, not just successful classifications. A standalone record list cannot prove that
every expected image was seen: supply `expected_cases` to verify IDs/truth/groups and
materialize missing rows. The saved validation runner always supplies the full set.
Limits are 10,000 rows, 128 classes, 128 cutoffs, 16 KiB per record and 16 MiB of records.

Definitions follow the conventional [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
and [precision/recall/F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
formulas. No scikit-learn dependency is introduced.

## Reusable graph blocks

**ML > Evaluation > Split labeled images by capture group** (`ml.split_image_groups`)
accepts metadata records with `id`, `path`, `class_label`, explicit `capture_group`
and optional `image_sha256`. It outputs train, validation and test lists plus a split
report. Fractions apply to whole groups, with deterministic seeded SHA-256 ordering,
nearest-count rounding and at least one group for each positive fraction. Input order
does not change assignments. Training must retain at least one group and, by default,
every class. Change the group design/seed/fractions when a class is absent.

This plans dataset membership; it does not load files, train a classifier, stratify
classes, cross-validate or create validation-set definitions automatically. Metadata
paths are opaque strings, relative to the consuming dataset's chosen root. The same
path or supplied hash cannot cross groups or labels, but those hash claims are not
file verification (`image_hashes_verified=false`). The validation runner separately
hashes the actual frozen image bytes and rejects a pinned-hash mismatch before execution.
Group declarations also cannot prove the training provenance of an arbitrary model.
The set rejects overlap with declared training groups; choose truthful groups and
preserve a separate training manifest. Group fractions have the same intent as
[group shuffle splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html),
using a different deterministic assignment algorithm.

**Evaluate image class predictions** (`ml.evaluate_classification`) accepts record
lists, a FlowGroup of records or a validation report, and emits the same metrics plus
an optional confusion plot. It can also keep fixed records in its parameters. The plot
uses its own diagnostic coordinate frame and displays the first 16 classes by index;
the data report and validation window retain all classes/names. Graphics-off/Lean
skips the plot without changing numeric results. This block does not collect a stream
automatically: use the validation runner or a control-flow batch/API collector.

Try [the six public projects](<../examples/opencv tools/classifier_evaluation/README.md>).
They use original, disjoint stripe captures with native HOG/KNN. Independent count
references, explicit error controls and persistence checks establish software behavior;
they do not establish inspection accuracy on production parts. Detector one-to-one
IoU matching and AP/PR are now available in the companion detector workflow.
Folds/bootstrap confidence and representative dataset qualification remain future work.
