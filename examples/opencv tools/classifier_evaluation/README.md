# Image classifier evaluation

Six native projects demonstrate dataset groups, recognized-class evaluation and
confusion reporting. All use original MIT stripe images; no plugin or model download
is needed. Open a project with **File > Open**.

| Project | Action and expected result |
| --- | --- |
| [correct.json](correct.json) | Results > Validation image sets > Run image set: both held-out orientations correct; accuracy/coverage 100%, report passes. |
| [rejected.json](rejected.json) | Same run: a strict distance cutoff rejects both images; accuracy/coverage 0%, no recognized classes, report fails. |
| [mislabelled_training.json](mislabelled_training.json) | Native model trained with intentionally swapped labels; held-out truth remains correct. Both predictions wrong, accuracy 0%, coverage 100%, off-diagonal confusion, report fails. |
| [wrong_contract.json](wrong_contract.json) | Signed HOG conflicts with the saved preprocessing contract. Both rows Error, report fails; retained predictions are never reused. |
| [metrics.json](metrics.json) | Run once; select Evaluate image class predictions to inspect Report/Plot from actual correct classifier records. |
| [split_groups.json](split_groups.json) | Run once; inspect train/validation/test metadata and group assignments. Each of four capture groups contains a vertical/horizontal pair and stays together. |

Classifier settings are optional and hidden for ordinary pass/fail image sets.
Class labels, capture groups and expected hashes are saved with these projects. The
classifier report has true-class rows, predicted-class columns and separate
Rejected/Missing/Error columns, plus per-class precision/recall/F1. Advanced settings
contain native KNN vote-fraction sweeps at 0, 0.8 and 1; agreement is not probability.
All ordinary pass/fail expectations are unknown so this fixture isolates recognition.

From the repository root, with the application environment active:

```powershell
opencv-dataflow-api run-validation "examples/opencv tools/classifier_evaluation/correct.json" --set heldout --report new-classifier-report.json --summary
```

The report must be a new file. Incorrect controls deliberately return status 1.
The API [contracts](../../../docs/classifier-evaluation.md) explain denominators,
partial/cancelled reports, missing events and settings. Validation copies do not
change production totals or the open job's runtime state.

Images/model are copied byte-for-byte from [KNN classification](../knn_classification/README.md).
Training images at three stripe positions differ from the held-out capture. The
mislabelled control is separately trained from the same six HOG rows with swapped
training labels. Group splitting is a metadata planner, not an automatic trainer;
its hash claims are not file verification. Validation independently verifies actual
image bytes before processing. These small synthetic fixtures establish software
behavior, not industrial classification accuracy.

[truth.json](truth.json) pins every image/model and expected metrics;
[SOURCES.md](SOURCES.md) records origin/license. `build_example.py` recreates the
fixtures in a fresh copy and preserves any existing assets. Every project includes
an on-canvas attribution/comment block.
