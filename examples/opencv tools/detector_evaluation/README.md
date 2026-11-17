# Detector evaluation

Eight native projects demonstrate labeled boxes, class-aware one-to-one IoU matching,
precision/recall/AP and failure reporting. Open a project with **File > Open**, then
use **Results > Validation image sets > Run image set**. No plugin/download is needed.

| Project | Independent expected result |
| --- | --- |
| [correct.json](correct.json) | Three true positives, no false positives/misses, mAP 1. Includes a clean negative image; validation passes. |
| [duplicates.json](duplicates.json) | Suppression off: 3 TP, 1 duplicate FP, no misses. Red AP 5/6; mAP 11/12. Validation fails precision. |
| [low_recall.json](low_recall.json) | Upstream score cutoff .7 removes two objects: 1 TP, 2 misses, mAP .25. Validation fails recall. |
| [localization.json](localization.json) | A declared 10-pixel label offset gives IoU 2/3. At the .75 operating IoU: 2 TP, 1 FP, 1 miss, mAP .625. The .5 report still matches all objects. |
| [wrong_class.json](wrong_class.json) | One explicit label mismatch: 2 TP, 1 FP, 1 miss, mAP .5. Validation fails. |
| [negative_false_positives.json](negative_false_positives.json) | Both images declared fully labeled negatives. All three retained color-image proposals are FP; mAP is unscored because there is no truth. Validation fails. |
| [decoder_error.json](decoder_error.json) | Unavailable tensor name: both images Error, three missed objects, validation fails. This is not a clean empty prediction. |
| [metrics.json](metrics.json) | Run once; select Evaluate image detections for the stored actual duplicate-control records, report and PR plot. |

In the validation window, **Detector evaluation** selects the detector and limits.
Double-click **Object labels** to review/edit IDs, classes and source-pixel XYWH
boxes. Empty labeled images are negatives; unlabeled images are excluded from object
metrics. The **Detector report** has class, image, score-sweep and IoU-cutoff tabs.
Export retains complete match IDs/IoUs/curves. Advanced controls expose AP interpolation
and cutoffs. All ordinary pass/fail expectations are Unknown to isolate detection.

```powershell
opencv-dataflow-api run-validation "examples/opencv tools/detector_evaluation/correct.json" --set objects --report new-detector-report.json --summary
```

Use a new report filename. Deliberately incorrect controls return exit status 1.
The app/session's processing history and production totals remain unchanged.
[Contracts](../../../docs/detector-evaluation.md) explain event/grid requirements,
score ties, denominators, AP interpolation and the limits of this local evaluator.

Images/model are copied unchanged from [DNN detection](../dnn_detection/README.md).
The original MIT ONNX fixture has fixed network-space proposals with scores driven
by image mean color. It is not a trained detector. The projects explicitly select
OpenCV 5's new CPU engine. Ground truth is derived independently
from declared 64×64 network boxes and exact stretch scaling to 160×80 source images;
it is never copied from decoder output. These controlled examples qualify software
behavior, not real part-recognition accuracy or official COCO benchmark performance.

[truth.json](truth.json) pins assets and numeric references;
[SOURCES.md](SOURCES.md) records origin/license. `build_example.py` regenerates all
fixtures exactly in a fresh folder and preserves existing assets. Every project
contains an attribution/comment block.
