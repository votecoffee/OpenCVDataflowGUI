# HOG sliding-window detection

Open any project here in the application and run it. Start with `multiscale.json`:
the source scene contains two bracket-like parts and a ring distractor. Both
known boxes should be drawn, with raw native margin labels around 1.0008.
All images, annotations and custom models are original synthetic MIT fixtures;
see [sources](SOURCES.md) and the [block contract](../../../docs/hog-detection.md).

| Project | Demonstrates |
| --- | --- |
| `multiscale.json` | Three search scales, threshold 0.5 and IoU suppression retain two exact boxes. |
| `single_scale.json` | Same scene and threshold with one search level; two exact boxes. |
| `raw_windows.json` | Suppression disabled: six overlapping multiscale candidates. |
| `low_threshold.json` | Threshold zero: six retained results, including four extras. |
| `selection.json` | Left-side rectangular selection retains only the left part. |
| `high_depth.json` | uint16 source retains original depth and the same source-coordinate boxes. |
| `count_fail.json` | Two detections exceed the configured maximum one: valid inspection failure. |
| `blank.json` | No detections: valid minimum-count failure. |
| `wrong_model.json` | Intentionally changed model revision is rejected; no detector output event. |
| `nonlinear_svm.json` | An RBF SVM is rejected by the linear detector converter. |
| `default_people.json` | Blank-image integration check for the native 64×128 people preset. |
| `daimler_people.json` | Blank-image integration check for the native 48×96 Daimler preset. |
| `train.json` | Sixteen image windows → HOG → batch → binary linear SVM → detector → save. |

`train.json` saves `trained-detector.json` without overwriting an existing file.
Choose a new destination to repeat a save. Detection projects use the checked-in
`assets/detector.json`, so training is optional. `truth.json` pins all 23 image/
model assets and expected states. `benchmark.json` records an independent 501-window
descriptor-dot-product search and scalar IoU suppression; `benchmark.py` recomputes
it without importing the application's detector. Native and independent boxes
agree with margin differences below 3.3e-8 on this fixture.

`build_example.py` regenerates assets and saved projects into an empty destination;
it refuses to replace existing assets. Training uses four noise variants each of
part/ring/cross/blank windows; the query uses a separate noise seed. These controlled
shapes verify integration and known coordinates, not field accuracy or calibrated
confidence. Lowering the threshold deliberately reveals extra detections. Stock
people models are checked only on blank inputs. No hardware or model download is
required. On-canvas comments include attribution and scope limits.
