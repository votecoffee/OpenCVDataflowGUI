# Native HOG and random forest classification

Open any project and use **Run All** over its finite image list. These controls
use original synthetic stripe images and base OpenCV; no optional ML dependencies
or external downloads are needed. Training and held-out images have distinct
pixel hashes.

| Project | Expected result |
| --- | --- |
| `train.json` | Batch six native HOG vectors and save a new forest. |
| `train_importance.json` | Also compute native permutation importance and its optional diagnostic plot. |
| `train_normalized.json` | Fit/save with per-row L2 normalization. |
| `train_weighted.json` | Apply explicit class priors and per-image sample weights. |
| `train_oob.json` | Use native OOB stopping with a hard maximum of 64 trees. |
| `classify.json` | Both held-out orientations pass the frozen model's 0.9 vote-agreement cutoff. |
| `normalized.json` | Apply a frozen L2 model to raw incoming HOG vectors. |
| `vote_threshold.json` | The fixed diagonal control fails the vote cutoff; recognized Label emits nothing, while candidates/votes remain available. |
| `accepted_classes.json` | Vertical passes; horizontal is recognized but fails the allowed-class rule. |
| `wrong_contract.json` | Deliberate error without predictions: signed HOG gradients change preprocessing at the same descriptor length. |

Each training project writes a new `trained-<project>.json` file and preserves
existing files. Choose another save path for a repeat training run. To inspect
a newly trained model, select it in a prediction project's loader. Sealed model
assets follow Save As.

Results include class mappings, tree counts, votes/fractions, margins and reasons.
Importance uses feature indices, not image coordinates. The primary Image is
unchanged; drawable annotations are composed before choosing the filmstrip image.
Lean omits optional plots/drawings.

The bundled ordinary forest returns votes `[60,4]` for the vertical query,
`[2,62]` for horizontal and `[12,52]` for diagonal. The first two clear 0.9; the
last does not. This qualifies a controlled policy fixture, not general unknown
detection or production classification accuracy. An unfamiliar image can receive
unanimous votes. Validate thresholds on the intended data/build.

`truth.json` pins every image/model and the declared expectations. Images are
copied unchanged from [K-nearest classification](../knn_classification/README.md).
`build_example.py` refuses existing assets. It prepares actual native forest
models in an isolated process with one OpenCV thread and recorded seeds 90113/90114.
Those fixture settings never change the GUI process; ordinary training remains
stochastic. Exact regeneration is qualified only on the stated Python/OpenCV
build. Every project contains an attribution comment. See [SOURCES](SOURCES.md)
and [full contracts](../../../docs/rtrees-classification.md).
