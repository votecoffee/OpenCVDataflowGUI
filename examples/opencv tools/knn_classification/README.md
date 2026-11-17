# Native HOG and K-nearest image classification

These nine projects use original synthetic stripe images and native OpenCV HOG
descriptors. Training and query images have disjoint pixel hashes. No optional ML
packages, external models or downloads are needed. Open a project and use
**Run All** to process its finite image list.

| Project | Expected result |
| --- | --- |
| `train.json` | Batch six HOG vectors and save a new `trained-classifier.json` with two orientation classes. |
| `train_normalized.json` | Same training workflow with per-row L2 normalization; save `trained-normalized.json`. |
| `classify.json` | Load the bundled model; both held-out images pass as vertical/horizontal within a 0.5 distance cutoff. |
| `single_neighbor.json` | Override stored K=3 with K=1; recognize both query images. |
| `normalized.json` | Apply the saved L2 model to both unnormalized incoming HOG vectors. |
| `reject_diagonal.json` | Fail as unknown: an unseen diagonal pattern exceeds the distance cutoff. Raw candidate/neighbors remain visible; scalar Label emits nothing. |
| `vote_tie.json` | Fail as unknown: K=6 produces three votes per class; default tie rejection withholds recognition. |
| `accepted_classes.json` | Vertical passes; horizontal is recognized but fails the inspection's allowed-class rule. |
| `wrong_contract.json` | Deliberate error without predictions: signed gradients change HOG preprocessing at the same descriptor length. |

Training saves new files and preserves existing ones. Choose a new destination
before another training run. To inspect a freshly trained model, change a
prediction project's loader to that file. The bundled model assets remain intact;
Save As carries the selected sealed model with the project.

View recognized Labels, Candidate labels, neighbor records and vote counts in
the block Results tab. Distances are Euclidean in the stored feature space;
the native squared distances are also reported. Vote fraction is agreement,
not a calibrated probability. See [full controls](../../../docs/knn-classification.md).

The Image output preserves the query pixels. Drawable annotations are composed
before the filmstrip image is chosen. Lean omits optional drawing work. These
small fixtures demonstrate the tool contracts; they establish no production
classification or OCR accuracy. The 0.5 cutoff is specific to this controlled
CPU fixture and requires qualification on another dataset/build.

`truth.json` records the declared orientation labels, pass/fail/error expectations
and every image/model hash. `build_example.py` copies eight pinned original stripe
images unchanged from [reusable clusters](../reusable_clusters/README.md), adds the
diagonal control, and rebuilds into a new destination without replacing assets.
Every project contains an attribution comment. See [SOURCES](SOURCES.md).
