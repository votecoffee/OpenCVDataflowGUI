# Image-feature outlier inspection

Open a project and choose **Run all**. Included models work offline without plugins
or downloads. Nine known-good training windows and separate held-out queries are
original MIT controls. See [sources](SOURCES.md) and
[the block contract](../../../docs/feature-outliers.md).

| Project | Demonstration |
| --- | --- |
| [train](train.json) | Nine images → HOG → Batch → covariance → Save model |
| [train_l2](train_l2.json) | Same training with saved per-row L2 normalization |
| [known_set](known_set.json) | Three held-out normal/lighting variants all pass |
| [crack](crack.json) | Larger dark crack fails the distance check |
| [missing_arc](missing_arc.json) | Removed outer arc fails the distance check |
| [low_contrast_miss](low_contrast_miss.json) | Subtle one-pixel edit passes: a recorded miss |
| [empty_view](empty_view.json) | Empty raw-HOG window fails |
| [normalized_set](normalized_set.json) | L2 model accepts three known views and rejects the crack |
| [diagnostic_only](diagnostic_only.json) | Reports the crack's outlier row with judgment disabled |
| [wrong_descriptor](wrong_descriptor.json) | Changed signed HOG setting causes a contained error |
| [singular_no_ridge](singular_no_ridge.json) | Omitting necessary regularization causes a contained error |
| [pca_known](pca_known.json) | Matching saved PCA/covariance revisions accept a known image |
| [pca_wrong_revision](pca_wrong_revision.json) | Different valid same-size PCA space is rejected |

Training saves `trained-covariance-none.json` or `trained-covariance-l2.json` beside
the project after nine inputs. Existing files are preserved; change the output
path before repeating training. Inference uses frozen models in assets. Select
the newly saved matching covariance to try a retrained model.

The examples use four-bin HOG (144 values), sample covariance, distance limit **4**
and Ridge **0.001** (raw) or **0.0001** (L2). Raw distances are **2.41–2.53** for the
known images, **17.09** for the crack, **29.63** for the missing arc and **92.35**
for the empty view. The subtle edit scores **2.53 and passes**. Distances are not
probabilities or a general accuracy guarantee. A zero descriptor cannot be L2
normalized; the benchmark records that unsupported case explicitly.

The image stays unchanged; a drawable full-window judgment outline reaches the
filmstrip. It does not locate defects. Debug retains plots and drawings; Lean
skips graphics while preserving decisions. Wrong settings/revisions are checked
before native scoring. The PCA cases demonstrate identity checks, not preserved
defect sensitivity after compression.

Run `benchmark.py` for the independent native HOG/NumPy comparison recorded in
[benchmark.json](benchmark.json). `build_example.py` deterministically regenerates
the controls, models and thirteen projects; it preserves an existing assets folder.
