# Image-descriptor PCA

Open a project and choose **Run all**. The included models work offline, without
plugins or downloads. Six training windows and three held-out windows have distinct
pixels; all images/models are original MIT controls. See [sources](SOURCES.md)
and [the block contract](../../../docs/descriptor-pca.md).

| Project | Demonstration |
| --- | --- |
| [train](train.json) | Six image → HOG → Batch → PCA → K-nearest training; saves both models |
| [train_l2](train_l2.json) | Same training with saved per-row L2 normalization |
| [classify](classify.json) | Reuse two-component PCA/classifier on two held-out images |
| [normalized](normalized.json) | Reuse the matching L2 PCA/classifier |
| [variance](variance.json) | Retain at least 95% training variance; this fixture selects one component |
| [reconstruct](reconstruct.json) | Reconstruct 324 descriptor values; output has a distinct feature identity |
| [reject_diagonal](reject_diagonal.json) | Deliberate reconstruction-error inspection failure |
| [wrong_descriptor](wrong_descriptor.json) | Changed signed HOG setting causes a contained contract error |
| [wrong_revision](wrong_revision.json) | A valid, same-size PCA space is rejected by the wrong classifier |
| [dimensions_only](dimensions_only.json) | Explicit unverified compatibility override, marked on output |

Training projects save `trained-pca-*.json` and `trained-classifier-*.json` beside
the project after six inputs. Existing files are preserved; choose new output
paths before repeating training. Inference projects use the frozen files in assets.
To try a retrained model, choose both newly saved matching files.

Two components replace 324 HOG values while retaining **99.9764%** of training
variance. Both known held-out orientations keep their raw nearest-neighbor labels.
Their feature reconstruction RMSE is **0.00407/0.01636**; the diagonal control has
**0.14491**, above the demonstration's **0.03** limit. The unguarded classifier
would still label the diagonal as horizontal. Variance and distances are not
probabilities or general accuracy guarantees. These tiny synthetic windows are
tool contract examples, not a production dataset or throughput benchmark.

The current image stays unchanged. Drawable labels appear on the filmstrip;
Debug retains numerical plots and Lean skips graphics while preserving decisions.
Projected feature contracts include the exact PCA revision, so changing descriptor
settings or mixing a different PCA with an old classifier is detected. L2
reconstruction retains normalized units and cannot restore original magnitude.

Run `benchmark.py` for the independent native HOG/NumPy SVD comparison recorded in
[benchmark.json](benchmark.json). `build_example.py` regenerates models and projects
from the nine unchanged public controls; it preserves an existing assets folder.
