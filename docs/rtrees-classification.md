# Random forest image classification

The **OpenCV → Machine Learning → Classical classification** tools provide
`ml.rtrees_train`, `ml.rtrees_predict`, `ml.rtrees_save` and `ml.rtrees_load`.
They wrap native OpenCV RTrees for image-derived numeric descriptors and
measurements. No optional ML framework or downloaded model is needed.

A typical training graph is **Image files → HOG → Batch serial entries → Train
forest → Save forest**. Supply one class string per feature row in batch order.
Use **Load forest + HOG → Predict forest** for other captures. Models retain
class mappings, normalization, preprocessing contracts and native trees.

## Training controls

| Control | Meaning |
| --- | --- |
| Max depth | Maximum tree depth, with the root at zero; 1..16. |
| Min sample count | Native minimum samples for a node split. |
| Active variables | Features considered per split; 0 uses the native rounded square root of descriptor length. |
| Max trees | Hard tree limit, 1..256, retained with either termination mode. |
| Termination | Tree limit alone, or tree limit with native OOB error stopping. |
| OOB error | Positive target for optional stopping when native error is below it. |
| Calculate importance | Optional native permutation importance; adds substantial training cost. |
| Class priors | Positive relative class weights in the stored first-appearance class order. Empty uses native equal priors. |
| Sample weights | Optional positive weight per training row; a wired input overrides the parameter list. Empty uses equal weights. |
| Normalization | None or per-row L2, stored and reused on query descriptors. Zero vectors cannot use L2. |
| Labels | One nonempty string per row; a wired input overrides the parameter list. |

Training supports 2..64 classes with first-appearance string-to-ID mapping. It
accepts numeric rows or a batch of image descriptors. Batched preprocessing
contracts must agree. HOG supplies encoding, depth, window geometry, native version
and settings through its `feature_contract` payload. Other image descriptor
blocks can supply their own JSON contract. Plain rows without one remain supported;
the caller then owns preprocessing consistency.

Every training invocation fits a new model. It is stochastic and does not change
global RNG seeds or expose a misleading promise of repeatable retraining. Training
does not opt into parallel branch execution. The bundled examples use explicitly
seeded preparation in an isolated child process; inference always reuses the
stored trees. Different builds may produce different training results.

The baseline uses finite ordered numeric features. Categorical feature encoding,
missing-value surrogates, pruning/cross-validation options, regression and
incremental training are outside this classifier interface. Accuracy-only stopping
is excluded so training always has a hard tree cap.

## Importance and native statistics

Importance output contains one native permutation-importance value per feature,
or emits no event when calculation is disabled. An optional diagnostic plot shows
the sixteen highest-ranked feature indices. It is a plot, not an image-coordinate
overlay. Lean omits the plot while retaining requested numerical importance.
Zero importance is possible when training supplied no measured importance signal.

Report records actual tree/node counts, training row/class counts, weight digest,
model revision and whether OOB computation was requested. Native OOB statistics
are read from the saved forest because this pinned Python binding lacks an
exposed OOB getter. When OOB computation was not requested the statistic is null,
instead of presenting the native placeholder zero as measured error. Requested
OOB is the backend's internal, potentially weighted statistic; it does not establish
held-out accuracy or full coverage of every training row.

## Predictions and judgment

Outputs include recognized Labels, raw Candidate labels, per-class integer Votes,
Vote fractions, Accepted flags and Report. The native vote-table header is checked
against the stored mapping and each row must total the actual tree count. Candidate
labels must agree with the native majority. Report supplies winning fractions,
vote margins, ties, unknown reasons and the model/settings revision context.

| Control | Meaning |
| --- | --- |
| Contract policy | Exact preprocessing match by default; Dimensions only is an explicit override. Saved normalization still applies. |
| Min vote fraction | Inclusive minimum fraction of trees voting for the candidate. |
| Min vote margin | Inclusive minimum `(winning votes - runner-up votes) / actual trees`. |
| Reject vote ties | Enabled by default; withhold recognition on tied maximum votes. |
| Accepted labels | Optional trained classes allowed by the inspection. Empty accepts every recognized class. |

Vote agreement and margin are **not calibrated probabilities** or general novelty
detectors. An unseen image can receive unanimous votes. Evaluate suitable thresholds
on held-out data for the actual inspection.

Unknown rows return null in Labels and retain raw candidates/votes. Scalar Label
emits only for one known row, with no event for unknown or multiple rows. Unknown
results fail with `rtrees_unknown_class`; row reasons distinguish vote fraction,
margin or tie. A known disallowed class still emits its label and fails with
`rtrees_unaccepted_label`. All rows must be accepted for the tool to pass. These
are ordinary inspection outcomes, not final ticket judgments. Invalid inputs or
native exceptions use common per-block diagnostics and emit no results.

An optional Image input must accompany one descriptor and match its known
frame/ticket/recipe/pyramid context before native loading. Image pixels, encoding
and depth remain unchanged. Current descriptor context travels with the results.
Drawable class/vote annotations can be composed with **Overlay result graphics**
before choosing a filmstrip image. Lean skips optional drawings. Existing
failed-input admission and missing-connected-input gates prevent work/events and
last-input updates when execution is skipped.

## Model files and bounds

Portable models are JSON containing native FileStorage trees, explicit class
mapping, normalization, feature contract, training settings/summary and a SHA-256
revision. Training settings are retained separately because native serialization
does not preserve every setting. No pickle, native object or executable content
travels along connections. The checksum detects modification, not authorship.

Save preserves existing files. Load rejects duplicate keys/nonfinite JSON,
changed revisions, wrong OpenCV major versions and malformed native structure.
The supported native subset checks dimensions, ordered variables, class IDs,
priors, tree counts, preorder binary topology, split indices/thresholds and node
bounds before handing trees to the native reader. This is not an importer for
every external RTrees model format. Sealed loader assets copy with Save As;
JSON/XML preserve all controls. Recipes can swap model inputs or loader paths
while the feature contract continues to guard preprocessing.

Inputs allow 2,048 rows, 65,536 dimensions, two million finite float32-representable
values within ±1e15 and no float32 underflow. Priors/weights are in [1e-6,1e6].
Models allow 8 MiB, 256 trees, 50,000 nodes and a 16 KiB feature contract.
Training checks `rows × dimensions × max_trees`, multiplied by dimensions when
importance is enabled, against 64 million work units. Prediction checks
`query rows × model nodes` against the same conservative budget. These estimates
bound supported work; they are not runtime guarantees. Reduce descriptors, trees
or batch size for larger jobs.

See [ten original projects](<../examples/opencv tools/rtrees_classification/README.md>),
[K-nearest classification](knn-classification.md), [SVM classification](classical-ml.md)
and the [native RTrees API](https://docs.opencv.org/4.x/d0/d65/classcv_1_1ml_1_1RTrees.html).
