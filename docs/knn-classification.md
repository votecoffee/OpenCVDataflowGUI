# K-nearest image classification

The **OpenCV → Machine Learning → Classical classification** tools provide
`ml.knn_train`, `ml.knn_predict`, `ml.knn_save` and `ml.knn_load`. They classify
image-derived descriptors using native OpenCV KNearest with exact brute-force
neighbor search. The base OpenCV installation is sufficient; no external model
or training framework is needed.

Train with **Image files → HOG → Batch serial entries → Train K-nearest → Save
classifier**. Provide one class string per descriptor in the same order. A wired
Labels input overrides the trainer's label list. Load the model in another graph
and connect **HOG / Descriptor → Predict / Samples**. An optional Image input
provides unchanged review pixels and a drawable class/distance/vote label. Compose
Result graphics with **Overlay result graphics** before selecting the filmstrip
image. Lean omits drawings. Numerical results and judgments remain available.

## Training and preprocessing

Training accepts numeric rows or batches of descriptor vectors. Models support
1..64 classes, using first-appearance order for the stored string-to-ID mapping.
The native classifier retains training samples rather than learning compressed
weights. Each execution creates its own native classifier; no native handle or
mutable training cache travels along connections. Independent branches can run
concurrently. Training replaces the model, without implicit incremental updates.

Choose **None** or per-row **L2 normalization**. The stored rows are already
normalized; inference applies the same transformation to query rows. Zero vectors
cannot use L2. The training input's `feature_contract` is retained, and contracts
within a batch must agree. HOG supplies its encoding, depth, window geometry,
native version and settings. Other descriptor blocks can provide the same payload
key. Ordinary rows without contracts remain supported; the caller then owns
preprocessing consistency.

Prediction requires an exact contract match by default. **Dimensions only** is an
explicit override of that guard, while saved normalization still applies. Known
frame, ticket, recipe or pyramid conflicts between the optional review image and
descriptor are rejected before native prediction. An image can accompany only
one descriptor row. Outputs retain the current descriptor's ticket and context,
and the image's pixels remain unchanged.

## Neighbor and judgment controls

| Control | Meaning |
| --- | --- |
| Training K | Stored default number of neighbors, 1..64 and no greater than training rows. |
| Prediction K | 0 uses the stored default; a positive value overrides it. Excess K errors instead of silently clamping. |
| Distance policy | Nearest neighbor of the voted class, nearest of all returned neighbors, mean of their Euclidean distances, or Kth/farthest returned neighbor. |
| Max distance | Inclusive cutoff for that statistic in the stored working feature space; 0 disables it. |
| Min vote fraction | Inclusive minimum fraction of K neighbors voting for the candidate. |
| Reject vote ties | Enabled by default; withhold recognition when classes tie for most votes. |
| Accepted labels | Optional trained classes allowed by the inspection. Empty accepts every recognized class. |

Native neighbor outputs contain squared L2 distances. The block preserves those
values and also reports their square roots as Euclidean distances. **Mean**
averages Euclidean distances, not squared distances. Default **Winning class
nearest** measures the closest returned neighbor belonging to the voted class;
a close minority-class neighbor cannot alone satisfy this default distance gate.

Native majority voting supplies the candidate. On equal vote totals, this tested
backend favors the lower encoded class ID, which is the first-appearing stored
class. Equal-distance neighbors at the K boundary favor earlier training rows.
The default tie rejection keeps these arbitrary priorities from being mistaken
for an unambiguous recognition. Vote agreement is not a calibrated probability;
the block does not claim a confidence score.

KD-tree mode is not offered in this classifier baseline: the installed Python
backend returned neighbor indices and a different auxiliary result layout in
the native probe. Its behavior does not satisfy this classifier's response and
distance contract. Approximate search, KD-tree adaptation, regression, weighted
votes, automatic K selection and incremental updates remain separate work.

## Known, unknown and failed inspections

Outputs include recognized Labels, raw Candidate labels, selected Distances,
neighbor class/distance records and Accepted flags. Report includes class mapping,
vote counts/fractions/ties, unknown reasons, model revision and judgment settings.

- Recognized labels contain a string per known row and JSON null for an unknown
  row. The scalar Label connector emits only for one known row; it emits no event
  for unknown or multiple rows.
- Distance, vote-fraction or tie rejection produces an unknown result. Raw candidate
  and neighbor diagnostics remain available, with a `knn_unknown_class` fail reason.
- A known class outside Accepted labels still emits its recognized label and
  produces a `knn_unaccepted_label` inspection failure.
- A run passes only when all rows are accepted. These are ordinary tool judgments,
  not final ticket judgments. Numerical/format errors and native exceptions use
  the common per-block diagnostics and emit no result events.

Failed-input admission is handled before execution by the existing input policy.
A rejected input produces no new work/event or last-input update. A wired missing
review image blocks work instead of silently falling back to no-image processing.

## Model files and bounds

Models are revisioned JSON containing exact float32 training rows, integer class
responses, explicit class mapping, normalization, default K, feature contract and
OpenCV major version. No pickle, native object or executable content is stored.
Save preserves existing files. Load rejects duplicate keys/nonfinite JSON,
changed revisions, malformed rows/mappings, non-unit stored L2 samples and
incompatible native major versions. The checksum detects modification, not
authorship. Models include training descriptors; apply the same sharing policy
used for the source inspection data.

The loader's model path is a sealed project asset copied on Save As. JSON and XML
preserve all controls. Recipes can swap model inputs or loader paths while the
descriptor contract continues to guard preprocessing.

Training/query tables allow up to 2,048 rows, 65,536 dimensions and two million
values. Finite feature magnitudes must not exceed ±1e15; float32 underflow is
rejected. These limits keep native squared distances representable. Models are
bounded to 8 MiB and preprocessing contracts to 16 KiB. Prediction checks the
query rows × training rows × dimensions against a 64-million comparison budget
before creating the native classifier. Use smaller descriptors/batches for
larger tasks. Different hardware/builds can produce small floating-point distance
differences; qualify thresholds on the deployment build.

The [nine original examples](<../examples/opencv tools/knn_classification/README.md>)
use disjoint synthetic stripe positions and an unseen diagonal control. They
qualify the declared CPU workflow, not general OCR or production accuracy.
See also [SVM classification](classical-ml.md), [reusable vocabularies](reusable-clusters.md)
and the [native OpenCV API](https://docs.opencv.org/4.x/dd/de1/classcv_1_1ml_1_1KNearest.html).
