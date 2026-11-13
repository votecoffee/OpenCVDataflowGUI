# OpenCV SVM image classification

The toolbox's **OpenCV → Machine Learning** group provides **Train OpenCV SVM
classifier**, **Predict OpenCV SVM classes**, and model save/load blocks.
They wrap the native C-SVC classifier for image descriptors such as HOG. No
external training framework or downloaded model is required.

A common training graph is **Image files → HOG → Batch serial entries → Train
SVM → Save SVM**. Set the batch size to the number of training images and provide
one string label per row, in the same order. A wired Labels input overrides the
trainer's label list. Class IDs use first-appearance order; the explicit string
mapping travels with the model. The report includes per-class counts, feature
dimensions and in-sample accuracy. Use independent images to evaluate accuracy.

For inference, connect **Load SVM → Predict SVM / Model** and **HOG → Predict SVM /
Samples**. Optionally connect the corresponding image for a pass-through preview
and drawable class label. Its known frame/ticket must match the features. Connect
Result graphics to **Overlay result graphics** and select the composed image for
filmstrip review. Lean can omit optional drawings; Graphics On requests them.

| Control | Meaning |
| --- | --- |
| Kernel | Linear, RBF, polynomial, sigmoid, chi-squared, or histogram intersection. |
| C | Misclassification penalty. |
| Gamma | Scale for RBF, polynomial, sigmoid and chi-squared kernels. |
| Degree / Coef0 | Polynomial exponent and polynomial/sigmoid constant term. |
| Class weights | Optional positive penalties multiplied by C, in stored class order. |
| Normalization | None or per-row L2, persisted and reused at inference. Zero vectors cannot use L2. |
| Max iterations / Epsilon | Combined native solver termination limits. |
| Contract policy | Require matching feature preprocessing, or explicitly check dimensions only. |
| Accepted labels | Optional allowed classes for tool pass/fail. Empty accepts all class outputs. |

Training accepts numeric feature rows or a batch of descriptor vectors. HOG now
adds a `feature_contract` payload to its existing descriptor output without
changing descriptor values. It records window geometry, working encoding, sample
depth, settings and OpenCV version. Batched contracts must agree. The model stores
this contract, and prediction compares it to the incoming descriptor payload.
RGB/BGR inputs canonicalize to BGR; grayscale and color descriptors have distinct
working encodings. Custom descriptor blocks may supply their own dictionary under
the same payload key. Numeric rows without contracts remain supported, with
dimension checks; the caller owns preprocessing consistency in that case.

Feature values must be finite and representable as float32. Chi-squared requires
nonnegative features. Inputs are bounded to 2,048 rows, 65,536 dimensions and
2,000,000 total values. Models support 2..64 classes, at most 2,048 native support
vectors within the same element budget, an 8 MiB bundle and a 16 KiB feature
contract. Native solver failures and malformed/incompatible models use ordinary
block diagnostics and emit no result event.

Models are portable JSON dictionaries containing native OpenCV YAML weights,
class mapping, normalization, feature contract, OpenCV version and a SHA-256
revision. No native object or pickle travels along a connection. Prediction
creates a local native model, so parallel workers do not share mutable handles.
Save preserves existing files and requires an existing destination directory.
Load resolves project-relative paths and checks duplicate keys, nonfinite JSON,
revision, native dimensions, class IDs, support-vector bounds and OpenCV major
version. The checksum detects changed bytes; it is not an authenticity signature.
Recipes can swap the model input while the contract guard checks its descriptor
compatibility.

`Labels` returns one string per feature row. `Label` emits only for a single row;
`Image` emits only if connected and there is one prediction. Prediction reports
the model revision and class mapping. Labels are closed-set decisions, not
probabilities or unknown-class detection. Accepted-label failures are tool
outcomes, not final ticket judgments. Regression, one-class anomaly detection,
automatic hyperparameter search and HOG sliding-window detector training remain
separate work.

The local study at `examples/private/learnopencv/75_hog_svm` has training,
held-out inference and deliberate contract-mismatch projects. It qualifies only
three synthetic printed characters with disjoint image hashes. It establishes
neither general handwritten OCR nor production classification accuracy.

Native API: [OpenCV SVM](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ml_1_1SVM.html).
