# Image-feature outlier inspection

Use **Fit feature covariance** (`cv.feature_covariance`) to learn normal variation
from known-good image descriptors. Save with `cv.save_feature_covariance`, load
with `cv.load_feature_covariance`, and connect a current descriptor to
**Mahalanobis distance** (`inspect.mahalanobis_distance`). The
[thirteen offline projects](<../examples/opencv tools/feature_outliers/README.md>)
include training, saved inference and deliberate failures. No extra package is needed.

## Features and fitting

Samples accepts numeric rows or a batch of descriptor packets. Grouped feature
contracts must agree. Image descriptors retain their preprocessing identity;
PCA features include the exact projection-model revision. A model learns only
the supplied training rows. It neither searches images for parts nor chooses
alignment, crop, descriptor or inspection threshold for the application.

Native `calcCovarMatrix` uses float64 rows with a full feature covariance. Choose
Sample (divide by N-1, at least two rows) or Population (divide by N). Optional
per-row L2 normalization is saved in the model and repeated during scoring.
Zero-length L2 descriptors are rejected. An empty Mean estimates the training
mean; a provided mean is a fixed center in the normalized feature units.

Tables retain float64 precision, including small variations around large offsets.
There are at most 100,000 rows, 512 features and two million table values. The
default 324-value HOG descriptor fits directly; PCA is optional. Work and estimated
workspace limits can reject a large fit before native calls. Defaults are
64 million work units and 128 MiB; these estimates are not process RSS limits.
Training work is estimated as N*D² + D³. Raising a dimension bound therefore does
not bypass the configured work allowance.

## Scoring and judgment

Distance uses the current row minus the trained mean and the inverse covariance,
with native SVD inversion and `Mahalanobis` scoring. Ridge adds a nonnegative
value to every covariance diagonal, in squared feature units. It is explicit
regularization, changes distance scale, and is often necessary when there are
fewer independent training rows than dimensions. Singular or ill-conditioned
unregularized models produce a contained error. Invalid covariance matrices are
rejected rather than repaired during scoring.

The default Require match policy checks the trained feature contract, including
descriptor settings and PCA revision. Dimensions only is an explicit unverified
override, recorded in the resulting distance feature identity. Untagged legacy
numeric tables remain supported. Output distances have their own measurement
contract and are not labeled as the original descriptor space.

Outputs include each distance, the maximum, outlier row indices/count and a report.
Check distance enables an inspection failure when any distance exceeds the
inclusive Distance limit. With that checkbox off, the same outlier data is
reported without failing. Distances are not probabilities; the block supplies
no probability score or final ticket judgment. Fit the threshold using separate
application tuning and validation images, rather than assuming a universal
statistical cutoff for an arbitrary descriptor.

The optional Image input is an unchanged review image for one descriptor packet.
Known ticket, pixel-grid, pyramid and recipe context must agree. A batch must be
selected or unbatched before attaching a single review image. Distances retain
current packet context and per-row provenance; the image retains its own pixels,
depth, encoding and companion metadata. Missing wired data and failed-input
policies gate execution before last-input history changes.

With graphics enabled, the block provides a distance plot and a drawable outline
of the entire review window, colored by judgment. **That outline does not locate
the defect.** Existing overlay/filmstrip tools can draw it on the final image.
Lean skips both plot and drawing while keeping identical numerical results.
Native exceptions are contained by the common executor; current outputs are
suppressed on errors and a later valid input can recover.

## Models and evidence

New `feature_covariance_v2` models store the mean, covariance, training count,
sample/population policy, row normalization, centering, feature contract and
content revision. Revision checks detect changed content, not publisher identity.
Readers check schema, shape, finite bounds, symmetry, positive semidefiniteness
and metadata before accepting a model. Strict JSON rejects duplicate/nonfinite
fields and files over 8 MiB. Legacy `feature_covariance_v1` models are accepted as
unsealed numeric models; they cannot verify image preprocessing identity.

Saving preserves existing destinations. Loading models uses sealed project assets,
so Save As carries the same model bytes; JSON/XML reopen does not retrain. Native
handles are local to an invocation and no native object is serialized. Scoring
has the same configurable work/workspace bounds as training.

The original MIT controls use nine known-good training windows with small position
variation, separate noise for held-out queries, and 144-value HOG descriptors.
At Ridge 0.001 and distance limit 4, the three known-good images score
**2.5298, 2.5063 and 2.4078**. The crack and missing-arc edits score **17.0936**
and **29.6331**; the empty view scores **92.3548**. The subtle one-pixel intensity
edit scores **2.5319 and passes**, a deliberate recorded miss. Edit masks describe
fixture construction, not localized detector results.

The L2 controls use their own model and Ridge 0.0001. An all-zero descriptor has
no L2 normalization and is reported as unsupported, not assigned an outlier score.
PCA examples demonstrate matching/mismatched model revisions on known images;
they do not claim defect sensitivity after compression. Independent native HOG,
NumPy covariance and quadratic-form references agree with native distances within
**3e-14** on the recorded benchmark. These controlled windows qualify contracts,
not real-camera sensitivity, accuracy or throughput.

Primary reference: [OpenCV 5 array operations](https://docs.opencv.org/5.0/main_modules/core_array.html)
(`calcCovarMatrix`, `eigen`, `invert` and `Mahalanobis`).
