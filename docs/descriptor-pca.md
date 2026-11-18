# Image-descriptor PCA

Use **HOG → Batch serial entries → Fit principal components** to learn a compact
feature space from training images. Connect the batch and fitted model to
**Transform principal components**, then feed its projected rows to an existing
classifier trainer. **Save principal component model** and **Load principal
component model** retain the exact mean, basis, normalization, training descriptor
contract and content revision in bounded, strict JSON. Existing destination files
are preserved. No optional library, model download or live native handle is needed.

At inference, connect the same descriptor tool and the loaded PCA model to the
transform block, then connect its Samples output to the trained classifier.
The [ten projects](<../examples/opencv tools/descriptor_pca/README.md>) include both
training and held-out inference. The older numeric table project still works.

## Feature and model identity

Training accepts numeric rows, a single descriptor or a group of descriptor packets;
at least two rows and positive fitted rank are required. Grouped preprocessing
contracts must agree, including known versus missing contracts. The limits are
65,536 dimensions, 100,000 rows and two million values, with additional work and
estimated workspace guards before native execution. Models are limited to 8 MiB;
orthonormal-basis verification has a separate fixed work ceiling.

New models use `principal_components_v2`. Their SHA-256 revision covers mean,
basis, normalization, training count, centering, retained eigenvalues/energy and
descriptor contract. The native float64 fit has estimated or explicitly supplied
mean, fixed component count or minimum retained variance, and None or per-row L2
normalization. Numerically zero-rank components are excluded. Largest-loading sign
normalization improves repeatability; repeated eigenvalues can still have different
bases across fits/builds. A new basis has a new revision and needs a matching
downstream classifier. This is not a semantic model identity or an authenticity
signature; a publisher can deliberately edit and rehash a model.

Projected wires carry `principal_components_projection_v1`, the PCA revision,
source descriptor contract, normalization and dimension count. A classifier trained
on those wires rejects features from a different valid PCA model even if their
dimensions match. Reconstruction requires the matching projected identity and
emits a distinct `principal_components_reconstruction_v1` contract. Approximate or
L2-normalized reconstructions are not silently labeled as original raw descriptors.
L2 reconstruction cannot restore the original vector magnitude.

**Require match** is the default. **Dimensions only** explicitly bypasses source
compatibility checks, marks the result unverified and gives its output a distinct
contract containing the actual source identity. A strict classifier trained on
verified features therefore rejects that bypassed output. Unknown numeric contracts
remain unknown; matching dimensions alone never qualify an image pipeline. Legacy
`principal_components_v1` mean/basis models remain readable, including untagged
numeric reconstruction, and are identified as legacy in the report.

## Results, judgments and provenance

Project mode reports per-row root-mean-square error between the normalized input
and its projected/back-projected reconstruction. An optional inclusive error limit
fails the inspection while keeping the valid projection, current image and error
available. It does not assign a final ticket judgment. Reconstruct mode does not
claim a residual against an unavailable original and emits no error value.
Training variance is not held-out accuracy. With a fixed supplied center it is
energy about that center rather than covariance about the estimated sample mean.
Residuals and nearest-neighbor distances are not calibrated probabilities.

An optional review image must belong to a single descriptor invocation and agree
with known ticket, spatial, pyramid and recipe identity. Its original pixels,
depth/encoding and metadata are unchanged. Numeric projection retains the current
descriptor's ticket/grid/companion data; grouped rows retain individual ticket,
grid, pyramid and recipe provenance in `feature_rows`. A result plot and drawable
image label respect graphics/Lean settings. Filmstrip overlays use those drawings.
Crop/mask/region an image before describing it; these tools do not reinterpret a
descriptor vector as a pixel mask.

Invalid/mismatched models, resource guards and native exceptions emit no current
result event; the common executor records diagnostics and valid inputs can recover.
Failed-input policies and missing connected inputs gate execution before last-input
history is changed. JSON/XML reopen and Save As retain the exact model assets.
No classifier training or native PCA fit occurs merely by opening a project.

## Qualification and limits

The original MIT example uses six distinct 32×32 stripe training windows and
three held-out windows with disjoint pixel hashes. Native HOG produces 324 values.
The count model reduces these to **two components**, retaining **99.9764%** of
training variance; the variance-target model selects one component. An independent
native HOG/NumPy SVD benchmark checks the saved subspaces, reconstruction error and
raw/reduced nearest-neighbor labels. Both known held-out orientations classify
correctly with and without reduction. Their ordinary two-component RMSE is
**0.00407/0.01636**, versus **0.14491** on the diagonal control; the demonstration
uses an explicitly chosen **0.03** limit. The unguarded nearest-neighbor classifier
still assigns the diagonal to a known class, illustrating why retained variance
and classification alone are insufficient novelty evidence.

Debug/Lean projects cover training/save, ordinary/L2/variance inference,
reconstruction, a residual failure, a changed HOG contract, a different valid PCA
revision and explicit dimensions-only operation. Focused checks cover independent
SVD/native references, finite float64/fixed-mean behavior, legacy compatibility,
boundaries, current image/provenance, guarded failures/recovery, portable assets
and cold background Run All shutdown. These controlled windows do not establish
production defect sensitivity, general classification accuracy or speed benefit.
Choose training populations, normalization, dimensionality and thresholds using
application-specific validation images excluded from fitting and threshold tuning.

Native API reference: [OpenCV PCA](https://docs.opencv.org/5.0/main_modules/classcv_1_1PCA.html)
and [projection/back-projection functions](https://docs.opencv.org/5.0/main_modules/core_array.html).
