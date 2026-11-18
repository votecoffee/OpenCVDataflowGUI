# Inspection evaluation under input shifts

Use **OpenCV → Deep Neural Networks → Input-shift evaluation** to compare an
inspection's baseline with controlled exposure, blur and affine pose changes.
The blocks accept arbitrary inspection masks, reported identities and scores;
the [eleven examples](<../examples/opencv tools/input_shift_evaluation/README.md>)
use the existing local anomaly tools.

## Create paired views

`ml.image_input_shift` accepts an image and an independent uint8 defect-truth mask.
A known-clean image needs an explicit empty mask. Truth and image must share
known grid/ticket context. The source supports gray/BGR/RGB uint8, uint16 and finite
float32 [0,1]; alpha, physical depth and other encodings need explicit conversion.

Choose a distinct Variant name and an independent Expected pass label. `baseline`
requires neutral settings. Pose uses a pixel-center affine rotation/scale about
the image center, followed by X/Y translation. Choose nearest or linear image
sampling. Truth always uses nearest sampling. Pixel grid size remains fixed.

The operation order is affine sampling, gain/full-scale offset, [0,1] clipping,
then optional Gaussian blur. Uint8/uint16 are returned at their original depth
with round-to-nearest conversion; RGB encoding is retained. Kernel zero disables
blur; otherwise choose odd 3..31. Sigma zero selects OpenCV's automatic value.

Support excludes affine pixels affected by synthetic padding and erodes by the
blur kernel. **Feed Support into the inspection's region/mask input**, as well as
the observation tool. Otherwise the inspection itself may reject synthetic borders.
The support mask does not assert that an arbitrary network receptive field is
unaffected by padding, blur or excluded pixels.

The payload retains the original acquisition ticket and a source identity formed
from its pixel/depth/shape hash and known ticket. Each view records its variant,
settings, source/output/truth/support hashes and reference-to-image transform.
Spatial/pyramid maps compose with the inverse affine transform. These are derived
views of one acquisition, not new camera captures or independent test samples.

## Capture a current inspection

`ml.capture_shift_observation` receives the unchanged shifted Image, Prediction
mask, shifted Truth/Support and inspection Report. Optional Model support further
restricts scored pixels. Optional Score preserves raw units or a declared [0,1]
probability. This declaration does not calibrate a model's confidence.

Prediction's current connection outcome supplies the inspection judgment. An
explicit Passed input can override it for tools with a separate decision output.
Capture validates image/truth/support hashes, matching shift metadata, known
tickets and grids. Stale/other-variant values are rejected. The model and decision
revision paths select bounded text fields from Report; these are supplied identity
assertions, not independent artifact authentication. Use the inference/model
contract tools to establish the underlying model identity.

Minimum truth retention rejects comparisons that lose too much known defect area
to cropping or unscored support. Expected area uses affine area scale; the test is
discrete raster evidence, not exact continuous occlusion measurement. A clean
sample has no defect-retention fraction.

The observation stores packed prediction/truth/support masks, identities, score,
judgment and the original ticket. Capturing a valid failed inspection succeeds as
a data operation; it does not turn that inspection into a passing judgment.
Normal failed-input policies still apply. Rejecting an input emits no observation
and preserves the previously processed image/history.

## Compare complete pairs

Collect observations using existing Batch/FlowGroup tools, then connect
`ml.evaluate_input_shifts`. Required variants is the exact set expected for each
source, including `baseline`. Missing/duplicate variants, nonidentity baselines,
changed source labels/bytes, mixed model/decision revisions and incompatible score
kinds are errors. Set pixel budgets to bound decoded masks and comparison work.

For each view the report includes:

- Actual/expected judgment, false reject or false accept, and baseline judgment change.
- Prediction/ground-truth mask IoU on that view's scored support.
- Baseline-mask agreement after mapping the baseline prediction to the shifted grid,
  restricted to shared scored support. A minimum shared canvas fraction prevents
  a tiny overlap from presenting misleading agreement.
- Raw score, delta from baseline and direction-aware score drop. Choose whether
  lower or higher scores are better. Negative Maximum score drop disables that gate.

Empty mask unions have no IoU, rather than an invented perfect result. Their values
are null and omitted from IoU means. Missing scores remain null; enabling score
gating requires every observation to have a current score. False-reject/accept
rates with no relevant labels are null and their numeric connectors emit no event.

The overall summary includes **all observations, including baseline**. Per-variant
tables provide their own rates, counts, mean scores/deltas and localization values.
For example, one failed shifted clean view and one passing baseline gives 50%
overall false rejects and 100% for that shift. Derived views are correlated; these
counts are not estimates from independent captures and have no confidence interval.

Inclusive limits cover false rejects/accepts, mean truth IoU, mean baseline agreement
and optional score drop. Violations produce ordinary `input_shift.*` failure reasons,
without assigning a final ticket judgment or calibrated Result score. The optional
result plot summarizes counts. Disabling graphics leaves numeric reports unchanged.

## Qualification and limits

The examples use original 32×32 synthetic surfaces, a transparent ONNX raw-patch
encoder and a known-good reference bank. Defect and uint16 pairs preserve exact
64-pixel truth/alignment through a two-pixel translation and small exposure change.
Exposure, blur and odd-phase translation controls reveal false rejects; an overly
loose distance threshold accepts known defects and fails evaluation. Cropped truth,
missing variants and changed valid reference revisions are negative controls.

Independent NumPy translation, separable Gaussian filtering, patch extraction,
exhaustive distances and mask arithmetic verify the results. Native ONNX patch
extraction agrees exactly. These controls qualify the evaluation workflow; they
do not establish pretrained-backbone accuracy, a realistic camera-motion model,
physical lighting tolerances or production acceptance. More general affine controls
use the recorded raster convention and need application-specific image evidence.
