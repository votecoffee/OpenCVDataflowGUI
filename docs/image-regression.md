# Image-based regression

`ml.decode_regression` turns a single-image ONNX scalar/vector head into a continuous
measurement. `ml.regression_metrics` summarizes referenced measurements. Both are
under OpenCV → Deep Neural Networks → Regression, using the existing inference,
model-bundle, ticket, exception and graphics contracts. Try the
[fifteen examples](<../examples/opencv tools/image_regression/README.md>).

## Decode a measurement

Connect Image, Tensors and Preprocessing from the same image acquisition, and
expose/connect the inference block's Receipt output. Vector layout accepts [M]
or [1,M]; NCHW scalar accepts [1,M,1,1]. Choose the output name and measurement
index explicitly. Multiple images and spatial maps are rejected rather than
flattened into a plausible value.

`value = raw_value * value_scale + value_offset`. Name and Units describe its
meaning; they do not calibrate the model or camera. Values remain finite and
bounded to +/-1e12. No softmax, confidence interpretation, clamping or final ticket
judgment is invented. The original image pixels, depth, encoding and payload are
returned unchanged; drawable text displays the value and any residual.

Known image/tensor/preprocessing tickets, grids and recipe revisions must agree.
The shared preprocessing validator checks image/network dimensions and inverse
maps. Inference can consume a converted working image while Image is the original
source for display, provided those identities/grids remain consistent. Selection
belongs in the model's preprocessing; decoding a global number does not create a
new image-space region or conceal a spatial output map.

Receipt is required by default. Optional is an explicit choice that permits an
unidentified model; its measurement has no model revision. When present, receipts
check artifact identity, selected output name/shape/type and preprocessing contract.
A sealed contract revision is the model revision; otherwise the artifact SHA-256
is retained. The decoder revision hashes the actual selected head, scale/offset,
name/units and preprocessing contract, so unsealed models with different
normalization are not silently treated as the same measurement pipeline.

Detailed single-image output hashes are verified against the tensor values.
Unpacked batched-image receipts retain row identity and dimensions but cannot
verify the full-batch tensor hash from a single row; the report says that check is
false. Source pixel hashes are not independently verified. These are metadata/
output-content association checks, not proof of camera exposure synchronization
or per-layer accelerator placement.

## References and inspection decisions

An optional Reference input overrides the configured reference when enabled.
Known reference tickets must match the image by default. External is an explicit
choice for a separately sourced reference; responsibility for its association
then remains with the caller. No reference emits no Reference/Residual/Error
events, while the measured Value event still exists.

Residual is prediction minus reference, and Absolute error is its magnitude.
Inclusive value-range and absolute-error limits can yield ordinary inspection
failures (`regression.range`, `regression.residual`). Error checks require a
reference. Malformed inputs, wrong heads, mismatched receipts or incompatible
identity yield an execution/validation error and no current measurement events.
Standard failed-input gating preserves the previous processed-input history.
The Result score remains unset because a measurement/error is not confidence.

## Evaluate a collection

Pass one measurement record, a list, or a FlowGroup to `ml.regression_metrics`.
Records retain sample ID, value, reference, residual, units, model/decoder revisions
and source ticket. References are required for evaluation. Stored residuals are
checked against values/references, and default policies reject duplicate record
IDs, missing model identity and mixed model/decoder revisions. Names/units must
always agree. Explicit policies can allow mixed identities or reused IDs.

The report contains MAE, RMSE, signed bias, maximum absolute error, R² and every
residual record. R² is null for constant references. Choose one inclusive error
metric/limit for the block's pass/fail decision; all statistics remain available.
Numeric outputs are unchanged when optional residual plotting is disabled. The
plot samples at most 512 display points; all accepted records contribute to the
statistics and full report.

Rejecting duplicate record IDs does not establish independent captures. A repeated
image processed again receives a new ID. The report explicitly leaves capture
independence unverified; image-level training/evaluation splits remain a dataset
responsibility. Reference values/units also remain caller-supplied assertions.

## Controlled qualification

The original example model estimates a bead-width proxy from image mean intensity.
Seven training widths fit an OpenCV linear EPS-SVR; its coefficients are exported
to a 298-byte GlobalAveragePool/Flatten/Gemm ONNX vector model. Native SVR training
is part of fixture generation, not a newly added graph training block. The graph
feature qualified here is regression decoding/evaluation of existing ONNX models.

Analytic pixel-cell coverage creates fractional-width stripes with known synthetic
labels and an assumed 0.1 mm/pixel scale. Training/query widths and noise seeds
are disjoint. Three held-out images have MAE about 0.000594 mm and RMSE 0.000710 mm.
A dim control violates the trained lighting assumption and has residual about
-0.46366 mm; its three-image report has MAE 0.1551 mm and fails. This proxy responds
to area/intensity and is not a production geometric bead-width measurement,
physical calibration or field accuracy benchmark.

Independent native SVR, direct ONNX and NumPy mean/linear/error calculations agree
within 2e-5 output units (ONNX versus NumPy) and 2e-7 (SVR versus NumPy). Fifteen
projects cover scalar/vector/NCHW heads, unit conversion, absent references,
valid value/residual failures, corrupt bundles, mixed identities, high depth,
three-image reports and dynamic batched rows. Debug/Lean, exact assets/regeneration,
JSON/XML/Save As, failed-input gates and cold Run All/model shutdown are checked.
No equipment, training-data download or calibrated prediction interval is needed
or claimed.
