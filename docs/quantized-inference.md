# Quantized image inference and model fidelity

Existing **DNN ONNX inference**, multi-input and batch blocks run supported
integer-quantized ONNX models with **float input/output boundaries**. Keep the
export's QuantizeLinear/DequantizeLinear or QOperator nodes inside the model;
use its declared scale, color, mean/std and dimensions for image preparation.
The wrapper does not automatically quantize, calibrate, rewrite or select a new
model. Model bundles and execution receipts pin settings, bytes and engine selection.

The Windows Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 CPU qualification covers:

- Original signed/unsigned activation Q/DQ CNNs with nonzero zero points and
  per-output-channel signed weight scales/zero points, under Classic, New and Auto.
- Unmodified ONNX Model Zoo MobileNetV2-12 FP32, QDQ and QOperator exports at
  revision `c5cae0f1942c8d7727372c7b1f6b485aa39e4421`, on an openly licensed coffee photo.
  All six quantized export/engine pairs retain floating-point top class 967
  (espresso). Top-five overlap is 0.6..1.0; maximum logit error is about 1.80..2.28.
  The example criteria are absolute error ≤2.5, matching top class and overlap ≥0.6.
  These are declared one-image fidelity limits, not general ImageNet accuracy.

Native layer inspection shows Int8 layer types for QOperator on all three choices
and QDQ on New/Auto; Classic QDQ exposes float convolution around Q/DQ operations.
Type names describe the native graph, without proving per-layer device placement.
Auto's final engine and accelerator execution remain unverified. No speed, memory
or production-recognition improvement is claimed. Qualify each export/backend
separately; the same model can produce different lower-ranked results by engine.

## Float boundaries and import failures

The application emits portable `tensor_v1` float32 records. Selected native int8
or uint8 terminal outputs now reject explicitly: they require a model float
DequantizeLinear output, rather than casting integer codes and presenting them as
dequantized values. A model can still expose an unselected raw diagnostic output.
Other existing integer index/class outputs retain their previous behavior.
Raw integer input-boundary adapters and arbitrary quantized detection heads are
outside this qualification. Scale and zero point cannot be inferred from codes.

An empty native import now reports an unsupported model/engine diagnostic before
input binding. Unknown-operator errors and native exceptions clear cached state
and emit no current result through the common executor. Changing to a supported
model recovers. CPU availability fallback applies only to an unavailable requested
target; it does not silently replace unsupported operators or model exports.
Native import may try its own older parser; the wrapper does not certify that
internal choice or silently retry a different configured engine.

## Native rounding limitation

The pinned build has a measured one-code half-tie discrepancy for **odd nonzero
zero points**, even with an exact binary scale (1/32). It rounds the shifted
value, whereas the ONNX reference rounds before adding the zero point. Signed and
unsigned controls expose a maximum dequantized difference of **1/32** under
Classic/New/Auto; off-tie values and clipping match the independent reference.
Exact half-tie ONNX conformance therefore remains unqualified. Raw unsigned
Classic output also uses the engine's signed internal code convention, another
reason the application rejects undequantized 8-bit output boundaries.

Normal CNN controls match independent round/clip/dequantize/convolve/ReLU/pool
arithmetic within 3e-6. The intentionally saturated nonbinary-scale control
permits 0.001 aggregate numerical discrepancy from native rounding while still
failing its declared float-model fidelity limit by a wide margin. The half-tie
discrepancy is retained as an explicit negative qualification test, not hidden
by changing expected ONNX results. New models need tolerances that reflect their
quantization and acceptance requirements.

## Compare image model tensors

**Compare image model tensors** (`ml.compare_image_tensors`) is under
**OpenCV > Deep Neural Networks (dnn) > Model qualification**. Connect an image
and the Reference/Candidate named tensor maps. Shapes must match, with explicit
float32 records; selectors can choose different output names. Image metadata
checks known ticket, grid/root dimensions, recipe and pyramid conflicts. The
primary image/depth is unchanged. Unknown metadata is permitted and recorded.

Choose **Tensor** for arbitrary matching shapes, or **Classification** for `[C]`
or `[1,C]`. Compare native values or independently softmax-normalized classifier
logits. Each element must satisfy `abs(candidate-reference) <= atol + rtol*abs(reference)`.
Limits are inclusive and use reference magnitude. Classification can additionally
require the same top class and a minimum shared fraction of top-k class indices.
Stable ties use the lower index; actual k is bounded by class count. Classifier
agreement is independent of numerical agreement and is not calibrated confidence.

Report contains maximum/mean absolute error, RMSE, bias, violation count/fraction,
worst flat index, class ranks/overlap and hashes of the original compared float32
arrays. It does not verify model identity: inspect the upstream pinned execution
receipts for weights/settings. Numeric/class failures emit ordinary inspection
results and distinct reasons. Invalid tensors, excessive element budgets and
known identity conflicts emit nothing. Rejected failed inputs retain history.

An optional bounded bar chart and source-image pass/fail label respect graphics
and Lean settings. Plot items zero hides only the plot. Tensor mode emits no
Top1 equal event. JSON/XML, Save As and offline worker execution use the ordinary
application contracts.

Try [the 13 bundled controls and six prepared pretrained comparisons](<../examples/opencv tools/quantized_inference/README.md>).
