# Quantized versus floating-point image inference

Open [signed_new.json](signed_new.json) and press **Run once**. Select **Compare**
to inspect differences, ranks, the bar chart and its source-image result label.
Both inference blocks expose model/settings receipts. The filmstrip uses the
comparison overlay. The small original image/CNN assets are included.

| Bundled project | What to inspect |
| --- | --- |
| [signed_new.json](signed_new.json), [unsigned_new.json](unsigned_new.json) | Activation Q/DQ with nonzero signed/unsigned zero points and per-channel quantized weights. |
| [signed_classic.json](signed_classic.json), [unsigned_classic.json](unsigned_classic.json) | The same controls through explicit Classic CPU dispatch. |
| [signed_auto.json](signed_auto.json) | Requested Auto with execution receipts; resolved native placement is unverified. |
| [strict_tolerance.json](strict_tolerance.json) | Expected numeric failure: zero tolerance does not accept quantization error. |
| [top1_change.json](top1_change.json) | Expected class failure: a close decision flips despite small score differences. |
| [saturation.json](saturation.json) | Expected fidelity failure from deliberately excessive input scaling and clipped quantization codes. |
| [high_depth.json](high_depth.json) | uint16 source with matching normalization, retaining original pixels/depth. |
| [selected.json](selected.json) | Matching region-limited preparation for both models. |
| [sealed.json](sealed.json) | Two pinned model/configuration bundles; Save As copies both models/manifests. |
| [raw_output.json](raw_output.json) | Expected execution error: raw 8-bit codes require a dequantized float model output. |
| [unsupported_op.json](unsupported_op.json) | Expected execution error: a valid ONNX custom operator is unsupported by the selected engine; no comparison event. |

These untrained 1×1 CNNs qualify rounding, clipping, scale/zero-point handling and
dataflow behavior. They do not demonstrate learned inspection accuracy. Debug
shows plots/overlays; Lean retains numerical reports. See
[the contracts and native half-tie limitation](../../../docs/quantized-inference.md).

## Pretrained MobileNet comparisons

Run this explicit preparation step from the repository root to download and
verify the three unmodified Apache-2.0 ONNX Model Zoo exports and seal nine local
model/settings bundles:

```powershell
.\.venv\Scripts\python.exe "examples/opencv tools/quantized_inference/prepare_models.py"
```

`--source <folder>` instead verifies already downloaded files. Conflicting
existing bundles are preserved. Opening/running graphs never downloads models.
The generated `.local` bundles stay ignored by Git; public graphs and the
preparation script are included. Save As copies required sealed assets.

Open any of these six projects after preparation:

- QDQ: [Classic](mobilenet_qdq_classic.json), [New](mobilenet_qdq_new.json),
  [Auto](mobilenet_qdq_auto.json).
- QOperator: [Classic](mobilenet_qoperator_classic.json),
  [New](mobilenet_qoperator_new.json), [Auto](mobilenet_qoperator_auto.json).

Each compares its quantized export to FP32 with the same requested engine and
RGB/mean/std preparation on the existing CC0 coffee photo. All retain top class
967 (espresso). Top-five overlap varies from 0.6 to 1; maximum logit errors range
about 1.80..2.28. The graphs explicitly allow 2.5 absolute error, require matching
top class and at least 0.6 top-five overlap. This is one-image fidelity, without
an ImageNet accuracy or performance claim.

`truth.json` pins the eleven bundled asset files. `prepare_models.py` pins model
revision/size/SHA-256, and refuses different bytes. `build_example.py` regenerates
public fixtures/projects in a new destination using the optional development
`onnx` package; normal graph execution does not require it.

[Sources, licenses and changes](SOURCES.md).
