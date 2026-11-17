# Semantic segmentation controls

Open `multiclass.json` and select **Run once**. No plugin, model download or optional
package is required to run these projects. The source, inference, decoder, overlay
and filmstrip nodes show how to turn per-pixel predictions into an inspectable
source-grid mask. Select the decoder to see labels, mask, confidence and per-class
statistics; the multiclass graph also feeds its mask to **Find blobs**.

The tiny bundled ONNX model performs color arithmetic on the current input image:
a fixed 1x1 convolution creates background/red/green/blue scores. It has no trained
weights and makes no learned recognition claim. Dark background, three colored
regions, one hole and an ambiguous gray patch make the boundary/confidence behavior
visible. See [the decoder contracts](../../../docs/semantic-segmentation.md).

| Project | Demonstrates |
| --- | --- |
| [multiclass](multiclass.json) | Four exclusive score channels, source-aligned labels and selected foreground masks; mask-to-blob handoff |
| [letterbox](letterbox.json) | 96x53 source -> 24x13 content in a 24x16 input, with asymmetric top/bottom padding |
| [channels_last](channels_last.json) | Explicit NHWC score axes |
| [probabilities](probabilities.json) | ONNX softmax output with probability-range/sum validation |
| [binary](binary.json) | One foreground logit, sigmoid, foreground threshold and two-class confidence |
| [class_ids](class_ids.json) | ONNX ArgMax/Cast output, explicit class count and absent confidence |
| [low_resolution](low_resolution.json) | 8x12 score head uniformly covering the network field, mapped to the source |
| [linear_scores](linear_scores.json) | Raw-score interpolation before softmax/argmax, distinct from nearest winner labels |
| [selected_roi](selected_roi.json) | Source-grid rectangle gates decoder outputs; inference remains full-image |
| [confidence_gate](confidence_gate.json) | Winning confidence >= 0.8 removes ambiguous/weak pixels |
| [empty](empty.json) | Intentional normal failure: dark input has no selected foreground |
| [area_failure](area_failure.json) | Intentional normal failure: valid mask exceeds the configured maximum area fraction |

Every project ignores background class zero, selects the other classes (only class
1 for binary) and explicitly connects drawable selection outlines to the filmstrip
image. In Debug, the outlines and area label are drawn on original pixels; Lean
skips optional graphics. Failed inspections retain their current numerical outputs
and have the `segmentation.area` reason; they are not exceptions.

`truth.json` records expected pass/fail, exact class/selected counts, ROI size and
asset SHA-256 values. `assets/expected_*.png` contains independently calculated
source-grid labels: 255 represents ignored -1, and other bytes are class IDs.
The generator uses color equations and scalar grid sampling, without executing
inference or the decoder to obtain expected results. The linear-score reference
qualifies the installed OpenCV 5 CPU floating-map implementation; other native
interpolation implementations can differ near a decision boundary.

To rebuild in a development environment with the optional `onnx` package:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python 'examples/opencv tools/semantic_segmentation/build_example.py'
```

Different existing asset bytes are preserved with an error. Regeneration checks
build in a temporary directory, compare all saved projects and exact asset hashes,
then verify the original examples remain unchanged. Save As uses the normal
project-relative image/model path handling; these local assets are always needed.

All images, model arithmetic, graphs and pixel references are original work by
OpenCVDataflowGUI contributors under this repository's [MIT license](../../../LICENSE).
Every graph includes the same attribution in an on-canvas comment. No customer,
third-party dataset or pretrained model is redistributed here.
