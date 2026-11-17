# Point- and box-prompted segmentation

Prepare the pinned model from the repository root with the application's Python:

```powershell
python "examples/opencv tools/prompted_segmentation/prepare_prompt_model.py"
```

This explicitly downloads and checks the 48,312,857-byte Apache 2.0 EfficientSAM-Ti
April 2025 ONNX model, then seals its settings/weights under ignored `.local`.
`--model <existing.onnx>` reuses identical weights. Different existing files are
preserved. Graph opening/execution never download or train anything. No extra
training framework, ONNX package or ONNX Runtime is needed for execution.

Open [coffee_point.json](coffee_point.json) and **Run**. Run All follows the app's
single-image fallback. Define segmentation prompts shows foreground/background
markers or a box; inference returns three candidates; decoding selects and grades
a mask. The filmstrip combines source pixels with prompt/mask graphics. Original
image pixels stay unchanged. No plugin is needed.

| Project | Expected result on qualified New / OpenCV / CPU | Purpose |
| --- | --- | --- |
| [coffee_point.json](coffee_point.json) | Pass, 9,342 pixels | Foreground point; candidate 1, predicted quality about 0.786. |
| [coffee_box.json](coffee_box.json) | Pass, 37,821 pixels | Box prompt; candidate 0. Box is not a clipping region. |
| [coffee_background.json](coffee_background.json) | Fail, 12,285 pixels retained | Nearby foreground/background points contradict every candidate; high predicted quality does not bypass the constraint. |
| [rectangle_box.json](rectangle_box.json) | Pass, 1,911 pixels | Box/nearest mask has Dice 0.977 against exact construction truth. |
| [rectangle_linear.json](rectangle_linear.json) | Pass, 2,000 pixels | Point/linear-logit mask has Dice 1.0; optional source logits are emitted. |
| [quality_fail.json](quality_fail.json) | Fail, 9,342 pixels retained | Deliberate quality floor 2; ordinary inspection failure keeps current geometry. |
| [selected_out.json](selected_out.json) | Fail, 0 pixels | Output region excludes the target; inference and raw prompt checks still use the full source. |
| [mismatched_prompts.json](mismatched_prompts.json) | Error, no decoded event | Decoder deliberately receives a different prompt revision from the model request. |

Select **Inspect native candidate**, change index 0..2 and view/pop out its Results
image to inspect the unthresholded 1024×1024 logit plane. The inference Candidates
output is a grouped set of images with native-grid provenance. Decode reports all
qualities, point consistency, raw mask areas and the selected candidate. Predicted
quality is not measured IoU or a calibrated probability.

Prompts use source `[x,y]` points with labels **1 foreground / 0 background**.
Box is `[x,y,width,height]`. Six model slots are available; a box consumes two.
The near-negative coffee example deliberately retains a real failed constraint.
Try other coordinates or select Report only to inspect the result without that
acceptance rule. Decoder quality/area/resampling settings are independent of the
pinned inference bundle. Editing model/engine/preprocessing settings requires an
explicit new bundle. Lean skips optional graphics and retained display images.

The fused model encodes the image on each execution; no reusable embedding cache
or general SAM-family compatibility is claimed. Source pixels, prompts and model
settings are bound together so changed/mixed requests reject. Exact rectangle
truth is specified independently at `[20,20,40,50]`, not derived from predictions.
Coffee has no pixel-level ground truth here. These are controlled software cases,
not a dataset or industrial accuracy benchmark.

All projects regenerate exactly. Tests compare independent native input tensors,
logits, qualities and source masks, and check Debug/Lean, normal failures, malformed
inputs, warm prompt rebinding, unchanged old buffers, XML, Save As and a cold GUI
worker. Save As copies model/image assets and declared source/license records.
[SOURCES.md](SOURCES.md) and each on-canvas comment retain the model, CC0 photograph
and original MIT fixture attribution. See [full contracts](../../../docs/prompted-segmentation.md).
