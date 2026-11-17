# SSD detection on a real photograph

Run a pinned pretrained **SSD ResNet-34** through OpenCV 5, decode its separate
box/class/score outputs and view source-image result graphics in the filmstrip.
The image is an openly licensed photograph; no customer data is included.

## Prepare and run

From the repository root, using the application's Python environment:

```powershell
python "examples/opencv tools/ssd_detection/prepare_model.py"
```

This explicit download verifies **80,366,315 bytes** and the pinned SHA-256, then
creates the ignored `.local/ssd_resnet34` folder beside these projects. It contains
unchanged weights and a sealed model/preprocessing contract. Different existing
files are preserved. `--model <existing.onnx>` can reuse identical local weights.
Opening or running a project never downloads anything.

Open [coffee.json](coffee.json) and Run. Expect one green cup rectangle and label,
approximately score **0.51** on the qualified CPU build. The original image output
stays unchanged; the overlay block draws result objects onto the filmstrip image.
Lean mode omits optional graphics. The project uses **New / OpenCV / CPU** explicitly;
Classic rejects this particular export. No accelerator or other-model support is implied.
The rejected Classic import can emit a native Windows fault diagnostic on this
build; that probe is isolated in a separate process. The sealed project rejects
an engine change before importing, and the qualified New-engine worker closes cleanly.

| Project | Run judgment | What it demonstrates |
| --- | --- | --- |
| [coffee.json](coffee.json) | Pass, one cup | Actual pretrained detections, original-image boxes and result graphics. |
| [selected_out.json](selected_out.json) | Fail, zero cups | A region excludes the box's center without cropping inference. |
| [confidence_fail.json](confidence_fail.json) | Fail, zero cups | A 0.95 score gate removes the prediction. |
| [count_fail.json](count_fail.json) | Fail, one cup | Two cups are required; the available result remains visible. |
| [empty.json](empty.json) | Fail, zero cups | A black image is a valid empty result, rather than an exception. |

Each project has a **cup** validation set. The photograph's outer cup, including
its handle and excluding the saucer, was hand marked as `[170,18,245,292]` in the
600×400 source image. This annotation was not generated from detector output.
At IoU 0.6 the coffee project has one match and no false positives or misses.
This is one integration control, not a detector accuracy benchmark. The exclusion
and confidence controls intentionally miss that known cup, so their detector
validation fails. The count-failure control still localizes the cup and meets its
expected failed judgment; the empty control has no labeled objects.

```powershell
python -m opencv_dataflow.assistant_cli run-validation "examples/opencv tools/ssd_detection/coffee.json" --set cup --report coffee-report.json --summary
```

Choose a new report filename. Save As copies the sealed model bundle and source
credits/license, rebases image references, and retains all decoder settings.

## Export contract

The pinned opset-12 export consumes a float32 NCHW RGB 1200×1200 stretch. OpenCV
bilinear resizing operates on float pixels; preprocessing is
`(RGB / 255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`.
This project's source-unit mean implements the same expression in the existing
DNN block. Small rounding differences from PIL/uint8 resizing are expected.

Outputs are `bboxes [1,N,4]`, `labels [1,N]` and `scores [1,N]`, with normalized
XYXY box edges and one-based contiguous COCO class IDs. The decoder uses class
offset -1 and keeps class 41 (cup). This model already includes suppression, so
additional decoder NMS is disabled. Native int64 labels are represented exactly
for these small IDs by the existing float32 `tensor_v1` connection contract; the
detailed inference receipt retains the native output dtype and model hash.

The reusable `ssd_heads` format also supports explicit YXYX ordering and an optional
valid-count head for padded exports. The existing packed `ssd` format remains
available. Neither format learns unknown anchors or guesses an export schema.
See [decoder contracts](../../../docs/object-detection.md).

The model, photograph, original black control and hand annotation are credited in
[SOURCES.md](SOURCES.md), and every graph contains an attribution comment.
The graph generator `build_example.py` is included for reproducibility and preserves
existing projects. Graph generation uses no inference, conversion, network access
or additional dependency.
