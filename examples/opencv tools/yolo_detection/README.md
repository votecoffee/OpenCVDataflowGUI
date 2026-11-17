# YOLO output layouts and source-image detections

These projects qualify a pinned **YOLOX-S** model through the existing DNN and
object-detection blocks. They compare raw grid outputs with an explicitly derived
decoded export, use an openly licensed photograph, and exercise real detection
scores, class-aware NMS and exact rounded letterboxing. No plugin is required.

## Prepare and run

Using the application's Python environment, run from the repository root:

```powershell
python "examples/opencv tools/yolo_detection/prepare_models.py"
```

Preparation uses optional ONNX 1.23.2 for the pinned derived bytes
(`python -m pip install "onnx==1.23.2"` if absent). No training framework or ONNX
Runtime is needed. The script explicitly downloads and verifies the pinned
35,858,002-byte source, then creates two ignored, sealed model/settings bundles
under `.local` beside these examples. `--model <existing.onnx>` reuses identical
source weights. Different existing files are preserved. Graph loading and
execution never download or convert models.

Open [raw_coffee.json](raw_coffee.json) and Run: expect one cup at roughly score
**0.77**. Result outlines/labels are drawable objects; the overlay block renders
them into the filmstrip image while the primary image stays unchanged. Lean skips
optional graphics and retains numerical results.

| Project | Run result | Purpose |
| --- | --- | --- |
| [raw_coffee.json](raw_coffee.json) | Pass, 1 cup | Unmodified raw output, objectness and YOLOX grid expansion. |
| [decoded_rows.json](decoded_rows.json) | Pass, 1 cup | Derived `[1,8400,84]` XYWH/class-score rows. |
| [decoded_channels.json](decoded_channels.json) | Pass, 1 cup | The same derived values as `[1,84,8400]`. |
| [portrait.json](portrait.json) | Pass, 1 cup | A 309×399 crop exercises horizontal padding and unequal rounded axis scales. |
| [duplicates.json](duplicates.json) | Pass, 8 proposals | NMS off; validation reports 1 match and 7 duplicate false positives. |
| [class_aware.json](class_aware.json) | Pass, 2 classes | Zero-IoU NMS retains a cup and a dining-table prediction in separate classes. |
| [class_agnostic.json](class_agnostic.json) | Pass, 1 cup | Cross-class suppression removes the overlapping table prediction. |
| [selected_out.json](selected_out.json) | Fail, 0 | A region excludes the cup's center without cropping inference. |
| [confidence_fail.json](confidence_fail.json) | Fail, 0 | A 0.95 score threshold removes the detections. |
| [objectness_fail.json](objectness_fail.json) | Fail, 0 | Strict objectness threshold 1.0 rejects every proposal. |
| [empty.json](empty.json) | Fail, 0 | A black image gives a valid empty inspection. |
| [wrong_layout.json](wrong_layout.json) | Error, no event | Deliberately interpreting channels as rows rejects; no detections are emitted. |

The source cup was hand marked independently as `[170,18,245,292]` in the
600×400 photograph, including its handle and excluding the saucer. The portrait
annotation subtracts the known crop origin. Each project has a **cup** validation
set with IoU 0.6 and explicit expected inspection outcomes. The two class-suppression
controls also label the wooden table's known full-image extent `[0,0,600,400]`.
Class-aware NMS retains both labels; cross-class NMS misses the table. This small
integration set is not an accuracy benchmark or industrial-parts qualification.
The duplicate, cross-class, excluded, confidence/objectness and wrong-layout cases intentionally
fail detector validation. The black negative is labeled as containing no cups.

```powershell
python -m opencv_dataflow.assistant_cli run-validation "examples/opencv tools/yolo_detection/decoded_channels.json" --set cup --report yolo-report.json --summary
```

Use a new report filename. Save As copies the required model contract/weights and
the declared credits/license, rebases image references and keeps every setting.

## Qualified export contracts

- **Raw:** original OpenCV Zoo YOLOX-S, explicit Classic / OpenCV / CPU;
  `[1,8400,85]` XY/log-WH/objectness/80-class outputs. Decoder grid strides are
  `[8,16,32]`; confidence is objectness times the best class score.
- **Derived:** explicit New / OpenCV / CPU. Local preparation appends standard
  ONNX grid offsets, exponential sizes, objectness multiplication and transpose.
  Learned weights are unchanged. Two outputs carry identical decoded boxes and
  class scores in row/channel layouts. This is a modified YOLOX export, **not a
  YOLOv8 model** or proof of compatibility with every other YOLO implementation.

Both consume float32 NCHW **RGB 0..255**, centered 640×640 letterboxing with fill
114 and no mean/std normalization. For the original photograph, the resized image
is 640×427 with top padding 106; for the portrait it is 496×640 with left padding
72. Exact per-axis scales, rather than an idealized single ratio, map detections
back into continuous source pixel-edge coordinates. Model scores are not calibrated
acceptance probabilities; rectangles do not measure foreground area or orientation.

The two model digests, modification record, photo/crop attribution and licenses
are in [SOURCES.md](SOURCES.md). All graphs include attribution comments.
The generator preserves existing projects. Tests reproduce projects/crop bytes,
compare native preprocessing/grid/NMS results, check thresholds and bad layouts,
and verify portable Save As and cold-worker shutdown. Accelerator, quantized,
anchor-based and other model-family paths remain unqualified.
