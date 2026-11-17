# ONNX engine dispatch

Open a project with **File > Open**, run once and select **DNN ONNX inference**.
Inspect its Tensors and Receipt results. No plugin or model download is needed.

| Project | Expected behavior |
| --- | --- |
| [classic_rgb.json](classic_rgb.json) | Classic imports the original RGB-mean model through a frozen temporary file. Tensor values are approximately 220/255, 180/255, 30/255. The temporary file is removed before inference. |
| [new_rgb.json](new_rgb.json) | New imports the same bytes directly from memory and produces the same RGB means. |
| [auto_rgb.json](auto_rgb.json) | Auto uses the native automatic-selection memory path. The receipt records the argument, without claiming to observe the final native engine. |
| [new_proposals.json](new_proposals.json) | New accepts the original image-driven proposal fixture; three boxes and their overlay appear. |
| [classic_rejected.json](classic_rejected.json) | Intentional classic-importer error. The same proposal bytes are accepted in the previous project. No current tensor/receipt events are emitted and no automatic retry occurs. |

The optional `engine` validation set tests the pinned color image. The rejected
control deliberately fails; it never satisfies an expected pass. For example:

```powershell
opencv-dataflow-api run-validation "examples/opencv tools/dnn_engines/classic_rgb.json" --set engine --report new-engine-report.json --summary
```

Use a new report filename. The valid controls return exit 0; the rejected control
returns 1. Receipts include `model_import`, describing the frozen byte count,
native engine argument and memory/file transport. No temporary path is retained.
This import evidence is separate from native engine observation and per-layer
accelerator placement, which remain unverified.

RGB mean comparisons allow `1e-5` absolute error for float32 reduction rounding.

The RGB model is original repository code, not a trained recognizer. It averages
each channel after RGB conversion/scaling. The proposal model contains fixed boxes
and image-dependent scores; it is a wiring fixture, not production detection evidence.
All assets are original MIT content, copied unchanged; [SOURCES.md](SOURCES.md)
and [truth.json](truth.json) record origins, hashes and numeric references. Every
project has an attribution/comment block. `build_example.py` reproduces the fixtures
in a fresh folder and preserves existing assets.

See [model import contracts](../../../docs/onnx-engine-loading.md) for the upstream
dispatch issue, cleanup/cache behavior, migrated examples and qualification limits.
