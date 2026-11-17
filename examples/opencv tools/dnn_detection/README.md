# DNN object detection

Open a project, press **Run**, and select **Decode object detections**. Inspect
its box collection, scores/count and report, or enlarge the rendered **Overlay**
image. The filmstrip uses that overlay. Inputs and outputs retain the image ticket
and source coordinates.

| Project | Expected retained boxes | Demonstrates |
| --- | ---: | --- |
| [stretch.json](stretch.json) | 3 | Exact source boxes after nonuniform resizing; class-aware NMS. |
| [letterbox.json](letterbox.json) | 3 | Padding removal and a partly visible clipped box. |
| [channels.json](channels.json) | 3 | Channels-first head layout. |
| [objectness.json](objectness.json) | 3 | Objectness times class score. |
| [ssd.json](ssd.json) | 3 | Normalized SSD boxes and one-based class IDs. |
| [agnostic.json](agnostic.json) | 2 | Suppression across classes when explicitly enabled. |
| [selected.json](selected.json) | 1 | Region selection by original-image box center. |
| [empty.json](empty.json) | 0, intentional failure | Black image drives all model scores to zero. |
| [count_failure.json](count_failure.json) | 3, intentional failure | Minimum count is four; available detections remain inspectable. |
| [yolox_coffee.json](yolox_coffee.json) | 1 cup | Optional pretrained model on an openly licensed real photo. |

The first nine need no download or extra runtime dependency. Their original MIT
ONNX model has fixed proposal boxes and image-dependent RGB-mean scores. It is a
wiring/geometry control, not learned part recognition. Its exact box references,
expected counts and model/image hashes are in `truth.json`. Regeneration uses the
optional `onnx` package and `build_example.py`; use a fresh folder to preserve
different existing assets. There is no customer data in these examples.

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/dnn_detection/letterbox.json" --summary
```

The two negative projects intentionally produce failed judgments and nonzero CLI
exit codes. Lean keeps box values and skips optional result-graphics generation.

## Pretrained coffee example

Run this explicit preparation command once from the repository root:

```powershell
python "examples/opencv tools/dnn_detection/prepare_model.py"
```

It downloads **35,858,002 bytes** into ignored `.local/models/yolox_s.onnx`, checks
the pinned SHA-256 and preserves a different existing file. Then open
`yolox_coffee.json` and Run. Loading/running graphs never downloads models.

Model: OpenCV Zoo YOLOX-S, Megvii YOLOX and OpenCV Zoo contributors, revision
`48b10fa505aeb51fbd127fc60a5c660040e823b7`,
[Apache 2.0](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/LICENSE).
Model SHA-256: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.
Weights are not included in Git. The project uses RGB 0..255, padding 114,
640×640 input, YOLOX grids and an explicit COCO cup-class filter. CPU execution
and postprocessing are qualified; accelerator support is not inferred.

Photo: Rachel Michetti, courtesy of Pikolo Espresso Bar, distributed by
scikit-image under CC0. The project reuses unchanged `examples/assets/coffee.png`;
see [image source/hash/credit](../../assets/photograph-sources.md). Each graph has
an attribution comment. No external source code is copied into the control model.

Box edges describe model rectangles, not foreground masks or measured part
orientations. See [format, selection, suppression and provenance contracts](../../../docs/object-detection.md).
