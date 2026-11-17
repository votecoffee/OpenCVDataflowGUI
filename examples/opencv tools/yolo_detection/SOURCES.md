# Sources, changes and licenses

## YOLOX-S

Megvii YOLOX and OpenCV Zoo contributors, including Sri Siddarth Chakaravarthy.
The [pinned OpenCV Zoo model directory](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox)
explicitly licenses all its files under **Apache 2.0**. The unmodified license
is included as [MODEL-LICENSE.txt](MODEL-LICENSE.txt).

| Export | Bytes | SHA-256 |
| --- | ---: | --- |
| Original `object_detection_yolox_2022nov.onnx` | 35858002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| Locally derived output-layout export | 35960492 | `b02b06f1a60e2bccdab23400b2c52d62ced6a02baf856ac53aef42ddb92b0d5c` |

Source revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
[Pinned source weights](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx).

**Modification notice:** OpenCVDataflowGUI contributors append ONNX Slice/Add/Mul/Exp,
Concat and Transpose operations to expose decoded centers/sizes and combined class
scores in row and channel layouts. Original learned initializers and graph nodes
are retained unchanged. A notice is embedded in the derived model's documentation.
No retraining, learned-weight conversion, simplification, quantization or external
model-family implementation is used. The deterministic derivative is built only
by explicit preparation; neither model is committed to Git. Reproduction used
ONNX 1.23.2. A different size/digest rejects instead of silently changing the export.

## Coffee photograph and crop

Rachel Michetti, courtesy of Pikolo Espresso Bar; distributed by scikit-image
under **CC0 1.0**. The main projects reuse unchanged `examples/assets/coffee.png`.
SHA-256: `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
See [dataset attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)
and the [pinned source image](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/coffee.png).

`assets/coffee_portrait.png` is a pixel-preserving crop at `[145,0,309,399]` (XYWH),
with its derived digest recorded in `truth.json`. It is not a separate acquisition
or an independent accuracy sample. The independent hand-marked outer cup box
includes its handle and excludes the saucer. Its cropped annotation subtracts
the declared origin; source/portrait validation cases share the same capture group.
The two cross-class suppression controls additionally label the wooden table's
full-image extent as `[0,0,600,400]`, directly from the photograph's frame. This
reference is independent of the model's predicted table rectangle.

The unchanged black-image control comes from the repository's `dnn_detection`
example, original MIT material. Graphs, scripts and annotations are original
OpenCVDataflowGUI material under the repository's
[MIT license](https://github.com/votecoffee/OpenCVDataflowGUI/blob/main/LICENSE).
Every graph contains the model/photo credit. This file and the Apache license
are declared program assets and accompany a portable Save As.
