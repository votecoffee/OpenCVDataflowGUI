# Sources, licenses and qualification limits

Original synthetic images, untrained CNNs, controls, comparison graphs and
authoring/preparation code are OpenCVDataflowGUI material under the
[repository MIT license](../../../LICENSE). No customer data is included.
The original CNNs use ONNX opset 13 / IR 8, signed/unsigned activation Q/DQ,
per-output-channel signed weights, 1×1 convolution, ReLU and global averaging.
An intentionally unsupported custom operation is a negative import control.
`truth.json` pins all eleven bundled image/model/manifest assets. Every graph
has an attribution comment on the canvas.

The six optional pretrained graphs use **unchanged** ONNX Model Zoo MobileNetV2-12
FP32/QDQ/QOperator weights. The pinned [upstream README](https://github.com/onnx/models/blob/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/classification/mobilenet/README.md)
credits the MobileNet/MXNet model and Intel Neural Compressor quantization
contributors and declares Apache 2.0. The exact [model license](MODEL-LICENSE.txt)
is included. Models are explicitly prepared into ignored local bundles, rather
than redistributed as public weights here.

| Export | Bytes | SHA-256 |
| --- | ---: | --- |
| `mobilenetv2-12.onnx` | 13,964,571 | `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5` |
| `mobilenetv2-12-qdq.onnx` | 3,593,903 | `41a36090dafe98f4ad8f9b7fe0b218c56ac3c031e547f0367c30655d2702bffe` |
| `mobilenetv2-12-int8.onnx` | 3,655,033 | `cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f` |

Preprocessing uses RGB, 224×224 stretch, mean `[0.485,0.456,0.406]` and standard
deviation `[0.229,0.224,0.225]` in normalized intensity units. The photo remains
unchanged on disk: coffee photograph by **Rachel Michetti**, courtesy **Pikolo
Espresso Bar**, distributed CC0 by scikit-image. See [shared photo provenance](../../assets/photograph-sources.md).
Its SHA-256 is `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.

Native/operator references:

- [OpenCV 5 DNN API](https://docs.opencv.org/5.0/main_modules/dnn.html) and
  [pinned ONNX importer](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/dnn/src/onnx/onnx_importer.cpp).
- ONNX [QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html),
  [DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) and
  [reference rounding implementation](https://github.com/onnx/onnx/blob/v1.19.0/onnx/reference/ops/op_quantize_linear.py).

Qualification uses Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 Windows CPU.
Independent numeric references, exact native-array comparisons and class/top-k
criteria qualify these exports. A separate half-tie control records the pinned
native build's one-code discrepancy for odd zero points; exact ONNX half-tie
conformance is unqualified. Engine layer-type observations do not establish
per-layer hardware placement. No new weights are trained or converted, and
accuracy, accelerators and speed improvements are not claimed.
