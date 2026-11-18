# Sources and licenses

All gradient images, untrained ONNX networks, graph projects and authoring code in
this folder are original OpenCVDataflowGUI material, available under the
[repository MIT license](../../../LICENSE). No customer data, pretrained weights
or third-party image collection is redistributed. Each graph includes an
attribution comment on the canvas. `truth.json` pins exact asset bytes.

The single-input network is an identity 1×1 convolution with fixed channel biases,
ReLU, global mean and flattening. The two-input network adds its inputs first.
They use ONNX opset 13 / IR 8. The examples test image arithmetic, layer timing
association, fusion controls and dataflow contracts, without learned accuracy claims.

Native API references:

- [OpenCV 5 Net API](https://docs.opencv.org/5.0/main_modules/classcv_1_1dnn_1_1Net.html),
  particularly `getPerfProfile`, `getLayerNames`, `getLayerId` and `enableFusion`.
- [Pinned OpenCV 5 net.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/dnn/src/net.cpp)
  and [net_impl.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/dnn/src/net_impl.cpp)
  for the legacy timing vector and native total. API/code references are linked;
  their source is not copied into the examples.

Qualified with Python 3.14.3, OpenCV 5.0.0 contrib 5.0.0.93 on Windows CPU.
Classic uses one-based and New zero-based native IDs in these controls. The
installed binding returns the tick-vector overload; alternate summary overloads
remain unavailable rather than being guessed. Selected/requested settings do not
establish actual per-layer placement or Auto's resolved engine. Other native
builds/accelerators and representative production performance need separate checks.
