# Sources and rights

All lossless videos, ONNX operations/weights, truth masks, project graphs and
evaluation code were created by OpenCVDataflowGUI contributors under the
repository's [MIT license](../../../LICENSE). No customer files, external video
datasets or third-party learned weights are included. Every graph has attribution.

The 64×40 frames contain same-class red/blue rectangles. Crossing positions advance
by four pixels/frame at 10 fps. Ordinary sequences draw red in front and remove
occluded blue pixels from visible truth. Transparent overlap adds both color
channels and retains both masks. Occlusion/expiry controls remove blue for the
declared frames. Truth is authored from rectangle visibility, not tracker output.

FFV1 encoding is decoded and compared to every source frame during generation.
The ONNX graph gathers red/blue image channels as instance masks. A zero-weight
Gemm supplies declared full-frame proposals. This transparent fixture has no trained
recognition/backbone claim. All 38 masks are independently compared with raw pixels
and authored visible truth using direct native inference.

The tracker uses the shared assignment solver and OpenCV histogram routines.
Native histogram-distance semantics were checked against the
[OpenCV 5 histogram API](https://docs.opencv.org/5.0/main_modules/imgproc_hist.html).
No external paper figures, datasets or documentation text were copied. Results
describe controlled CPU association, not field identity or occlusion accuracy.
