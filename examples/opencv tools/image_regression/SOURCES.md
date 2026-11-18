# Sources, rights and construction

All fifteen PNG images, native regression parameters, ONNX exports, graphs and
reference calculations were created for this repository and use its
[MIT license](../../../LICENSE). No customer captures, external datasets or
third-party learned weights are included. `truth.json` records SHA-256 for all
30 artifacts, including the deliberately changed/corrupt model controls.

Training widths are 4,8,12,16,20,24,28 pixels, with seeds 0..6. Held-out widths are
6.5,13.25,21.75 pixels with seeds 100..102. Analytic pixel-cell overlap gives edge
intensities; a small separate sensor-noise sample is added before quantization.
Query/train hashes and width labels are disjoint. Reference labels use an assumed
0.1 mm/pixel. The dim control reduces foreground/background contrast from 200 to
130 units while retaining width and noise seed. Batch controls use the same width
with new seeds 110..112. `test16.png` is the original middle image multiplied by
257, preserving full-scale intensity correspondence.

OpenCV EPS-SVR uses a LINEAR kernel, C=1000, P=1e-5 and an explicit solver limit.
`assets/native-eps-svr.yml` retains its native model. Its signed coefficient/bias
are exported to original ONNX GlobalAveragePool/Flatten/Gemm/Identity operations;
the second channel is mean normalized intensity. NCHW and dynamic-batch variants
are explicit exports. Existing model-contract APIs seal the files/settings without
inference. The changed model scales the width coefficient by 1.05; the corrupt
bundle changes one model byte while preserving its original checksum record.

The independent reference calls native SVR and ONNX directly and separately
calculates image means, linear predictions and residual statistics. It imports no
application decoder/metrics implementation. Native training semantics were checked
against the [OpenCV SVM API](https://docs.opencv.org/4.13.0/d1/d2d/classcv_1_1ml_1_1SVM.html).
These are controlled CPU integration examples, not physical calibration, industrial
bead inspection or confidence-interval qualification.
