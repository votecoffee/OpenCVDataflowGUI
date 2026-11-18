# Sources, rights and construction

All 29 PNGs, the native-trained linear SVM coefficients, ONNX files, graphs,
annotations and reference calculations were created for this repository and are
covered by its [MIT license](../../../LICENSE). No third-party images, learned
weights, customer examples or private data are included. `truth.json` records all
33 image/model/contract/coefficient SHA-256 pins and the 24 training image labels.

Training windows use seeds 0..11 for each local stripe class. Held-out scenes use
seeds 100..105 and analytic inverse pose coordinates to draw rotated silhouettes
and textures. Their centers, angles and class identities are construction truth;
measured blob moments supply the actual inspection poses. The unmarked part is a
low-confidence control, not a representative out-of-distribution dataset.

The model was trained by OpenCV's LINEAR C-SVC on 32×16 float32 resized, 1/255
normalized windows. Signed decision weights/bias are exported as two logits using
original ONNX Flatten/Gemm operations, with an explicit factor three. This is a
margin display choice rather than confidence calibration. The canonical model
appears unchanged in `assets/classifier.onnx` and sealed `bundle/model.onnx`;
`assets/classifier-weights.json` retains coefficients and native rho. The existing
model-contract API creates the checksum/configuration bundle without inference.

Native sampling semantics were checked against [OpenCV geometric transforms](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
The reference uses direct native component labeling plus independent NumPy
moments, coordinates, bilinear interpolation and coefficient multiplication. Its
native inference comparison is API parity, not an independent neural-network
runtime or a production accuracy benchmark.
