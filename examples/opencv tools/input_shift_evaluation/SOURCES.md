# Sources and rights

Images, original ONNX weights and reference vectors come from this repository's
original [local anomaly controls](../local_anomaly/SOURCES.md). They were created
by OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
No customer captures or external learned weights/datasets are included.

Training surfaces use noise seeds 0,1,2; normal and defective query surfaces use
100,101. The defect truth is independently specified as an 8×8 rectangle at
X=12,Y=10. The high-depth image is the defect multiplied by 257. Empty known-clean
truth is explicit. A changed reference control modifies one float32 feature and
reseals its revision. Each graph includes an attribution comment.

Exposure offsets, Gaussian kernels and integer pose translations are recorded per
project in `truth.json`. Truth and support follow known geometry. The benchmark
imports no application shift, inspection, observation or metrics implementation:
NumPy computes integer translation, separable Gaussian weights, pixel patches,
exhaustive distances and mask statistics; OpenCV executes only the native ONNX
reference and reads images.

Native control semantics were checked against the OpenCV
[geometric transformations](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
and [Gaussian filtering](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html)
APIs. No documentation text or third-party figures were copied. These are paired
synthetic software controls, not a field accuracy or independent-capture benchmark.
