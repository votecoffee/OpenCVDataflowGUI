# Sources and rights

All images, ONNX weights, reference vectors, projects and numeric truth were
created for this repository and use its [MIT license](../../../LICENSE).
No customer images, outside datasets or third-party learned weights are included.
Each project has an on-canvas attribution comment.

The 32×32 striped surfaces use training noise seeds 0,1,2 and query seeds
100..104. A dark defect and removed texture occupy rows 10..17, columns 12..19.
The weak defect increases pixel (12,10) by 30 gray levels. The lighting control
adds 55 levels globally. The high-depth image is the dark defect multiplied by
257. The original ONNX Conv has four 2×2 one-hot kernels and stride two; it
rearranges normalized pixel samples into local vectors. These are controlled raw
patch features, not a pretrained backbone or a reproduction of PatchCore.

The general reference-memory concept is described in Roth et al.,
[Towards Total Recall in Industrial Anomaly Detection, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).
No source code, pretrained weights, paper graphics or benchmark images were copied.
This implementation uses direct global nearest-reference distances and optional
even-stride sampling, without that method's coreset or image-score weighting.

The native search API was checked against
[OpenCV batchDistance](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html).
`benchmark.py` imports no application decoder/trainer/inspector and independently
checks native patch extraction and exhaustive distances. Exact asset hashes and
construction settings are in `truth.json`; measured controls are in `benchmark.json`.
