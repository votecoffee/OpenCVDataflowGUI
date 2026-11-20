# Confidence-weighted depth smoothing

Open [depth_smoothing.json](depth_smoothing.json) and click **Run once**.
The example splits a 16-bit PNG into noisy depth in millimeters, confidence,
and reference depth. **Smooth depth while preserving edges** filters the
depth plane and passes it to **Depth to point cloud**. Select the smoother to
inspect its depth, valid and rejected masks, and sample counts.

The near plane is 1 m and the far plane is 2 m. Across the 9,436 retained
pixels, RMS error falls from **21.0 mm to 3.8 mm**. The maximum error on the
two columns next to the step is **15 mm**. The cloud omits 114 original
holes and 50 samples with below-threshold confidence.

## Use your own depth

Connect a physical depth plane or a raw single-channel uint16 image. For
raw integers set the physical units and units per sample. An input with
declared units retains them, and the range sigma always uses meters.
You can connect a confidence image from the same pixel grid and capture.
Float confidence uses 0–1. Integer confidence uses the full uint8 or uint16
scale. Below-threshold and zero-confidence samples are excluded from the
filter and marked in the rejected mask if they were valid and selected.
Existing holes stay invalid. Use **Fill small depth holes** separately
when you want to estimate missing samples.

A mask or region limits writes. Trusted neighbors outside it still contribute
to the estimate. Outputs retain the input pixel grid and storage type. Adjust
the spatial radius and sigma for the noise scale, and choose a depth sigma
small enough to preserve the surfaces you want to distinguish. If the range of
trusted values exceeds 4,096 depth sigmas, the block reports a validation error.
Exclude outliers using confidence or adjust sigma to keep OpenCV's range table
from becoming too coarse.

The [fixture builder](build_example.py) recreates the original assets and graph.
The fixture is synthetic and released by the OpenCVDataflowGUI contributors under
[MIT](../../../LICENSE). [OpenCV joint bilateral source](https://github.com/opencv/opencv_contrib/blob/5.x/modules/ximgproc/src/joint_bilateral_filter.cpp).
