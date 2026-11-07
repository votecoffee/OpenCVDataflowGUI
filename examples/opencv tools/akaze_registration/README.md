# AKAZE registration

Open [akaze_registration.json](akaze_registration.json) with **File → Open…**
and click **Run once**. The graph shifts a synthetic texture by 9 pixels right
and 13 pixels down, detects AKAZE keypoints on both views, and matches their
descriptors. The homography uses 130 inliers and estimates a translation of
approximately (8.94, 12.97) pixels on the bundled image.

**AKAZE keypoints and descriptors** accepts a mask or region for keypoint
centers. Choose binary MLDB or float KAZE descriptors, with upright versions
for a fixed orientation. The detector exposes threshold, octaves, octave
layers, diffusivity, and the feature limit. **Descriptor bits** and
**Descriptor channels** apply to MLDB. The tested OpenCV build requires
three channels for MLDB extraction. The descriptor length can still be
set explicitly in bits.
[OpenCV AKAZE reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1AKAZE.html)

**Match AKAZE descriptors** requires matching descriptor modes, bit counts,
and channel settings. It uses Hamming distance for binary MLDB and L2
for float KAZE, with a ratio test or mutual cross-check. Its **Matches**
output connects to **Estimate image homography**.

```powershell
opencv-dataflow-run "examples/opencv tools/akaze_registration/akaze_registration.json"
```

[All examples](../../README.md)
