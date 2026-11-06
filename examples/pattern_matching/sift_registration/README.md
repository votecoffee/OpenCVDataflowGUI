# SIFT registration

Open [sift_registration.json](sift_registration.json) with **File → Open…**
and click **Run once**. The graph shifts the bundled image by 9 pixels right
and 13 pixels down. It detects SIFT keypoints on both images, matches their
descriptors, and estimates a homography from the original to the shifted image.
On the bundled image, the estimate uses 184 inliers and recovers about
(8.98, 12.98) pixels of translation.

**SIFT keypoints and descriptors** accepts a mask or region to restrict
keypoint centers. It exposes OpenCV's feature count, octave layers, contrast
and edge thresholds, initial blur, descriptor depth, and precise upscale.
Both float32 and uint8 SIFT descriptors have 128 components.
[OpenCV SIFT reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1SIFT.html)

**Match SIFT descriptors** compares either storage depth using L1 or L2
distance. Choose a nearest-neighbor ratio test or mutual cross-check, with
a configurable distance limit. The **Matches** output connects directly to
**Estimate image homography**.
[OpenCV BFMatcher reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1BFMatcher.html)

```powershell
opencv-dataflow-run examples/pattern_matching/sift_registration/sift_registration.json
```

[All examples](../../README.md)
