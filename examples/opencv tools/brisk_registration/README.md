# BRISK registration

Open [brisk_registration.json](brisk_registration.json) with **File → Open…**
and click **Run once**. The graph detects BRISK features in a synthetic
texture and its shifted copy, matches them, and estimates a homography.
On the bundled fixture, 304 inliers recover about (9.06, 13.08) pixels of
the known (9, 13) pixel shift.

**BRISK keypoints and descriptors** uses OpenCV's standard sampling
pattern. **Threshold** controls AGAST detection, **Octaves** sets the number
of detection scales, and zero octaves selects a single scale.
**Pattern scale** resizes the sampling neighborhood. The block can restrict
keypoint centers to a connected mask or region and cap the number of
returned features. [OpenCV BRISK reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1BRISK.html)

**Match BRISK descriptors** pairs 64-byte binary descriptors with the
same pattern scale using Hamming distance. It supports a nearest-neighbor
ratio test or mutual cross-check, and emits paired points for homography
estimation.

```powershell
opencv-dataflow-run "examples/opencv tools/brisk_registration/brisk_registration.json"
```

[All examples](../../README.md)
