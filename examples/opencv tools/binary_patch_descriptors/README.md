# Compare binary patch descriptors

Open [binary_patch_descriptors.json](binary_patch_descriptors.json) and choose
**Run Once**. LATCH, BEBLID, and TEBLID describe the same ORB keypoints in both
images. The train image is rotated 7° about its center, enlarged by 4%, shifted
by (11, −9) pixels, and brightened by nine gray levels. Each branch uses a
256-bit descriptor and a 48-bit Hamming distance cutoff.

| Descriptor | Accepted matches | Unique correct matches | Precision | Median accepted Hamming distance |
| --- | ---: | ---: | ---: | ---: |
| LATCH | 100 | 85 | 85.0% | 38 |
| BEBLID | 122 | 122 | 100.0% | 37 |
| TEBLID | 180 | 172 | 95.6% | 33.5 |

The feature evaluator uses the known transform and a 2.5-pixel reprojection
tolerance. All three branches have 289 possible one-to-one correspondences and
70.8% geometric repeatability on the shared keypoints. Multiple accepted
matches to the same feature can contribute only one correct match. Select any
**Evaluate** block to review the correspondence overlay and precision-recall
curve.

The three extractors are under **OpenCV → Extra 2D Features → Features**.
LATCH exposes descriptor width, angle compensation, patch radius and smoothing.
Turn off angle compensation for points supplied without measured angles.
BEBLID and TEBLID expose sampling scale and 256- or 512-bit patterns. Their
integral-image path accepts up to 4,194,304 source pixels. Keep the descriptor
family, width and sampling settings consistent between inputs to **Match
descriptors within radius**. [OpenCV LATCH](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1LATCH.html) · [OpenCV BEBLID](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1BEBLID.html) · [OpenCV TEBLID](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1TEBLID.html)

The graph and both images are original MIT assets. Rebuild the example from
the repository root.

    python "examples/opencv tools/binary_patch_descriptors/build_example.py" --output "examples/opencv tools/binary_patch_descriptors"
