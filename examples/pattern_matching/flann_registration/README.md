# FLANN feature registration

Open [flann_registration.json](flann_registration.json) with **File → Open…**
and click **Run once**. The graph detects SIFT features in a synthetic texture
and a shifted copy, uses **Match descriptors with FLANN** to pair them, and
estimates a homography. It recovered about (8.98, 12.98) pixels of the known
(9, 13) pixel shift on the bundled fixture.

The block builds an approximate KD-tree index for matching SIFT,
[standalone KAZE](../kaze_registration/README.md), and float AKAZE
descriptors. Query and train must use the same descriptor family and mode. SIFT descriptors stored as uint8 are converted to float32 for
the L2 search. Binary MLDB and ORB descriptors use their existing Hamming
matchers. [OpenCV FLANN reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1FlannBasedMatcher.html)

**Trees** controls the number of index trees. **Checks** controls the search
budget. Choose a nearest-neighbor ratio test or reciprocal nearest matches.
Both modes apply **Max distance** in L2 descriptor units. The approximate
search can return different neighbors than an exhaustive matcher.
[OpenCV feature matching tutorial](https://docs.opencv.org/5.0/py_tutorials/py_features/py_matcher/py_matcher.html)

```powershell
opencv-dataflow-run examples/pattern_matching/flann_registration/flann_registration.json
```

[All examples](../../README.md)
