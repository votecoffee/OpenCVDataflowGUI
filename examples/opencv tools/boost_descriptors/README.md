# BoostDesc matching

Open `boost_descriptors.json` and choose **Run Once**. The three branches share
the same SIFT keypoints for each image. The train image rotates the query by
9 degrees, enlarges it by 6.5%, shifts it by (10, −8) pixels, and applies a
gain and offset to its intensity. The evaluation blocks check matches
against the known map with a 2.5-pixel geometric tolerance.

The BGM Bilinear and BinBoost 256 branches use a Hamming cutoff of 48.
The LBGM branch uses an L2 cutoff of 1.3. Review **Metrics** for precision,
repeatability, and recall, or open **Overlay** to see individual matches.
Both ends of a match must use the same variant, scale factor, and orientation
setting.

| Variant | Correct / accepted | Precision | Recall |
| --- | ---: | ---: | ---: |
| BGM Bilinear | 182 / 183 | 99.45% | 90.55% |
| BinBoost 256 | 189 / 189 | 100% | 94.03% |
| LBGM | 169 / 169 | 100% | 84.08% |

These results apply to the saved fixture and thresholds on OpenCV 5.0.0.
All branches shared 201 geometrically recoverable correspondences.

**BoostDesc descriptors** also supports BGM, BGM Hard, BinBoost 64, and
BinBoost 128. Use measured keypoint angles with orientation enabled.
For unoriented points, turn it off. The scale factor still sets the sampling
diameter in both modes. The optional mask and region select keypoint centers.
The OpenCV patch can sample neighbors beyond the selected area.

The block requires OpenCV contrib built with the BoostDesc model data.
The standard application wheel contains all seven trained variants.
`assets/query.png` and `assets/train.png` are original synthetic images under
the repository's MIT license. Run
`python build_example.py --output <new-directory>` to reproduce the graph
and images.

[OpenCV BoostDesc reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1BoostDesc.html)
