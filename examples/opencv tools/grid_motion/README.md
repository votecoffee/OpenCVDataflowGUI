# Filter matches with grid motion statistics

Open [grid_motion.json](grid_motion.json) and choose **Run Once**. The graph
compares nearest ORB matches with and without **Grid motion match filter**.
Both branches use the same 160-iteration partial affine RANSAC fit. Open the
two **Review** blocks to see the correspondences and the inliers of each fit.

The train image has a 7° rotation about (300, 220), scale 0.92, shift
(19, −13) pixels, and two occluders. A correct pair falls within three
pixels of the known transform. The fit error is measured at five independent
query marks, including the center and four points near the corners.

| Input to the affine fit | Candidate pairs | Correct pairs | Correct fraction | Fit RMS at held-out marks |
| --- | ---: | ---: | ---: | ---: |
| Nearest ORB | 4,200 | 304 | 7.2% | 261.13 px |
| After GMS | 247 | 198 | 80.2% | 0.48 px |

**Grid motion match filter** is under **OpenCV → Extra 2D Features →
Matching**. It accepts up to 10,000 paired points with both source image
sizes. Use **Nearest** in **Match ORB descriptors** for up to one candidate
per query. GMS needs enough neighboring pairs to support motion estimates.
A higher **Threshold factor** demands stronger support. The rotation and
scale options test additional grid configurations. [OpenCV GMS reference](https://docs.opencv.org/5.0/extra_modules/xfeatures2d_match.html)

Accepted and rejected outputs retain the original point order, metadata
and scores. **Kept mask** has one flag per input pair. Send the accepted
matches onward to an affine or homography fit.

Both images and the graph are original MIT assets. Rebuild them from the
repository root.

    python "examples/opencv tools/grid_motion/build_example.py" --output "examples/opencv tools/grid_motion"
