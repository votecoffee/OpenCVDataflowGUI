# Robust interpolation of correspondences

Open [the project](ric_correspondences.json) and choose **Run Once**. Two
panels move in opposite directions with a small vertical stretch. The block
fits a dense source-to-target flow from 206 supplied point pairs, including
12 intentionally incorrect matches. No matches are supplied in the upper
right of the query image.

With the saved settings on OpenCV 5.0, all 16,603 supported pixels are
within 0.001 pixels of the analytic displacement, including the motion seam.

Inspect the **RIC motion with support radius** flow preview. Hue shows
direction and brightness shows displacement. The support mask excludes
out-of-bounds destinations and pixels farther than 13 pixels from a source
seed. **Warp only supported pixels** uses that mask to leave the gap blank.
The field still contains native extrapolations there. The mask measures
geometric support and seed proximity. Confidence and visibility require a
separate check.

Connect an optional matcher **Matches** record and the corresponding query
and train images, or supply ordered point lists. Tune the superpixel
method, size, support count and model refinement for the match density.
RIC requires at least the larger configured neighbor count in distinct
source pixels. Concentrated matches can still leave too few usable neighbors
for native interpolation. The worker has a per-run timeout and a 1 GiB job
memory limit on Windows. Image pairs are limited to 1,048,576 pixels per
raster and 8,192 distinct source seeds.

The original MIT fixtures derive from the
[sparse-to-dense example](../sparse_to_dense_flow/README.md). Run
`python build_example.py --output <directory>` to regenerate the project and
images. [OpenCV RIC reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ximgproc_1_1RICInterpolator.html)
