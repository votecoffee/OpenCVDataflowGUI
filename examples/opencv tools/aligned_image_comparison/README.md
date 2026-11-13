# Aligned image comparison

Open `project.json` and choose **Run all**. The intact part passes the minimum
similarity of **0.98**. The second candidate has a missing bore and fails.
The images, generator and project are original synthetic fixtures under the
repository's MIT license.

**Compare aligned images (correlation)** (`inspect.image_correlation`) compares
the same pixel positions in two images. Register/crop them upstream if needed.
The example uses native `computeECC`; a normalized-correlation option computes
the equivalent Pearson coefficient in double precision. Both return signed
similarity on **[-1, 1]**, with 1 meaning perfectly correlated variation.
The native path centers/scales selected samples in floating point to avoid
integer mean rounding and extreme-range overflow.

The default uses the lowest B/G/R channel score. The report includes each
channel, its standard deviations, the aggregation and selection settings.
RGB, HSV and Lab inputs are interpreted by their encoding before comparison;
alpha is ignored unless requested. Grayscale mode can compare a color image
with a monochrome reference. An optional mask and region intersect on the
candidate's grid and select matching positions in both images.

Open the normalized-residual diagnostic to see where the samples differ after
removing each channel's mean and contrast. The optional float32 `residual`
output contains the maximum absolute channel z-score difference and is zero
outside the selection. The original candidate output retains its encoding,
ticket and spatial payload. The status label is a drawable graphic on that image.

Correlation ignores positive gain/brightness offset; use **Compare image quality**
with an absolute-error metric when those changes must fail. Correlation is a
global score and can miss small defects. Use a tightly selected inspection
region, residual thresholding, or a local inspection tool for small features.

Constant channels fail by default because correlation is undefined. The explicit
`match_equal` policy accepts two constant regions only when their samples match
within the configured absolute tolerance. It reports a policy similarity of 1
and a **null coefficient**, rather than claiming a measured correlation.
An undefined channel fails the complete comparison even with threshold checking
disabled. Constant tolerance and minimum standard deviation use raw sample units;
mixed pixel depths are not rescaled for these controls.

`pixel_grid` alignment compares equal-size pixels by index, allowing a separately
acquired reference. `same_frame` additionally requires matching spatial payloads
on both images. Masks and regions must belong to the candidate's frame.

API reference: [OpenCV computeECC](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).
Regenerate from the repository root with
`python "examples/opencv tools/aligned_image_comparison/generate.py"`.
