# Multiscale image registration

Open `mapper_registration.json` and choose **Run Once**. A coarse transform
from three approximate correspondences initializes the **Align images
(multiscale mapper)** block. The block refines rotation, uniform scale and
translation, then places the moving texture on the reference grid for the
filmstrip.

Inspect the registration block's **Report** for coverage and residual. The
before and after RMSE values compare the same valid pixels and are normalized
by the reference's intensity range. The **Coverage** image marks the
reference pixels with valid source samples. The **Residual** image shows
absolute grayscale error inside that coverage.

The `transform` output's `source_to_target` matrix maps moving-image pixels
to reference pixels and connects to **Warp perspective** or **Transform
points**. Its `target_to_source` matrix and the `geometry` output describe
how the moving image was sampled. Both grids and the moving image's ticket
are retained.

Choose translation, Euclidean rotation and shift, similarity with uniform
scale, affine, or projective motion. Pyramid levels and refinement iterations
control the native fit. The intensity solver uses both full images. Optional
source and reference masks or regions restrict coverage and quality
measurements. Use matching sample depths and dimensions. The CPU path
accepts up to 2.1 million pixels per image and needs usable texture in two
directions.

The assets are original synthetic fixtures under the repository's MIT
license. `assets/truth.json` records the known reference-to-moving matrix.
Rebuild the graph and assets with
`python build_example.py --output <new-directory>`.

[OpenCV multiscale mapper reference](https://docs.opencv.org/5.0/extra_modules/mapperpyramid_8hpp.html)
