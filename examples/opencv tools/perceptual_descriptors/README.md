# Color and Radial Perceptual Hashes

Open [Color Moment](color_moment.json) or [Radial Variance](radial_variance.json) and click **Run once**. Both graphs hash the same synthetic reference part and compare it to a 15° rotation, a smaller resampling, and a different part. Select a comparison block to inspect **Distance** or **Similarity** and the **Measurements** output.

| Sample | Color Moment distance ↓ | Radial Variance similarity ↑ |
| --- | ---: | ---: |
| Identical | 0.000 | 1.000 |
| Rotated 15° | 0.553 | 0.557 |
| Resampled to 173 × 173 | 1.112 | 0.988 |
| Different part | 67.183 | 0.378 |

## Use the blocks

Set **Compute image hash → Method** to **Color Moment** or **Radial Variance**, use the same settings for each pair, and wire both **Hash** outputs into **Compare perceptual descriptors**. Input images are 8-bit grayscale or color. The methods ignore alpha. Crop upstream before hashing a selected area.

- **Color Moment** preserves 42 float64 coefficients. Its distance is the Euclidean difference multiplied by 10,000, following OpenCV. **Maximum color distance** sets an inclusive upper limit.
- **Radial Variance** preserves 40 uint8 coefficients and exposes Gaussian sigma and projection angle count. The compared value is peak cyclic correlation within [0, 1]. **Minimum radial similarity** sets an inclusive lower limit. Uniform grayscale inputs or constant descriptors report insufficient variation and emit no similarity. The measurement report retains the native correlation before normalization of floating-point roundoff.

Threshold checking is disabled in both example projects. To try it on these fixtures, set **Maximum color distance** to 2 or **Minimum radial similarity** to 0.45. Set production limits with representative images at the expected rotation and scale. These compact descriptors can collide, and their agreement varies with content. For example, the radial score drops from 1.000 to 0.557 after the 15° rotation in this fixture.

[OpenCV image hash API](https://docs.opencv.org/5.0/extra_modules/img_hash.html) · [Color Moment implementation](https://github.com/opencv/opencv_contrib/blob/5.x/modules/img_hash/src/color_moment_hash.cpp) · [Radial Variance implementation](https://github.com/opencv/opencv_contrib/blob/5.x/modules/img_hash/src/radial_variance_hash.cpp)

The synthetic part images are original to this repository and covered by its MIT license. Rebuild both projects and their PNGs with `python build_example.py --output <folder>` in the application environment.
