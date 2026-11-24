# Low-light enhancement

Open `low_light_enhancement.json` and choose **Run Once**. The upper branch
chooses a BIMEF exposure ratio from shadow samples. The middle branch uses a
fixed ratio of four. The lower branch applies global gamma correction.
Compare the dark patches and fine stripes on the left with the bright patches
on the right. The filmstrip shows the automatic result.

| Method | Shadow contrast, 8-bit levels | Shadow noise, 8-bit standard deviation | Pixels at upper saturation |
| --- | ---: | ---: | ---: |
| Original | 8.09 | 1.29 | 5.91% |
| Gamma 1.8 | 14.47 | 2.55 | 5.91% |
| Explicit BIMEF, SciPy | 15.12 | 2.82 | 5.91% |
| Automatic BIMEF, SciPy | 16.53 | 3.03 | 9.63% |

Contrast is the grayscale mean difference between the interiors of the two
dark patches. Noise is the grayscale standard deviation within the darker
flat patch. A pixel counts as saturated if at least one channel is at 255.
These results are for the generated chart and the portable backend.

The block's **Auto** backend uses OpenCV for 8-bit inputs when the build
includes Eigen. It uses SciPy on other supported inputs or builds. The SciPy
path solves the texture-weighted illumination system and searches a fixed set
of exposure ratios between one and seven, so automatic settings may differ
between backends. The portable path preserves uint16 and normalized float32
precision. Both paths preserve the source encoding and straight alpha. Masks
and regions select pixels to update. The enhancement uses full-image context.

`assets/shadow_chart.png` is an original synthetic fixture under the
repository's MIT license. Rebuild the example and asset with
`python build_example.py --output <new-directory>`.

[OpenCV BIMEF documentation](https://docs.opencv.org/5.0/extra_modules/intensity_transform.html)
