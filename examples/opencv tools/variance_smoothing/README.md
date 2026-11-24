# Variance-adaptive edge smoothing

Open [the project](variance_smoothing.json) and choose **Run Once**. Compare the
saved adaptive filter and a 7 × 7 Gaussian on a color step with seeded Gaussian
noise and 36 salt-and-pepper pixels. Each branch reports flat-area RMSE against
the clean image and mean column intensities across the step.

| Branch | Flat-area RMSE | Adjacent-column edge contrast retained | Estimated edge center |
| --- | ---: | ---: | ---: |
| Unfiltered | 11.14 DN | 98.2% | 79.503 px |
| Variance-adaptive | 1.24 DN | 99.0% | 79.504 px |
| Gaussian | 2.29 DN | 27.9% | 79.509 px |

The intended edge center is x = 79.5. The reported center interpolates the
half-level crossing between the two columns next to the known step. Contrast
retention divides their mean intensity difference by the clean step height.
These measurements describe the saved synthetic fixture and parameters.

The block accepts 8-bit grayscale and RGB/BGR(A) and preserves straight
alpha. Convert higher-depth or HSV/Lab inputs explicitly before filtering.
Diameter is the native square window size, from 3 to 31. Both axes must exceed
it. The threshold controls which windows count as low variation. Higher
settings or larger windows can mix the two sides of a sharp edge. A mask or
region limits output writes while neighborhoods use the full input. The native
filter keeps the last row and column unchanged. Work is capped at 1,048,576
image pixels and 32 million window samples.

The two bundled PNGs are original synthetic assets under the repository's
MIT license. Rebuild the project with
`python build_example.py --output <directory>`.
[OpenCV ximgproc filter reference](https://docs.opencv.org/5.0/extra_modules/edgepreserving__filter_8hpp.html)
