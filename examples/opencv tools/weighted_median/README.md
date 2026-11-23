# Weighted median filtering

Open [weighted_median.json](weighted_median.json) and choose **Run Once**.
The guide follows a stepped part outline and a two-pixel stripe. The source
contains salt-and-pepper noise plus a four-by-four saturated patch. All filter
branches use the same five-by-five neighborhood.

| Result | Pixels different from the clean target | Errors along known boundaries |
| --- | ---: | ---: |
| Noisy source | 914 | 44 |
| Unweighted median | 151 | 143 |
| Gaussian guide weights | 8 | 0 |
| Gaussian with damaged samples excluded | 0 | 0 |

The exclusion branch uses a sample mask generated from the known damage in
this synthetic fixture. The region branch applies Gaussian weighting and
retains all source pixels outside its inspection rectangle. The **Sample mask**
chooses which neighbors vote. The **Mask** and **Region** choose where to write
the result. A neighborhood with zero eligible samples retains its source pixel.

The filter offers Gaussian, inverse-linear, inverse-squared, cosine, Jaccard,
and unweighted modes. The guide is converted to 8-bit. Color guides are
quantized internally by OpenCV. Color cosine and Jaccard require nonzero
quantized colors, and monochrome Jaccard requires positive guide values.
Masked-center reconstruction also requires positive guide weights. Sigma sets
the range scale for Gaussian and inverse modes.

The output retains source encoding, depth and alpha. Signed and unsigned
16-bit signals use OpenCV's float path. That path groups source values into
up to 256 bins per channel. Weighted median calls run one at a time inside
the application because the OpenCV implementation uses shared scratch memory.
[OpenCV weighted median implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/weighted_median_filter.cpp)

The project, builder and synthetic images are original MIT assets. Regenerate
them from the repository root with the example builder.

    python "examples/opencv tools/weighted_median/build_example.py" --output "examples/opencv tools/weighted_median"
