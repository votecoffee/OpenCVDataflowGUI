# Stereo disparity evaluation

Open [disparity_reference.json](disparity_reference.json) and run once.
The two **Evaluate stereo disparity** blocks score a floating-point
estimate against an aligned reference PNG with 256 raw samples per
disparity pixel. A separate confidence TIFF supplies values on [0, 1].
The sources share a fixed ticket. Pixel Grid alignment declares the imported
rasters already aligned.

| Report field | Whole field | Center region |
| --- | ---: | ---: |
| Selected pixels | 13,824 | 4,608 |
| Valid reference pixels | 11,520 | 4,608 |
| Coverage | 80% | 100% |
| Bad fraction among evaluated estimates | 50% | 0% |
| Bad or missing fraction among valid reference | 60% | 0% |
| Mean absolute error | 1.5 px | 0.5 px |
| Mean square error | 3.5 px² | 0.5 px² |

The one-pixel errors sit exactly on the bad-pixel threshold and are
accepted. In the whole field, all 2,304 below-cutoff samples are bad.
At or above the 0.5 cutoff, 2,304 of 6,912 samples are bad. The cutoff
changes those slices without changing the evaluation domain.

In Debug, inspect the disparity error heatmap in Results. Black marks
unscored space. Magenta marks the band with valid truth and missing
estimates. **Bad Mask**, **Missing Mask** and **Evaluated Mask** are
available in both profiles. To see the coverage gate fail, raise the
whole-field minimum coverage above 0.8 and rerun.

All maps are original synthetic assets by the OpenCVDataflowGUI
contributors (MIT). Rebuild to a new directory with
`python build_example.py --output <directory>`.

[OpenCV disparity evaluation functions](https://docs.opencv.org/4.x/da/d17/group__ximgproc__filters.html)
