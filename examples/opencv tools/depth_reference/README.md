# Camera depth evaluation

Open [depth_reference.json](depth_reference.json) and run once. The source
blocks load a depth estimate in meters and a reference in millimeters. They
share a ticket and use Pixel Grid alignment because the two maps are already
aligned. The estimate shifts the 1 m to 1.5 m step one pixel to the right,
contains two biased patches, and has a hole at a different location than
the reference. Both branches compare the input depths in millimeters.

| Result | Whole field | Four-column step band |
| --- | ---: | ---: |
| Valid reference pixels | 10,160 | 320 |
| Matched pixels | 10,080 | 320 |
| Coverage | 99.21% | 100% |
| Missing estimates | 80 | 0 |
| Estimates without reference | 80 | 0 |
| Pixels outside 20 mm tolerance | 180 | 80 |
| Mean absolute error | 4.43 mm | 125 mm |
| Maximum absolute error | 500 mm | 500 mm |

Inspect the **Residual** output or color heatmap in the Results tab. The
shifted column has a signed residual of −500 mm. The two smaller patches
measure +15.625 mm and −31.25 mm. Magenta marks missing estimates, and gray
marks estimates without a valid reference. The corresponding binary masks
remain available in Lean. The step band has 25% outside tolerance. Lower
its maximum bad fraction to see the error gate fail.

The reference and estimate are original synthetic assets by the
OpenCVDataflowGUI contributors (MIT). Rebuild them in a fresh directory
with `python build_example.py --output <directory>`.

[OpenCV physical depth and RGB-D conventions](https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html)
