# Binary Census stereo

Open [binary_stereo.json](binary_stereo.json) and run once. The right view
has an eight-pixel shift and a monotonic exposure change. Both branches
search disparities 0 through 15 with a nine-pixel effective footprint.
A mask excludes the same unavailable patch in the right view.

| Result | Binary Census | Native SGBM |
| --- | ---: | ---: |
| Pixels with full matching support | 14,684 | 14,684 |
| Retained estimates | 14,040 | 14,684 |
| Coverage of the image | 73.13% | 76.48% |
| Mean absolute error | 0 px | 0.000115 px |
| Errors greater than one pixel | 0 | 0 |

Census uses a 5 × 5 local comparison window and a 5 × 5 aggregation
window. Its reverse check removes 644 pixels with unavailable or
inconsistent reverse support. SGBM uses its default native consistency
setting. Open the matching report or the two comparison blocks to see
coverage and error. Census also produces Best Cost, Second Best Cost,
and their relative gap as confidence. These describe the forward search.
Use Valid Mask to identify the final supported estimates.

The project uses the app's NumPy/OpenCV integer Census implementation
because the installed OpenCV 5 wheel exposes the binary matcher constants
only. The original synthetic rasters are by the OpenCVDataflowGUI
contributors (MIT). Rebuild in a fresh directory with
`python build_example.py --output <directory>`.
[Original Census correspondence paper](https://woodfill.com/Papers/Census94.pdf)
