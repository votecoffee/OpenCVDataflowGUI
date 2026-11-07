# Choose a content-aware seam

Open [stitch_seam.json](stitch_seam.json) with **File → Open…** and run
once. Two 16-bit views overlap across 95 columns. The views have
different brightness, with a narrow strip where their samples agree.
Each view also contains a bright mark seen only in that view.

**Average the overlap** fades both marks. **Choose a content-aware seam**
follows the strip of agreement and keeps both marks at full intensity.
The graph-cut mosaic selects a source value at all 9,120 overlap pixels
and covers the same 17,760 pixels as the averaged mosaic.

The [large overlap project](stitch_large_seam.json) enlarges the same
views eightfold. **Auto** estimates the seam at half resolution using
168,064 graph pixels, including padding. The final 1,480 × 768 mosaic
retains both 16-bit marks and selects source values throughout its
583,680-pixel overlap.

- Set **Blend** to **Graph cut** on **Stitch image pair** or
  **Stitch scan sequence**. **Seam cost** offers **Color** and
  **Color gradient**. The terminal cost and bad-region penalty control
  the preference for supported pixels and the cost of crossing an
  invalid boundary. [OpenCV seam estimator](https://docs.opencv.org/5.0/extra_modules/classcv_1_1detail_1_1GraphCutSeamFinder.html)
- The cost image excludes alpha and uses a common intensity scale for both
  views. The mosaic copies the selected warped samples at their original
  depth, including alpha. In a scan, optional exposure correction runs
  before seam selection.
- In a scan, each incoming view is compared with the current mosaic.
  **Layout** reports how many overlap pixels each source contributed at
  each merge, along with the downsample factor and work size.
- **Seam resolution** defaults to **Auto**. It starts at full resolution
  and halves both work dimensions until the graph fits the 524,288-pixel
  limit per merge. **Full**, **Half**, and **Quarter** select a fixed
  work scale and report an invalid setting if the work area exceeds the
  limit. Reduced resolution averages valid samples for the seam calculation
  and can miss narrow features. The final image and support masks use the
  original mosaic grid. [OpenCV multiresolution stitching example](https://docs.opencv.org/5.0/examples/samples_cpp_stitching_detailed_cpp.html)

```powershell
opencv-dataflow-run "examples/custom tools/stitch_seam/stitch_seam.json"
opencv-dataflow-run "examples/custom tools/stitch_seam/stitch_large_seam.json"
```

[All examples](../../README.md)
