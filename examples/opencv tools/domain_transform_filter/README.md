# Domain transform filter

Open [domain_transform_filter.json](domain_transform_filter.json) with
**File → Open…**, then click **Run once**. The graph runs normalized
convolution, interpolated convolution, and recursive filtering side by
side on the same noisy grayscale image.

On the bundled fixture, the left patch's standard deviation falls from
8.92 gray levels to 1.57, 1.09, and 2.99 respectively. The mean contrast
between patches on either side of the edge remains about 130 gray levels.

**Domain transform filter** is under **Image processing → Filters**.
**Sigma spatial** controls the smoothing scale in pixels. **Sigma color**
controls sensitivity to differences in the guide. Integer guide samples
are normalized to [0, 1]. **Iterations** controls the number of filtering
passes, and **Mode** selects the algorithm. Connect a guide in the same
pixel frame, or leave it disconnected to use the input image as the guide.
A mask or region limits where the filtered pixels are written. Output
depth defaults to the source depth, and alpha is copied through in the
chosen depth. [OpenCV Domain Transform reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_filters.html#dtfilter)

```powershell
opencv-dataflow-run "examples/opencv tools/domain_transform_filter/domain_transform_filter.json"
```

[All examples](../../README.md)
