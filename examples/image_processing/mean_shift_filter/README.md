# Pyramid mean-shift filtering

Open [mean_shift_filter.json](mean_shift_filter.json) with **File → Open…**
and click **Run once**. The source contains two distinct colors with a
small checkerboard variation on both sides. **Pyramid mean-shift filter**
flattens the checkerboard texture. **K-means image pixels** then assigns
the two color classes and displays them in the cluster map.

**Pyramid mean-shift filter** is under **Image processing → Filtering**.
Set the spatial search radius in pixels and the color radius in
three-channel Euclidean units. The highest pyramid level controls
coarse-to-fine processing. Set it to zero to work directly at source
resolution. **Max iterations** and **Epsilon** control convergence.
The block outputs a filtered image. The downstream K-means block
provides the discrete labels.

The filter supports 8-bit grayscale, BGR, RGB, BGRA, and RGBA and
preserves the source encoding and alpha. For grayscale, the same
intensity is copied into all three working channels, so the color
distance is the absolute intensity difference multiplied by √3.
A mask or region limits where the filtered pixels are written.
Neighborhoods can still read pixels outside the selection.
[OpenCV mean-shift filtering reference](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html)

```powershell
opencv-dataflow-run examples/image_processing/mean_shift_filter/mean_shift_filter.json
```

[All examples](../../README.md)
