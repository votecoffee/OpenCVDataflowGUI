# Fitted ellipse inspection band

Open [fitted_shape_region.json](fitted_shape_region.json) with **File → Open…**
and click **Run once**. The project detects the outline of a synthetic oval,
fits an ellipse, and grows and shrinks the fit by three pixels on each
semi-axis. **Combine regions** subtracts the inner region from the outer
one. **Measure selected region** reports the pixel count, bounds, and
centroid of the resulting inspection band.

Find **Region from fitted shape** under **Image processing → Regions**.
Connect an ellipse or enclosing-circle fit and the image in the same frame.
The block emits a reusable region, a binary mask, and a preview. Wire the
region or mask into an inspection that accepts selections. The axis margin
changes the ellipse's semi-axes. The resulting band width varies around an
elongated ellipse. [OpenCV ellipse drawing reference](https://docs.opencv.org/5.0/main_modules/imgproc_draw.html#ellipse)

```powershell
opencv-dataflow-run examples/geometry/fitted_shape_region/fitted_shape_region.json
```

[All examples](../../README.md)
