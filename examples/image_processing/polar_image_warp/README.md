# Inspect a circular part in polar coordinates

Open [polar_image_warp.json](polar_image_warp.json) with **File → Open…**
and click **Run once**. The graph pastes a patch of background over part
of a circular target to simulate a missing edge segment. **Polar image
warp** straightens the circular edge into a vertical boundary. The
missing segment interrupts that boundary near the 28-degree row.

Find **Polar image warp** under **Image processing → Geometry**.
Columns increase with radius. Rows advance clockwise from the
positive X axis in image coordinates. In this example each column
advances one source pixel in radius and each row advances one degree.
Choose **Linear** for equal radial spacing or **Semilog** to sample more
densely near the center. Semilog requires a radial extent greater than
one pixel. For column `c` in a grid of width `w` and radial extent
`R`, the sampled radius is `c * R / w` in Linear and
`exp(c * ln(R) / w) - 1` in Semilog. Unwrap derives the width from
radius and the height from approximately π times the width when those
settings are zero.

The polar image has its own coordinate frame. **Rewrap** uses the
attached mapping to return to the original Cartesian frame when the
polar grid is intact. With an untagged polar image, set the Cartesian
width, height, center, radial extent and scale manually. Rewrapping
resamples the available disk and produces zero beyond the sampled area.

The **Valid mask** uses nearest-neighbor sampling to show where source
pixels were available. A mask or region restricts validity in input
coordinates. It leaves the resampled image values intact. Higher-order
interpolation can mix samples around mask boundaries and the angular seam.
Wire the valid mask through the graph to preserve coverage restrictions. The
image retains its pixel depth and color encoding. Use **Nearest** for
signed label images. Source and output are each limited to 16,777,216
pixels and 32,764 pixels per axis.
[OpenCV polar warp reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#warppolar)

```powershell
opencv-dataflow-run examples/image_processing/polar_image_warp/polar_image_warp.json
```

[All examples](../../README.md)
