# Pixel comparison

Open [pixel_comparison.json](pixel_comparison.json) with **File → Open…**,
then click **Run once**. The graph estimates a local background, computes
the absolute difference, and selects residuals of 40 sample levels or more.
The mask separates the textured checker from the flat patch.

**Compare pixels** is under **Image processing → Image math**. Compare
**Image A** with **Image B**, **Scalar B**, or the configured **Scalar**.
Choose equal, not equal, less, less or equal, greater, or greater or equal.
Equality is exact by default. Set **Tolerance** for an inclusive
absolute-difference threshold with equal or not equal.

For color images, **Channel rule** controls whether all active channels
or any active channel must match. Alpha is excluded by default. The two
images must have equal pixel dimensions. A monochrome Image B broadcasts
across the active channels of Image A. The frame check can be set to
**Pixel grid** for images already aligned by pixel index.

A mask or region limits the comparison. **Mask** contains 255 for
matching pixels and 0 elsewhere. **Count** gives the number of matching
pixels. **Fraction** divides that count by the selected pixel count and
is unavailable for an empty selection.
[OpenCV compare reference](https://docs.opencv.org/5.0/main_modules/core_array.html#compare)

```powershell
opencv-dataflow-run examples/image_processing/pixel_comparison/pixel_comparison.json
```

[All examples](../../README.md)
