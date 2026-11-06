# Colorize scalar image

Open [colorize_scalar.json](colorize_scalar.json) with **File → Open…**
and click **Run once**. **Distance transform** measures how far each
pixel in a circular part lies from its nearest background pixel.
**Colorize scalar image** maps those distances through Viridis. The
part's center is bright, its edge is dark, and the background is black.

Find **Colorize scalar image** under **Image processing → Color and depth**.
**Auto** uses the selected valid pixels' minimum and maximum. **Fixed**
keeps the same mapping between runs and clips values beyond the entered
bounds. A constant auto-range image uses palette index zero. Select
an OpenCV palette or wire a 256-pixel uint8 row or column in grayscale,
BGR, or RGB. The custom palette supplies all 256 colors.
[OpenCV palette reference](https://docs.opencv.org/5.0/main_modules/imgproc_colormap.html)

The block accepts grayscale, signed response maps, and physical depth.
Nonfinite values and nonpositive physical depth are invalid. Masks and
regions restrict the selected pixels. Invalid and unselected pixels
are black in the result. **Levels** contains the uint8 palette indices,
**Valid mask** distinguishes valid index-zero pixels from excluded pixels,
and **Range** reports the actual bounds and depth units when present.

```powershell
opencv-dataflow-run examples/image_processing/colorize_scalar/colorize_scalar.json
```

[All examples](../../README.md)
