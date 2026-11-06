# Tile image

Open [tile_image.json](tile_image.json) with **File → Open…** and click
**Run once**. The project repeats a 112 × 96 shape reference into two
rows and three columns. **Find template instances** finds all six
copies in the 336 × 192 output.

Find **Tile image** under **Image processing → Geometry**. Set **Rows**
and **Columns** to repeat the full input with its original pixel values,
depth, and channel encoding. Tiles touch edge to edge. The
output is limited to 16,777,216 pixels. [OpenCV repeat reference](https://docs.opencv.org/5.0/main_modules/core_array.html#repeat)

The **Layout** output includes tile size, grid size, and the input's
coordinate and pyramid metadata. For a tile at zero-based row `r`
and column `c`, its output origin is
`[c × tile_width, r × tile_height]`. A multi-tile image has its own
coordinate frame. A one-tile image keeps the input frame.

```powershell
opencv-dataflow-run examples/image_processing/tile_image/tile_image.json
```

[All examples](../../README.md)
