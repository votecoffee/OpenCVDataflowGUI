# Find nonzero pixels

Open [nonzero_pixels.json](nonzero_pixels.json) with **File → Open…**,
then click **Run once**. The graph builds a texture mask and finds 7,680
nonzero pixels. It returns 24 sample coordinates for the overlay and
reports the full bounds `[80, 0, 80, 96]`.

**Find nonzero pixels** is under **Image processing → Geometry**.
Connect a monochrome image or a binary mask. Every nonzero sample is
eligible, including negative samples. An optional mask or region limits
the search. **Points** returns `[x, y]` pairs in the source pixel frame.
**Count** is the full number of matching pixels, and **Returned count**
is the number of coordinates emitted.

**Max points** caps the coordinate list. When the count exceeds the cap,
the block samples in row-major order across the full selection. With a
cap of one it returns the first match. **Bounds** is the full selection's
`[x, y, width, height]` rectangle and is unavailable for an empty selection.
**Draw points** adds the returned coordinates to the result graphics.
[OpenCV findNonZero reference](https://docs.opencv.org/5.0/main_modules/core_array.html#findnonzero)

```powershell
opencv-dataflow-run examples/nonzero_pixels/nonzero_pixels.json
```

[All examples](../README.md)
