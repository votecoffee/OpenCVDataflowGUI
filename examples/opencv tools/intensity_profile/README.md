# Intensity profile

Open [intensity_profile.json](intensity_profile.json) with **File → Open…**,
then click **Run once**. The graph measures mean intensity per column in
the central strip. The two bars create peaks at 140 and 220 sample levels,
against a background of 20. Columns outside the strip remain gaps in the
plot, so each peak stays aligned to its source pixel index. A parallel
**Last maximum per column** branch returns the coordinates of the
brightest pixels. At column 42, the last brightest pixel is `[42, 71]`.

**Intensity profile** is under **Image processing → Statistics**.
Choose **Rows** for one value per source row or **Columns** for one value
per source column. The available reductions are sum, mean, minimum,
maximum, and sum of squares. Values use raw sample units, calculated
separately for each channel. Alpha is excluded by default.

A connected mask or region limits the contributing pixels. The
**Profile** output includes the ordered values and per-position sample
counts. Positions with no contributing pixels have a zero count and a
null value. **Selected count** reports the total number of contributing
pixels across the image.

For **Minimum** and **Maximum**, every channel also reports a list of
`[x, y]` locations aligned with its values. An empty row or column has a
null location. **Extrema tie** chooses the first or last matching pixel
along the reduced axis. **Points** returns just the valid locations for
the selected channel, in source row or column order. The port is available
in minimum and maximum modes.
[OpenCV array reduction reference](https://docs.opencv.org/5.0/main_modules/core_array.html#reduceargmax)

```powershell
opencv-dataflow-run "examples/opencv tools/intensity_profile/intensity_profile.json"
```

[All examples](../../README.md)
