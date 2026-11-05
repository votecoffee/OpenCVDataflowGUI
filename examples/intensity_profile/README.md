# Intensity profile

Open [intensity_profile.json](intensity_profile.json) with **File → Open…**,
then click **Run once**. The graph measures mean intensity per column in
the central strip. The two bars create peaks at 140 and 220 sample levels,
against a background of 20. Columns outside the strip remain gaps in the
plot, so each peak stays aligned to its source pixel index.

**Intensity profile** is under **Image processing → Statistics**.
Choose **Rows** for one value per source row or **Columns** for one value
per source column. The available reductions are sum, mean, minimum,
maximum, and sum of squares. Values use raw sample units, calculated
separately for each channel. Alpha is excluded by default.

A connected mask or region limits the contributing pixels. The
**Profile** output includes the ordered values and per-position sample
counts. Positions with no contributing pixels have a zero count and a
null value. **Selected count** reports the total number of contributing
pixels across the image. [OpenCV array reduction reference](https://docs.opencv.org/5.0/main_modules/core_array.html#reduce)

```powershell
opencv-dataflow-run examples/intensity_profile/intensity_profile.json
```

[All examples](../README.md)
