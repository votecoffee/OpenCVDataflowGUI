# Image summary statistics

Open [image_statistics.json](image_statistics.json) with **File → Open…**,
then click **Run once**. The graph measures a central strip and marks its
darkest and brightest pixels using **Overlay result graphics**. The example
reports 8,192 selected pixels, a minimum of 60 at (45, 41), and a maximum
of 190 at (115, 57). The 1st, 50th, and 99th percentiles are 75, 110, and
155. The more extreme values outside the strip are excluded.

**Image summary statistics** is under **Image processing → Statistics**.
For each channel it reports minimum, maximum, mean, population standard
deviation, RMS, sum, nonzero count, and any requested percentiles. Results
use raw sample units. Set **Channel** and **Statistic** to choose the
numeric **Value** output. The location outputs and result graphics use
that same channel. Alpha is excluded by default.

Connect a mask or region to limit the measurement. Coordinates remain in
the source pixel frame. **Percentiles** accepts up to 16 levels. When
**Statistic** is **Percentile**, **Value percentile** chooses the level to
emit on **Value**. For an empty selection, **Selected count** is zero,
undefined summary fields are null, and location outputs are unavailable.
[OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run examples/inspection/image_statistics/image_statistics.json
```

[All examples](../../README.md)
