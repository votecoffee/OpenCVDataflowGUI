# Local window statistics

Open [local_statistics.json](local_statistics.json) with **File → Open…**,
then click **Run once**. The left side is flat and the right side contains a
fine checker texture. Both have about the same local mean. The contrast
preview highlights the checker because its standard deviation is about
80 sample levels, compared with zero in the flat patch.

**Local window statistics** is under **Image processing → Intensity**.
It uses OpenCV integral tables for rectangular neighborhoods. Configure
**Width**, **Height**, and **Anchor** to set each window's size and position
relative to its output pixel. **Crop** uses the in-image portion at edges,
and **Valid** suppresses windows crossing the edge.

Connect a mask or region to limit output centers. By default the selection
also excludes unselected samples from each window. Choose **Centers** to
use all neighboring samples when computing a selected center.

- **Mean** and **Standard deviation** are float64 maps in the original
  sample units, one value per channel. The deviation uses the population
  formula and includes alpha if present.
- **Sample count** is a monochrome map of the number of contributing pixels.
  Zero marks unavailable or unselected windows. Mean and deviation are
  also zero at those locations.

The **Local contrast** graphic stretches the maximum per-channel deviation
for display. [OpenCV integral image reference](https://docs.opencv.org/5.0/main_modules/imgproc_misc.html#integral)

```powershell
opencv-dataflow-run "examples/opencv tools/local_statistics/local_statistics.json"
```

[All examples](../../README.md)
