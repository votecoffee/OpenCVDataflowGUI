# Tilted window statistics

Open [tilted_statistics.json](tilted_statistics.json) with **File → Open…**,
then click **Run once**. The graph measures two diagonal patches. Each
window contains 288 pixels. Their mean intensities are 180 and 30.
**Overlay result graphics** outlines both windows.

**Tilted window statistics** is under **Image processing → Statistics**.
Set **Windows** to a list of `[x, y, width, height]` entries. The first
two values give the top vertex on the integral grid. The width steps
down-right and the height steps down-left, each at 45 degrees. The grid
extends to the image width and height. Every vertex must be inside the
grid. A window contains `2 × width × height` pixels before selection.

A mask or region limits which pixels contribute. The **Measurements**
output gives the input-grid vertices, sample count, and each channel's
sum, mean, and population standard deviation in raw sample units. Alpha
is excluded by default. Empty windows have zero sum and null mean and
standard deviation. An optional **Windows** input can supply up to 1,024
windows in place of the configured list.
[OpenCV tilted integral image reference](https://docs.opencv.org/5.0/main_modules/imgproc_misc.html#integral)

```powershell
opencv-dataflow-run "examples/custom tools/tilted_statistics/tilted_statistics.json"
```

[All examples](../../README.md)
