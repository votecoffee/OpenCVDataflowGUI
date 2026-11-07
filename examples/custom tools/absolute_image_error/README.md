# Absolute image error

Open [absolute_image_error.json](absolute_image_error.json) with
**File → Open…**, then click **Run once**. Three **Compare image quality**
blocks measure the same pair of registered images. A large patch differs
by 40 sample levels and a smaller patch differs by 60.

| Metric | Value in sample levels |
| --- | --- |
| Mean absolute error (MAE) | 3 |
| Root mean square error (RMSE) | 11.8322 |
| Maximum absolute error | 60 |

The metrics are under **Inspection → Measurements → Compare image quality**.
They operate in raw sample units across the selected pixels and channels.
Connect a mask or region to limit the scalar measurement. The **Map**
output is the full float64 per-channel absolute difference for all three
metrics. The **Absolute error** preview stretches the largest channel
error at each pixel for display.

Enable **Check threshold** to flag a result when the selected error metric
exceeds the limit. The two source images must have the same dimensions,
channel layout, and pixel depth. [OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html#absdiff)

```powershell
opencv-dataflow-run "examples/custom tools/absolute_image_error/absolute_image_error.json"
```

[All examples](../../README.md)
