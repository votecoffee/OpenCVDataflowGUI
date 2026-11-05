# MSER region detection

Open [mser_regions.json](mser_regions.json) with **File → Open…** and
click **Run once**. The bundled image has a dark intensity minimum and a
bright maximum. The upper branch finds four stable regions across both
extrema. The bright-only branch finds two nested regions around the maximum.

**Detect MSER regions** accepts grayscale or color input. Grayscale mode
offers **Both**, **Bright**, and **Dark** polarity. Color mode exposes the
native evolution, diversity, margin, and edge blur controls. Both modes
expose the stability interval and region area limits.
[OpenCV MSER reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1MSER.html)

A connected mask or region keeps detections whose pixels all lie inside the
selection. The block returns bounds, area and center for each detection,
plus a union mask and an optional box overlay. Stable regions may overlap
or nest. **Max regions** caps the retained results in native detector order
and reports truncation in the **Regions** output.

```powershell
opencv-dataflow-run examples/mser_regions/mser_regions.json
```

[All examples](../README.md)
