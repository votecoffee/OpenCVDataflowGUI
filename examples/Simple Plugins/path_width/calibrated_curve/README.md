# SimplePathWidth · Plane-normal scans on a curved bead

Open the [project](project.json) with SimplePathWidth 0.1.0.dev8 or later.
Run once. Both branches use the same synthetic image, taught path,
calibration, and physical tolerances. The bead is 6 mm wide with a
signed offset of −1 mm along the taught path.

Select each measurement block and compare **Result / overlay** and
**Result / physical_trend**. The image-normal branch intentionally fails
on some stations in this skewed view.

| Scan direction | Stations measured | Stations out of tolerance | Mean absolute width error |
| --- | ---: | ---: | ---: |
| Image normal | 161 | 21 | 0.093 mm |
| Calibrated plane normal | 161 | 0 | 0.030 mm |

To switch a recipe, open **Train path width…** and change **Cross-scan
direction**. In the trainer, select **Use wired** under **Preview
calibration** after running the graph, then preview the reference.
Stage the recipe and apply the inspector changes when ready.

The normal is determined on the calibrated plane and projected into the
image before edge detection. Search distances and sampling still use
pixels. The example uses generated geometry and a known mapping.
