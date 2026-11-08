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

## Compare edge search bands

Open [bands-project.json](bands-project.json) with SimplePathWidth
0.1.0.dev10 or later. Both blocks use plane-normal scans and the same
physical limits. The fixed pixel windows approximate the physical
windows near one quarter of the loop. The projected windows follow
the part's local image scale around the full loop.

| Edge search bands | Stations measured | Verdict |
| --- | ---: | --- |
| Fixed pixel bands | 26 of 161 | Fail, low coverage |
| Calibrated millimeter bands | 161 of 161 | Pass |

Open **Train path width…** on the calibrated block to see **Nominal
width physical** (6 mm) and **Edge search radius physical** (1.2 mm
on each side of the expected edge). Run once, attach **Use wired**
calibration, then select **Preview on reference** to try a different
radius. The pixel scan span must contain both projected windows.

With SimplePathWidth 0.1.0.dev11 or later, open the same preview on
**Fixed pixel edge bands** and click along the trend or step through
stations. At a missed station the shaded windows can fall away from the
gradient peaks. At a measured station cyan lines mark the selected edges.
The dashed line on the image marks the sampled cross-section.
