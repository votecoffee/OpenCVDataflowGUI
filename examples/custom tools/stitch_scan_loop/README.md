# Close the loop on an inspection scan

Open [stitch_scan_loop.json](stitch_scan_loop.json) with **File → Open…**
and run all five images. The synthetic views follow a rectangular path
over a textured field. The fifth view overlaps the first, with a known
offset of 10 pixels down. A slight shear in the second view creates
drift in adjacent registration.

**Adjacent registration** and **Close the scan loop** use the same batch
and blend settings. Compare the final **Layout** transforms relative to
the first view. In the translation example, the last view lands about
1.71 pixels too far right with adjacent registration. Loop closure
reduces that error to about 0.002 pixels while keeping the first view
as the coordinate reference. **Measure covered pixels** uses the
corrected mosaic's coverage mask.

- Enable **Loop closure** on **Stitch scan sequence** when you expect the
  last view to overlap the first. The ordered batch needs three to eight
  views. Translation and similarity registration both support closure.
- The block fits the last view directly to the first and spreads the
  correction across intervening views by batch order. **Max loop drift
  pixels** bounds how far any view corner can move in first-view
  coordinates. The block rechecks adjacent reprojection error and valid
  overlap after correction.
- **Layout → closure** reports the direct fit, the accumulated drift
  before adjustment, the largest adjustment, and the valid overlap.
  If the fit or overlap falls short, the block reports a failure and
  withholds the mosaic.

```powershell
opencv-dataflow-run "examples/custom tools/stitch_scan_loop/stitch_scan_loop.json" --all-images
```

[All examples](../../README.md)
