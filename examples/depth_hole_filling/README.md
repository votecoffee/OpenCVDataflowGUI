# Depth hole filling

Open [depth_hole_filling.json](depth_hole_filling.json) with **File → Open…**
and click **Run once**. The bundled uint16 depth image represents a plane
at 1.5 m. **Fill small depth holes** restores 29 samples in two enclosed gaps.
The larger gap and the gap touching the top edge remain invalid. The
**Filled mask** identifies the copied samples and **Remaining mask** shows
the 200 pixels still invalid. The final block projects the repaired plane
into a point cloud.

The block treats zero, negative, and nonfinite samples as invalid. It copies
an original valid sample into each eligible pixel using a Manhattan or
chessboard distance limit. **Maximum hole area** limits the connected
component size, and **Hole connectivity** chooses four or eight neighbors.
**Include border** is off by default. A mask or region limits where the
block writes. Donors and hole sizes come from the full depth image.
[OpenCV labeled distance transform](https://docs.opencv.org/5.0/main_modules/imgproc_misc.html#distancetransform)

The output preserves the source depth format and units. Filled pixels
are estimates copied from nearby measurements. Use the fill mask to
track or exclude those estimates in downstream inspection.

```powershell
opencv-dataflow-run examples/depth_hole_filling/depth_hole_filling.json
```

[All examples](../README.md)
