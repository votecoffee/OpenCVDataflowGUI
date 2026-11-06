# Circle grid calibration

Open [circle_grid_calibration.json](circle_grid_calibration.json) with
**File → Open…** and click **Run once**. **Calibration grid** detects
24 dark circles in a 4-column, 6-row staggered target and marks the
configured origin and axes. **Pixel to physical** maps the detected
centers into millimeters. The selected center maps to `[0, 0]`.

In **Calibration grid** or **Calibrate lens**, select
`circles_asymmetric` for this board or `circles_symmetric` for a regular
array. This example uses 12.5 mm between rows, 25 mm between centers
on one row, and a 12.5 mm shift on alternating rows. The input
is a synthetic image and the mapping applies to the target plane.

```powershell
opencv-dataflow-run examples/calibration/circle_grid_calibration/circle_grid_calibration.json
```

[Calibration workflows](../../../docs/calibration-depth.md) · [All examples](../../README.md)
