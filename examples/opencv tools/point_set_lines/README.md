# Hough lines from sparse points

Open [point_set_lines.json](point_set_lines.json) with **File → Open…**
and click **Run once**. The synthetic image has a horizontal row of
23 points, a vertical column of 20 points, and four outliers. The two
line spans are ranked by voting strength and drawn on the image.

Find **Hough lines from points** under **Inspection → Features**.
Wire a list of `[x, y]` coordinates and the image that defines their
frame. An optional mask or region restricts which points vote.
**Minimum votes** is inclusive. Choose automatic or custom signed rho
bounds and configure the rho and angle bin sizes. Normal angles near
0° indicate vertical lines. Normal angles near 90° indicate horizontal
lines. [OpenCV point-set Hough reference](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html#houghlinespointset)

**Polar lines** reports native vote totals and the number of selected
points near each reported line. **Support distance** controls that
neighborhood. Zero uses half the rho bin spacing. The collection's
segment endpoints span the projected support points. Native peaks
without nearby points appear in the polar report only. Use
**Strength** in collection tools to filter or sort by votes.

```powershell
opencv-dataflow-run "examples/opencv tools/point_set_lines/point_set_lines.json"
```

[All examples](../../README.md)
