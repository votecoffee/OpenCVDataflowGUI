# Rotated rectangle overlap

Open [rotated_rectangle_overlap.json](rotated_rectangle_overlap.json) with
**File → Open…** and click **Run once**. **Find contours** locates the part.
**Minimum-area rotated rectangle** fits its footprint. **Rotated rectangle
overlap** compares the fit to an offset reference and shows both outlines
and the intersection. The example measures an IoU of about 0.652 and
passes the minimum of 0.6.
Raise **Minimum IoU** above the measured value to see the intersection
turn red and the block report a failure reason.

Both blocks are under **Inspection → Measurements**. The fit accepts
2D points or a detected collection and item index. Connect either
rectangle to the overlap block or configure its center, width, height,
and angle in the editor. The angle is measured clockwise from the
positive image X axis to the rectangle's width axis. Both rectangles
and the optional preview image must share a pixel coordinate frame.

**Measurements** reports the area of each rectangle, the intersection
area, intersection-over-union, and the fraction of each rectangle covered.
The classification is disjoint, touching, partial, or contained. Equal
rectangles are included in the contained case. **Points** contains the
intersection vertices, including shared corners or edges. A shared
corner or edge alone has zero area. The rectangle outlines and
intersection are separate result graphics layers.
[OpenCV shape fitting reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html)

```powershell
opencv-dataflow-run examples/rotated_rectangle_overlap/rotated_rectangle_overlap.json
```

[All examples](../README.md)
