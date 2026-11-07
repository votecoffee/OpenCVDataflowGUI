# Enclosing convex polygon

Open [enclosing_polygon.json](enclosing_polygon.json) with **File → Open…**
and click **Run once**. The two branches fit an enclosure to the same
five-sided part. The triangle covers 2,400 square pixels. The quadrilateral
covers 1,950 square pixels. The input hull covers 1,600 square pixels, so
its fill ratio rises from about 66.7% to 82.1% with the extra side.

Find **Enclosing convex polygon** under **Inspection → Measurements**.
Connect a point list or a contour collection. Set **Maximum sides** from
3 to 16. If the hull fits within the limit, the block returns it directly.
A limit of 3 uses the minimum-area triangle solver. Larger limits allow
the native polygon search, which is capped at 256 hull vertices. The block
checks the candidates for containment and keeps the tighter valid result
from the polygon search and the minimum-area rectangle.

**Measurements** identifies the selected method and reports the number of
sides, enclosure and hull areas, and fill ratio. The example enables graphics
on both measurement blocks so their outlines appear in Debug and Lean.
[OpenCV enclosing polygon reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html#minenclosingconvexpolygon)

```powershell
opencv-dataflow-run "examples/opencv tools/enclosing_polygon/enclosing_polygon.json"
```

[All examples](../../README.md)
