# Contour approximation

Open [contour_approximation.json](contour_approximation.json) with **File → Open…**
and click **Run once**. **Find contours** captures the full boundary of the
bundled chamfered part. **Approximate contour** simplifies its 376 sampled
points to six vertices at a 2-pixel tolerance. Its overlay shows the simplified
polygon on the original mask.

Connect a contour or blob collection and select its **Item index**, or supply
an ordered list of `[x, y]` points. **Closed** joins the last vertex to the first.
For an open path, clear **Closed** and provide distinct endpoints. Tolerance can
be a fixed pixel distance or a fraction of the original perimeter or path
length. [OpenCV shape analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html#approxpolydp)

The block outputs the simplified points, vertex count, length, area for a
closed polygon, and the applied pixel tolerance. A closed polygon that collapses
at the chosen tolerance returns a failed result with the remaining vertices.
Connect an aligned image to view the overlay. The simplified path is also
available as result graphics.

```powershell
opencv-dataflow-run examples/contour_approximation/contour_approximation.json
```

[All examples](../README.md)
