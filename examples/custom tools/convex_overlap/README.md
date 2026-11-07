# Convex polygon overlap

Open [convex_overlap.json](convex_overlap.json) with **File → Open…** and
click **Run once**. The graph detects the contour of an L-shaped part,
forms its convex hull, and compares the hull to an editable reference
rectangle. The intersection covers 512.5 square pixels and the IoU is
about 0.313. The overlay shows the two outlines and their intersection.
The example enables graphics on the overlap block so the overlay is also
available in the Lean run profile.

Find **Convex polygon overlap** under **Inspection → Measurements**.
Enter vertex lists, wire point lists, or select an item from a wired
contour collection. **Ordered** takes convex boundary vertices in either
winding direction. **Convex hull** forms a hull from each input's points.
Inputs with coordinate frames must share a frame.

**Include containment** controls strictly nested polygons. Shared edges
and corner contacts have zero area and can return contact points.
**Check IoU** enables an overlap judgment. IoU is the intersection area
divided by the union area, while the two fractions in **Measurements**
give the proportion of each input covered by the intersection.
[OpenCV intersectConvexConvex reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html#intersectconvexconvex)

```powershell
opencv-dataflow-run "examples/custom tools/convex_overlap/convex_overlap.json"
```

[All examples](../../README.md)
