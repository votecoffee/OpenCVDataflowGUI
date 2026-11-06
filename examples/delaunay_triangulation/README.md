# Delaunay triangulation

Open [delaunay_triangulation.json](delaunay_triangulation.json) with
**File → Open…** and click **Run once**. The project detects five dots,
triangulates their centers, and overlays the eight unique edges of the
four triangles on the source image.

Find **Delaunay triangulation** under **Inspection → Measurements**.
Connect an `[x, y]` point list or a blob, contour, or line collection with
measured centers. The block accepts 3 to 10,000 input sites with finite
coordinates within ±10 million. Exact duplicates use their first occurrence.
The remaining sites must span at least 0.000001 pixels and form a
non-collinear set that OpenCV can resolve.

- **Points** lists the unique sites in input order. **Source indices** gives
  their zero-based positions in the original point list or collection.
- **Triangles** and **Edges** use zero-based indices into **Points**.
  Indices within each entry are sorted. **Edge lengths** is in the same
  order as **Edges**, measured in the input coordinate units.
- **Result graphics** contains drawable edges. The example enables graphics
  on the measurement block so the overlay appears in Debug and Lean.

```powershell
opencv-dataflow-run examples/delaunay_triangulation/delaunay_triangulation.json
```

[OpenCV planar subdivision](https://docs.opencv.org/5.0/main_modules/geometry_subdiv2d.html) · [All examples](../README.md)
