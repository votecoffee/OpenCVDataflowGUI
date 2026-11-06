# Voronoi cells

Open [voronoi_cells.json](voronoi_cells.json) with **File → Open…** and click
**Run once**. The project detects five dots and partitions the image around
their centers. The overlay highlights the selected cell and the statistics
block measures intensities using that cell's mask.

Find **Voronoi cells** under **Inspection → Measurements**. Connect a
reference image and either 1 to 10,000 `[x, y]` sites or a blob, contour,
or line collection with measured centers. Sites must be within the image
and use its coordinate frame. Exact duplicates use their first occurrence.

**Cells** lists each site's coordinates, zero-based source index, area in
square pixels, and polygon vertices. Polygons are clipped at the outer
pixel edges, from −0.5 to width or height minus 0.5. Their areas sum to
the image area within numerical precision.

**Cell index** selects a zero-based entry from the output list. **Mask**
is the rasterized selection in the reference image frame. Adjacent masks
can include the same boundary pixels. **Selected area** measures the
continuous polygon. The overlay has separate layers for all outlines and
the selected outline, and the example enables them in Debug and Lean.

```powershell
opencv-dataflow-run examples/voronoi_cells/voronoi_cells.json
```

[OpenCV planar subdivision](https://docs.opencv.org/5.0/main_modules/geometry_subdiv2d.html) · [All examples](../README.md)
