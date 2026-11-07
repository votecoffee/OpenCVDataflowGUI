# Contour hierarchy

Open [contour_hierarchy.json](contour_hierarchy.json) with **File → Open…**
and click **Run once**. The binary scene contains two outer boundaries and
three holes, including a hole inside an island. **Contour hierarchy**
reports six contours in all. The overlay uses green for outer boundaries,
orange for holes, and cyan for islands.

The **Hierarchy** output lists root IDs and a record for each contour. A
record contains its parent, first child, all direct children, previous and
next siblings, and nesting depth. Missing links are null. The same IDs
identify items in the **Collection** output, which also includes the contour
points and measurements. IDs are local to each run. [OpenCV contour hierarchy reference](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html#findcontours)

Supply a monochrome binary image. Every nonzero pixel is foreground. A
connected mask or region clips the foreground before extracting the tree.
This can change the topology at the selection boundary. Use **Approximation**
to choose how densely points are sampled along each boundary. Disable
**Draw overlay** to skip the image output while retaining the hierarchy and
collection.

```powershell
opencv-dataflow-run "examples/opencv tools/contour_hierarchy/contour_hierarchy.json"
```

[All examples](../../README.md)
