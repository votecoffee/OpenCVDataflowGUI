# Convex contour approximation

Open [convex_contour_approximation.json](convex_contour_approximation.json)
with **File → Open…** and click **Run once**. The project detects a chamfered
part and compares two convex footprints. One reduces the outline to four
sides with an unlimited budget. The other keeps six sides within a 2%
added-area budget.

Find **Approximate convex contour** under **Inspection → Measurements**.
Wire a simple closed contour or select an item from a contour or blob
collection. Choose whether to start from the convex hull or require a
strictly convex input. Set **Target sides** to the exact count you want.
**Max added area percent** budgets the area added beyond the hull, as a
percentage of the source contour area. Use -1 for an unlimited budget.

Try a 1% budget on the four-sided branch. The block reports a failed
side-count judgment and shows the valid polygon it reached. If the native
polygon cuts through the source hull or otherwise fails geometry checks,
the block reports a failure and outputs the source hull for review. The
block supports up to 1,024 hull vertices. An optional image in the contour
frame provides an overlay. [OpenCV approximation reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html#approxpolyn)

```powershell
opencv-dataflow-run "examples/opencv tools/convex_contour_approximation/convex_contour_approximation.json"
```

[All examples](../../README.md)
