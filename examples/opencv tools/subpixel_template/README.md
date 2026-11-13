# Subpixel template locations

Open [subpixel_template.json](subpixel_template.json) with **File → Open…**
and click **Run once**. Both branches find the same two targets. The lower
branch enables **Subpixel** and refines each target's center and pose using
the samples around its OpenCV template response peak. Inspect the **matches**
output on each branch to compare the coordinates.

| Known top-left (x, y) | Integer match | Refined match |
| --- | --- | --- |
| (23.34375, 18.625) | (23, 19) | (23.341, 18.627) |
| (84.25, 49.375) | (84, 49) | (84.248, 49.371) |

**Match template** also exposes **Subpixel**. When refinement succeeds, the
match metadata includes `subpixel_top_left`. Integer bounds and top-left
coordinates still identify the sampled patch. Scores, acceptance, overlap
suppression, and the response map use the original pixel-grid search.
When a peak is at the response or search-mask boundary, or its neighborhood
doesn't support a clear quadratic peak, the block keeps the integer-based
center. The option defaults to off for existing jobs.
[OpenCV template matching reference](https://docs.opencv.org/5.0/main_modules/imgproc_object.html#matchtemplate)

```powershell
opencv-dataflow-run "examples/opencv tools/subpixel_template/subpixel_template.json"
```

[All examples](../../README.md)
