# Run-linked contours

Open a project with **File → Open**, then **Run once**. No plugins or downloads
are needed. These examples demonstrate a lower-allocation native contour path
with explicit flat-list and run-polygon semantics.

| Project | Expected result |
| --- | --- |
| [Sparse comparison](sparse_compare.json) | Four known rectangles. Ordinary/run methods agree on areas and filled mask; image norm is zero. |
| [Nested boundaries](nested_boundaries.json) | Three boundaries; native areas 4/48/196 differ from ordinary 4/62/196. Run links are flat; the tree tool reports depths 0/1/2. |
| [Selection and offset](selection_offset.json) | Clip to a region, then shift reported coordinates: bounds `[530,-220,80,80]`, area 6241. Rasters stay on the input grid. |
| [Run budget guard](run_budget_guard.json) | Intentional validation error before native extraction; no processing output. |
| [Count failure](count_failure.json) | Intentional inspection FAIL: four retained boundaries versus five expected; measurements and a failure reason are emitted. |

Select the Run-linked contours block to inspect Data and Collection. List and
Simple are its only supported modes. Use ordinary contours for the ordinary
boundary convention or full hierarchy. Rasters are enabled in these examples
for review; the tool defaults them off to reduce allocation. Native polygon
areas are not foreground pixel counts.

Original synthetic masks/projects: **OpenCVDataflowGUI contributors, MIT**.
No customer or third-party images are included. Each canvas contains attribution
and expected geometry. [truth.json](truth.json) records coordinates, expected
areas and asset hashes. These fixtures demonstrate software behavior, not
physical metrology.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/run_contours/build_example.py"
.venv/Scripts/python.exe -m opencv_dataflow.assistant_cli run "examples/opencv tools/run_contours/sparse_compare.json" --summary
.venv/Scripts/python.exe "examples/opencv tools/run_contours/benchmark_native.py"
```

See [the boundary, resource and benchmark contract](../../../docs/run-contours.md).
