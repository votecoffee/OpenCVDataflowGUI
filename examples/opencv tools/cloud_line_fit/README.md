# Native 3D line fitting

Open [cloud_line_fit.json](cloud_line_fit.json) and run once to fit a measured
line through interleaved clutter. **Fit 3D line to point cloud** uses OpenCV's
XYZ `fitLine` with Welsch loss. The supplied fixture pose orients the result
along its +Z direction. Debug mode shows the fitted segment and the accepted
and rejected rows.

The original synthetic PLY has 120 support rows and 22 clutter rows with
source IDs. The fit recovers all 120 supporting rows, a **179.000 mm**
observed span, and about **0.022°** angular error relative to the independently
specified axis. [least_squares_control.json](least_squares_control.json)
uses the same cloud and tolerance with L2 loss. It deliberately fails the
support check.

## Use the block

- Connect a point cloud, optionally after isolating the feature or sampling
  irrelevant surrounding geometry. The outputs use the cloud's named frame
  and units. For example, enter 0.6 for a 0.6 mm threshold on a millimeter cloud
  or 0.0006 for the same threshold on a meter cloud.
- Choose L2, L1, L12, Fair, Welsch or Huber. The fitter centers the sample and
  expresses it in units of the inlier tolerance before calling OpenCV.
  Fair, Welsch and Huber can use an explicit scale in cloud units. Zero uses
  the native default in the normalized coordinates. With the sample budget
  enabled, the seeded fit subset is smaller, while classification still
  evaluates every input row.
- The line's endpoints are the projections of the extreme accepted samples.
  The span describes observed support. The line's mathematical sign is
  arbitrary until a reference pose resolves it. Without a reference, the
  dominant XYZ component is positive and `direction_is_oriented` is false.
  The block rejects a reference whose +Z is too nearly perpendicular.
- Set support thresholds for inlier count, fraction, axis dominance and span.
  Weak geometry withholds measurement outputs and keeps the report. Accepted
  and rejected clouds retain their attributes and input order. Row-index
  lists refer to the current input, while `source_scan` and `source_index`
  retain scan identity.

The original fixture is licensed under the repository's MIT license. Its
[reference record](assets/truth.json) gives the axis, expected limits and
support IDs. Rebuild the fixture and both projects in an empty directory.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/cloud_line_fit/build_example.py" --output <scratch-directory>
```

[OpenCV line fitting reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html#fitline)
