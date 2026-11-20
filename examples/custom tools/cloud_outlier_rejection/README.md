# Point-cloud outlier rejection

Open [the project](cloud_outlier_rejection.json) and run it once. Both branches
recover a folded panel from a colored PLY in millimeters. The source has
198 surface rows followed by six isolated noise rows.

- **Radius** retains points with at least three other rows inside 8 mm.
- **Statistical** measures the mean distance to six nearest other rows.
  The cutoff is the whole-cloud mean plus 1.5 population standard deviations.

Both branches retain the 198 surface rows, including the panel corners
and the raised flange. **Removed indices** lists `[198, 199, 200, 201, 202, 203]`.
The **Removed cloud** result holds those original rows and their measured
attributes. The report records the neighbor settings and observed cutoff.

The block uses batched SciPy queries. Radius mode requests only counts,
and statistical mode processes a bounded number of neighbor distances at
a time. [SciPy radius query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html) · [SciPy nearest-neighbor query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)

The synthetic fixture is MIT-licensed with the repository. Run
`build_example.py --output <empty-folder>` to regenerate the PLY and project.

```powershell
opencv-dataflow-run "examples/custom tools/cloud_outlier_rejection/cloud_outlier_rejection.json" --profile lean
```
