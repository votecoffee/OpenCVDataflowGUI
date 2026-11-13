# Real T-LESS plane measurement review

| Capture | Role | Height (mm) | Plane angle (degrees) | Surface range (mm) |
| --- | --- | --- | --- | --- |
| [Scene 2 / frame 3](frame000003/project.json) | Training replay control | 48.4735 | 4.8380 | 16.9063 |
| [Scene 2 / frame 13](frame000013/project.json) | Independent capture | 48.0345 | 1.7039 | 21.0517 |
| [Scene 2 / frame 24](frame000024/project.json) | Independent capture | 45.1649 | 0.1191 | 22.0731 |

Each project loads a real first-return point cloud. A **Supplied annotation
fixture** node maps object-6 coordinates into the capture. It is provided
ground truth for this measurement workflow, not a detected pose. No object
mask, CAD samples, invisible backsides or synthetic sensor returns are added.

One recipe teaches the table and part-surface regions from frame 3. The same
regions, learned physical fit bands and default 5-degree parallel guard run
on frames 13 and 24. No per-capture tuning is performed. Fit bands are roughly
3.2 mm for the table and 3.3 mm for the part; the part region includes nonplanar
features. Height varies by about 3.3 mm across the views. There is no supplied
independent dimensional gauge truth and no acceptance limits. A graph pass
means measurements are available; it is not a dimensional acceptance verdict.

Select the tool and inspect **Cloud / fit** in the native flyaround thumbnail
or popout. In **Review measurement recipe…**, select `part_surface` to focus
its inliers, exclusions, ROI and range graphics. Toggle scene/regions to inspect
the fit closely. The value/status table and structured report distinguish
available values, insufficient support and datum extrapolation. A projected
datum foot outside the observed table bounds is reported; bounds alone do
not prove visibility or support.

Each folder retains a portable training recipe, input cloud, runnable graph,
expected numerical report and RGB result PLY. `reference.npz` and
`configuration.json` permit independent CLI retraining. `comparison.json`
keeps all three reports. Original RGB/depth images and camera/pose metadata
remain byte-exact under `source/` with publisher hashes. The subset manifest
records hashes; the full upstream test archive was not downloaded.

[T-LESS authors, CC BY 4.0 license and transformations](ATTRIBUTION.md).
This is a development workflow/coverage example; E3 physical qualification
and interactive region teaching remain open.
