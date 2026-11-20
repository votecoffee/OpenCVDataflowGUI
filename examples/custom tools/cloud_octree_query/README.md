# Point-cloud octree queries

Open [the project](cloud_octree_query.json) and run it once. It loads a colored
PLY in millimeters and queries both five nearest neighbors and an inclusive
125 mm radius on the same explicit cube. Inspect **Report** for original row
indices and Euclidean distances in meters. The **Neighbors** point cloud
preserves source colors and orders selected rows by their original IDs.

The input has 165 rows. The explicit cube includes 164 and excludes one.
Rows 40 and 162 share XYZ and keep distinct colors and row indices. The first
radius query has more hits than its 12-neighbor limit and marks the result
as truncated. The second includes row 163 exactly 125 mm away on the
upper boundary of the cube.

The block builds a sparse octree from occupied cells for each run.
Depth and resolution control subdivision. Input, query and result limits
bound the index and returned data. OpenCV 5.0.0 exposes native octree methods.
In the tested wheel, the bulk depth constructor rejected boundary points
and KNN failed to return usable point attributes for recovering original row
IDs. The block uses the exact sparse implementation for stable indices.
[OpenCV octree reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1Octree.html)

The synthetic fixture is MIT-licensed with the repository. Run
`build_example.py --output <empty-folder>` to regenerate it.

```powershell
opencv-dataflow-run "examples/custom tools/cloud_octree_query/cloud_octree_query.json" --profile lean
```
