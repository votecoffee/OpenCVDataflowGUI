# Representative-point voxel sampling

Open [the project](voxel_representatives.json) and run it once. Both branches
group 11 measured points into six occupied voxels. The grid uses
10 mm along X, 20 mm along Y, and 30 mm along Z.

**Representative** returns the original row nearest each voxel's centroid.
Its **Indices** output lists rows `[0, 10, 3, 9, 5, 7]` in lexicographic
voxel order. The first two input rows are equidistant from their shared
centroid, so row 0 wins. The fixture also includes negative coordinates and
points exactly on X, Y, and Z cell boundaries. Colors and normals travel with
the selected rows. The block also preserves source pixels when supplied.

**Centroid** computes a position and average color for each occupied voxel.
The **First** mode retains the first input row in each voxel. Set **Spacing**
to **Uniform** to use a single size on all axes.

The pinned Python wheel omits OpenCV's C++ voxel sampler. The graph uses
a NumPy equivalent for representative selection. [OpenCV voxel sampling reference](https://docs.opencv.org/5.0/main_modules/_3d.html#voxelgridsampling)

The synthetic colored PLY is MIT-licensed with the repository. Run
`build_example.py --output <empty-folder>` to regenerate it.

```powershell
opencv-dataflow-run "examples/custom tools/voxel_representatives/voxel_representatives.json" --profile lean
```
