# Keep gaps in an organized depth mesh

Open [organized_depth_mesh.json](organized_depth_mesh.json) and run it.
Both branches use the same calibrated 42 × 28 scan with an 800 mm surface,
a 1,040 mm surface, and a missing oval. Compare the two wireframes and
each block's report. The 20 mm depth-jump limit keeps the step open.
The 300 mm control adds 54 triangles across the step. The oval stays
open in both meshes.

**Mesh an organized depth grid** accepts a depth plane with matching
undistorted pinhole intrinsics or a metric organized cloud with unique
source pixels. An organized PCD uses its declared acquisition viewpoint.
For a cloud without a viewpoint, XYZ must already be in a camera optical
frame. The cloud path retains RGB when available. A mask or region can
exclude more samples on the source grid.

Each 2 × 2 cell is evaluated along both diagonals. The block uses the split
that keeps the most valid triangles, preferring a shorter diagonal if the
counts match. Invalid or excluded samples stay open. An optional maximum
XYZ edge length is independent of the axial depth-jump limit.

The mesh carries camera-facing unit normals and UVs at the source texel
centers. The report records the source grid, frame and calibration or
viewpoint. Recover the source pixel as
`floor(U × width), floor((1 − V) × height)`. Connect **Save triangle mesh**
to write OBJ or PLY. The tool processes grids of up to one million samples
and withholds the mesh if the triangle budget is exceeded.
[OpenCV depth backprojection reference](https://docs.opencv.org/5.0/main_modules/ptcloud.html)

The depth scan and [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
