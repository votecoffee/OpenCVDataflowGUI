# Mesh simplification for bounded review

Open `mesh_simplification.json` and run in Debug to compare the input and
simplified wireframes. The first branch reduces a 544-triangle crowned
panel to 220 triangles within a 35 mm two-way surface displacement bound.
Its outer perimeter and rectangular cutout retain all 80 indexed boundary
edges. Every retained vertex keeps its original coordinates, RGB, normal and
texture coordinates.

The second branch applies a zero displacement limit to the same face budget.
It keeps the original geometry and reports an unmet budget, so the example as
a whole returns a failed inspection in the headless runner. Check
`surface_deviation_bound`, `output_faces` and `status` in each report.

The meshes are intended for bounded review. Retained normals and texture
coordinates are original samples, and the block checks local face winding
and normal change. Global self-intersections, volume drift and smooth
attribute interpolation are outside its qualification. See
[CGAL's discussion of edge collapse and bounded face-normal changes](https://doc.cgal.org/latest/Surface_mesh_simplification/).

The input is original synthetic geometry by OpenCVDataflowGUI contributors,
under the repository's MIT license. Use `build_example.py --output <empty
directory>` to regenerate the project and PLY fixture.
