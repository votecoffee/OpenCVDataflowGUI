# Mesh from posed depth

Install the optional `.[skimage]` extra in the application environment, then
open [tsdf_mesh.json](tsdf_mesh.json) and click **Run once**. The graph loads
a synthetic, tilted surface observed from a calibrated camera pose and
shows the reconstructed triangle wireframe. Select the fusion block to see
the mesh and fit report.

The reference plane is `Z = 1.02 + 0.14 X - 0.09 Y` meters in the world frame.
The example produces **3,663 triangles** and leaves an eight-pixel-wide
unobserved strip open. The maximum deviation from the plane is **1.94 mm**
along world Z. A separate automated test uses two views with different camera
positions and verifies the shared and single-view coverage.

## Use your own depth

Supply one depth image or a group of up to 32 depth images and one
camera-to-world pose per image in group order. Images must use the same
undistorted pinhole intrinsics. Depth may be in meters or millimeters.
For raw uint16, set the units and units per sample. Poses supply a shared
world-frame name, and the output mesh is in meters.

The block integrates truncated camera-Z distances on a bounded world grid.
Set the origin, spacing and XYZ sample counts to cover your region of interest.
Only cells with enough observations at all eight corners contribute triangles.
A smaller spacing improves sampling and increases memory and runtime. The
report includes observed voxels, surface cells and face count. The mesh
is withheld if no supported crossing is found or a resource limit is reached.

Connect **Save triangle mesh as PLY** to the mesh output and choose a new
`.ply` path to write vertices and triangle indices. The binary PLY includes
the world frame and units in a header comment. Relative paths use the project
folder, and existing files are preserved. The saved example performs no writes
during Run once.

The [fixture builder](build_example.py) recreates the project and depth image.
Original fixture by OpenCVDataflowGUI contributors under [MIT](../../../LICENSE).
[scikit-image marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes).
