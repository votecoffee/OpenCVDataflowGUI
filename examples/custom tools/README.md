# Custom tools

This category demonstrates project-defined algorithms, inspection/metrology
tools and integrations with libraries other than OpenCV or scikit-image.
Examples include flat-field/reference-shadow models, contrast and slanted-edge
measurements, trained shape patterns, region construction/transforms, controlled
distortion fixtures, LightGlue and Annoy. Check each README for dependencies.

The [complete custom-tool index](../README.md#custom-tools) lists these examples.
Native OpenCV operations and module workflows are under
[OpenCV tools](<../opencv tools/README.md>), while scikit-image examples are under
[skimage tools](<../skimage tools/README.md>). Guided editor/runtime tours are
under [Demos](../demos/README.md). Python plugin authoring examples remain in
[user_tools](../README.md#user-tools-and-plugins), and Simple Plugins keep their
[dedicated collection](<../Simple Plugins/README.md>).

[All examples](../README.md)

For external descriptor tensors, learned matching and explicit detector limits,
try [local features](local_features/README.md). The [SuperPoint v6 projects](superpoint_v6/README.md)
show trained extraction and planar matching with a known shift.

For real learned extraction, known transforms and separate PCB photographs,
try [DISK and LightGlue](disk_matching/README.md).

For ALIKED presets, native subpixel/fallback controls and matching local weights,
try [ALIKED and LightGlue](aliked_matching/README.md).

For image measurements on a shared pixel or physical grid, try [profile resampling](profile_resampling/README.md).
For conditional motion-kernel estimation, separate validation and gated restoration, try [motion blur estimation](motion_blur_estimation/README.md).

[Scan-to-mesh inspection](scan_mesh_comparison/README.md) measures a scan
against a reference mesh and separates open boundaries from defects.

[Cloud dimensional comparison](cloud_dimensional_comparison/README.md) reports
metric offsets, row-level defects and bidirectional overlap.

[Registered scan merge](cloud_scan_merge/README.md) combines mixed-unit
scans with overlap suppression and source row provenance.

[Triangle mesh interchange](mesh_interchange/README.md) compares matching
OBJ and PLY topology and vertex attributes.

[OBJ point-cloud interchange](cloud_obj_interchange/README.md) compares two
file formats for the same colored, oriented scan.

[Organized PCD interchange](organized_pcd/README.md) compares ASCII and binary
scans with missing pixels, color and normals.

[Organized depth mesh](organized_depth_mesh/README.md) shows how an axial
depth-jump limit changes which neighboring pixels are joined into triangles.

[Oriented cloud extents](oriented_cloud_extents/README.md) measures a
rotated cuboid and compares PCA with known fixture axes.

[Point-cloud surface quality](cloud_surface_quality/README.md) estimates
oriented normals and curvature and reports sparse or ambiguous neighborhoods.

[Robust plane and sphere fitting](cloud_model_fitting/README.md) recovers
metric models and row-aligned residuals from two scans with known outliers.

[Normal-based cloud region growing](cloud_region_growing/README.md)
separates touching faces and isolates high-curvature seam rows.

[Euclidean cloud clustering](cloud_clustering/README.md) separates parts by
XYZ distance and filters components by point count and physical size.

[Point-cloud outlier rejection](cloud_outlier_rejection/README.md) compares density
rules and verifies retained corners and flange points on a colored PLY.

[Voxel representatives](voxel_representatives/README.md) compare measured rows
and centroids in the same per-axis voxel grid.

[Point-cloud sampling](cloud_sampling/README.md) compares coverage at an
equal point budget and preserves the source cloud's row IDs.

[Point-cloud octree queries](cloud_octree_query/README.md) check nearest and
radius results against known coordinates in a colored PLY.

[Time-of-flight depth cleanup](tof_depth_cleanup/README.md) flags low-confidence,
mixed-edge and isolated depth samples while preserving the raw exposure and true
depth step.
