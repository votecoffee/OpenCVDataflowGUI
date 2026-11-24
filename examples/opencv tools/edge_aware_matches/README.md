# Edge-aware interpolation of image matches

Open [edge_aware_matches.json](edge_aware_matches.json) and choose **Run Once**.
The block interpolates 252 supplied point pairs and reports source-to-target
displacement in pixels. The panels move in opposite directions with a small
vertical stretch. The visualization uses color for direction and brightness
for magnitude.

The edge-aware fit averages about **0.08 px** of error on the 8-pixel strip
just left of the seam. Setting **Geodesic lambda** to zero yields about **2.90 px**
on the same strip. Both measurements use the fixture's analytic
motion and exclude the top and bottom 18 rows.

The sample uses the block's configured point lists. In a graph with a descriptor
matcher, connect the **Matches** record and the query and train images. You can
also wire the two ordered point lists. The block validates available match
frames and tickets and reports distinct seed counts and fit error at the
supplied controls. Its support mask identifies in-bounds target coordinates
within the output selection.

The synthetic images are original MIT assets derived from the repository's
[sparse-to-dense fixture](../sparse_to_dense_flow/README.md).
Run `build_example.py --output <directory>` to regenerate the images and project.
