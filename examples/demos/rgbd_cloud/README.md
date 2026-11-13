# Typed 3D results and the point-cloud viewer

Open [rgbd_cloud.json](rgbd_cloud.json) and run once. Synthetic RGB-D input
provides the depth and color; this tour needs no camera or external scan.

1. Follow **Depth to point cloud**, **Filter point cloud** and
   **Voxel-sample point cloud** to see native cloud connections between blocks.
2. Select a cloud-producing block and inspect its 3D result/thumbnail. Open the
   native cloud popout for a larger view and try its flyaround controls.
3. Inspect **Measure point cloud** for values and **Estimate point-cloud normals**
   for another cloud result. **Render point cloud** also provides a raster view.

This tour introduces 3D payloads and viewing alongside the ordinary image/data
ports. Algorithm-specific cloud processing examples remain under OpenCV tools.
[Demos tour](../README.md) · [All examples](../../README.md)
