# Typed 3D results and the point-cloud viewer

Open [rgbd_cloud.json](rgbd_cloud.json) and run once. A real T-LESS Primesense capture
provides RGB, measured depth and publisher intrinsics; no camera or new scan is needed.

1. Follow **Depth to point cloud**, **Filter point cloud** and
   **Voxel-sample point cloud** to see native cloud connections between blocks.
2. Select a cloud-producing block and inspect its 3D result/thumbnail. Open the
   native cloud popout for a larger view and try its flyaround controls.
3. Inspect **Measure point cloud** for values and **Estimate point-cloud normals**
   for another cloud result. **Render point cloud** also provides a raster view.

This tour introduces 3D payloads and viewing alongside the ordinary image/data
ports. Algorithm-specific cloud processing examples remain under OpenCV tools.
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real T-LESS scene 1/frame 220 (CC BY 4.0), with its original 720×540 RGB
and depth. Publisher depth scale is 0.1 mm/sample: the loader uses 0.0001 m/sample.
Intrinsics are taken from that frame's `scene_camera.json`. The initial sampled
cloud has 23,921 points; all coordinates remain in the camera frame.

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
