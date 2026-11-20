# Rasterize a triangle mesh onto a calibrated camera

Open [mesh_projection.json](mesh_projection.json) and run it. The 49 × 41
camera has different horizontal and vertical focal lengths and an off-center
principal point. A blue triangle crosses the left image boundary. An orange
triangle crosses the near and far planes and occludes part of the blue one.
The yellow triangle uses the opposite winding.

**Render all windings** covers 909 pixels. **Cull counterclockwise faces**
covers 840. Both branches use 0.55 m and 2.7 m clipping planes. Colors show the
nearest visible surface. Depth is optical-axis Z in meters and uncovered
pixels are NaN. The coverage mask is 255 on covered pixels and zero elsewhere.

Use rectified pinhole camera intrinsics. A mesh in another named frame
requires a metric transform to the camera frame. Choose combined, color-only
or depth-only output. Shading can be interpolated RGB, first-vertex RGB,
or white. The block clips in calibrated camera space before invoking the
corresponding OpenCV rasterizer and limits triangle and pixel work.
[OpenCV triangle rasterization API](https://docs.opencv.org/5.0/main_modules/ptcloud.html)

The PLY and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
