# Fuse three RGB-D views of a painted panel

Open [colored_tsdf_fusion.json](colored_tsdf_fusion.json) and run it. Three
synthetic recordings observe a sloped panel with red, blue and black
patches. The poses are in millimeters and the fused cloud is in the
`panel` frame in meters. **Fuse posed RGB-D frames (ColorTSDF)** exposes
colored points, normals and a report of admitted and confidence-rejected samples.

The analytic plane is `z = 1.023 + 0.06x − 0.03y` in meters. With
12.5 mm voxels, over 99% of the recovered points are within one voxel
of the plane on the tested OpenCV 5.0.0 wheel. The three views collectively
exclude 802 intentionally mismeasured, low-confidence magenta pixels.
The valid black patch remains black.

To use your own images, supply registered uint8 RGB or BGR, scalar
depth and the camera-to-world pose to **Pack posed RGB-D view**. Connect
a single view or a batch of up to 64 views to **Fuse posed RGB-D frames
(ColorTSDF)**. The views need one set of undistorted pinhole intrinsics.
Use **Calibrated grid** only when separate source roots are already
registered to that same grid. A positive confidence threshold requires
a confidence plane in every view.

The fixed-volume workspace check estimates native volume, extraction and
per-view raster memory. Frame storage and graph snapshots have separate
runtime costs. This example admits an estimated 42.4 MiB under its 64 MiB
limit. [OpenCV Volume reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1Volume.html)

The recordings and the [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
