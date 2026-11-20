# Scanner point cloud to range image

Open `scanner_range_image.json` and run the graph. The supplied pose
places a synthetic building-frame PLY in the scanner frame. The six columns
span 165° through 225°, crossing the 180° seam. Three rows use unequal
elevation bands. The range, intensity, row IDs and return counts share
this angular grid.

Four returns fall into the same bin. The nearest branch selects 2 m
with intensity 10, the farthest selects 4 m with intensity 90, and the
strongest selects 3 m with intensity 90. Two equally strong 3 m returns
tie, so the lower source row wins. A single return with unknown intensity
falls back to nearest in strongest mode. The empty-view branch intentionally
fails with an all-invalid raster, so the headless project reports failure.

The scanner axes are +X forward, +Y left and +Z up. Azimuth advances from
+X toward +Y, elevation increases upward, and image rows run from high
elevation to low. Angular bins include their low edge and exclude their
high edge. Range is Euclidean distance from the scanner origin in meters.
Missing range and intensity use NaN, and missing row IDs use −1. Use the
validity mask to distinguish unsampled bins. Tagged spherical rasters
are accepted by image ports and are rejected by camera-depth ports.

The Debug profile shows a heatmap of selected range, with empty bins in
black. The asset is original synthetic geometry by OpenCVDataflowGUI
contributors under the repository's MIT license. Run `build_example.py
--output <empty directory>` to regenerate the project and PLY.
