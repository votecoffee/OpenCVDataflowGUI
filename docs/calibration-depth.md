# Toolbox, histogram, calibration and depth workflows

![Toolbox tree, histogram and grid calibration](images/calibration-toolbox.png)

## Block classification

The left toolbox is a tree: **type → subtype → block**. Input, Output,
Image processing, Inspection and Calibration are examples of types. G-API,
Statistics, Lens distortion, Color balance, and Grid and coordinates are subtypes.
Search matches every level. Double-click a leaf to add an instance; group entries
only expand/collapse. Custom block classes declare `block_type` and `subtype`;
older plugins fall back to their existing `category` and the General subtype.
Stable `type_id` values still identify implementations in saved projects.

Calibration also includes editable ToolBlock templates. Open their internal graph
to configure primitive tools. They use the same public-port and result contracts
as other reusable ToolBlocks; category/subtype metadata survives export/import.
Their inspector allows editing type/subtype before exporting a reusable definition.

## Histogram

Connect a monochrome or color image to Histogram. It returns structured bin counts
and edges per channel, plus a plot image. Channel labels honor GRAY, BGR, RGB,
BGRA, RGBA, HSV or Lab encoding. Alpha is excluded unless enabled.

Choose 2–4096 bins. Automatic ranges use the full unsigned integer range (HSV8 hue
uses 0–180) or each float channel's finite minimum/maximum. Manual ranges use the
configured minimum/maximum for every channel. Nonfinite values are omitted; each
channel reports finite samples and samples outside the range. The upper edge is
inclusive, following NumPy histogram semantics. Plot curves are independently
scaled to their peak; compare numeric counts rather than line height across channels.

## Grid origin and physical coordinates

1. Load a calibration target image. For chessboards, set **columns and rows of
   inner corners**. A 10×7 square board has 9×6 inner corners. For circle grids,
   set the number of **circle centers** per row and the number of rows.
   Configure the measured spacing and units for the selected target.
2. If necessary, undistort the image first using a matching lens profile.
3. Run Calibration grid. It detects features, fits a planar homography, and anchors
   the selected `(origin_column, origin_row)` feature at physical `(0,0)`. +X follows
   increasing detected columns and +Y follows increasing detected rows. The overlay
   marks the origin. `reverse_order` reverses both axes.
4. Connect `mapping` to Pixel to physical. Supply a list of pixel `[x,y]` points
   through its optional `points` port or the JSON `pixel_points` parameter.
   Outputs include input pixels, physical coordinates, units and `plane_only`.
5. Set physical-space fit-error thresholds appropriate to the application. A grid
   fit exceeding its threshold is an inspection failure; Pixel to physical rejects
   mappings above its own configured threshold.
6. Connect a mapping to Save calibration profile, choosing a new JSON path. In a
   later job, use Load calibration profile → Pixel to physical without detecting
   the grid again. Save rejects an existing file; disconnect saving for repeated runs.

This mapping applies to the calibration **plane**, at unchanged camera pose, optics,
image size and preprocessing coordinates. It is not a 3D camera/robot extrinsic
transform. Points off that plane have parallax error; extrapolation beyond the
observed board needs independent verification. Pixel data alone does not identify
the source image resolution: the caller must use the mapping in its recorded
`image_size` coordinate space. A low fitting residual is not a held-out accuracy test.

On an unmarked target with rotational symmetry, the origin is detector-relative.
Use the overlay and orientation setting to identify the physical origin.
A marked, uniquely identifiable target is planned under the camera/robot milestone.

### Circle targets

Select `circles_symmetric` for a regular array or `circles_asymmetric` for
staggered rows. For a symmetric target, spacing is the center pitch on both
axes. For a staggered target, spacing is the vertical row pitch and half
the horizontal center pitch. The `corners` port carries the detected centers
in circle modes, so it can connect directly to Pixel to physical.

Set **Circle polarity** to match the dots. The circle controls tune candidate
size, separation, thresholds, and shape filters. Clustering is intended
for uncluttered scenes with perspective distortion. If it misses an
even-row staggered board, try detection with clustering disabled.
[Circle grid example](../examples/circle_grid_calibration/README.md) ·
[OpenCV calibration patterns](https://github.com/opencv/opencv/blob/5.x/doc/tutorials/calib3d/camera_calibration_pattern/camera_calibration_pattern.markdown)

## Lens distortion

Calibrate lens accepts a JSON `image_paths` list with at least three distinct grid
views of identical dimensions. Use substantially varied board orientations and
positions. It uses the same detector and target spacing settings as Calibration grid
and solves a standard OpenCV pinhole
model. Outputs include camera matrix, distortion coefficients, image dimensions,
units, view count and reprojection RMS in pixels. Duplicated detected views and
failed detections are rejected; excessive RMS produces an inspection failure.
Pose diversity beyond exact-duplicate rejection remains the user's responsibility.

Connect the calibration output to Undistort lens along with the image. The tool
checks matrix validity, image dimensions and its reprojection-error limit. Save
and reload the lens profile to avoid solving the dataset on every runtime cycle.
The output retains the input dimensions and uses the original camera matrix.
For reusable correction maps, connect the lens profile to **Build lens
correction maps** and wire both map outputs to **Remap image coordinates**.
Use a source image from the camera and input pixel grid used for calibration.
Remap validates the map pair and assigns the corrected image a separate
coordinate frame. The map builder also returns the output camera matrix
in a zero-distortion profile. [Lens correction maps example](../examples/lens_correction_maps/README.md)

Wire the map pair to **Inspect lens map displacement** to see correction
strength and direction, measure displacement in source pixels, and identify
samples outside the source. [Lens map displacement example](../examples/lens_map_displacement/README.md)

Connect both maps to **Save lens correction maps** to publish a versioned
`.npz` file. **Load lens correction maps** restores the pair, its coordinate
metadata, the rectified profile and any valid ROI. Use **Project** path base
for a portable saved graph. Saving requires a new file name.
[Lens map files example](../examples/lens_map_files/README.md)

Fisheye models, uncertainty estimates, dataset management and independent validation
are future work. Current checks use offline synthetic geometry; they do not prove
accuracy for a particular physical camera.

## White and color balance

White / color balance supports gray-world gains from the whole image or a JSON
`roi` of `[x,y,width,height]`. Choose a neutral reference region for white balancing;
gray-world over a colored scene does not necessarily estimate the illuminant.
Manual mode applies explicit red/green/blue gains. Integer outputs saturate at
their dtype maximum; float outputs preserve amplified numeric values. Alpha is
preserved. This is channel-gain correction, not a full color-chart/ICC calibration.

## Depth and RGB-D

Depth image file decodes a single-channel uint16 or floating image. Set
`units_per_sample` and physical units explicitly: counts in millimeters can use
0.001 with units `m`, or 1.0 with units `mm`. Values are stored as float32 physical
depth. Zero, negative and nonfinite samples become NaN. The distinct `depth` port
preserves units and invalid values through snapshots and nested ToolBlocks.

Depth is separate from both color and alpha. Pair color and depth checks matching
dimensions and passes both planes independently. It performs **no registration or
synchronization**; the caller must supply corresponding aligned observations.
Sample depth accepts whole-pixel coordinates and returns physical depth values,
using JSON `null` for invalid samples. It rejects coordinates outside the image.

Node/inspector previews use a configured physical display range and a false-color
palette; invalid samples are black. The preview is separate from raw depth values.
Depth ports cannot accidentally wire into ordinary color-processing inputs.
Depth-file score reports the fraction of valid samples; a plane without any valid
depth returns an inspection failure while retaining its diagnostic output.
Point-cloud projection, XYZ/normal types, 3D viewing, RGB-D device adapters and
registration are future work rather than an interpretation of a fourth color channel.

## Future milestones

- [Point clouds and 3D vision](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/1):
  [point-cloud types, files, projection and inspection](https://github.com/votecoffee/OpenCVDataflowGUI/issues/1).
- [Camera frames and robot integration](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2):
  [managed intrinsic/extrinsic profiles and offline hand-eye calibration](https://github.com/votecoffee/OpenCVDataflowGUI/issues/2),
  then [robot pose capture and calibrated measurement exchange](https://github.com/votecoffee/OpenCVDataflowGUI/issues/3).

The current lens solver estimates basic camera intrinsics as needed for distortion
correction. The future intrinsic/extrinsic milestone adds managed coordinate frames,
transform provenance and robot integration. Hand-eye calibration and live robot
integration are not implemented in this prototype.

Algorithms follow [OpenCV calibration and 3D reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).
