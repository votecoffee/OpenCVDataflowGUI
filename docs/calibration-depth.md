# Toolbox, histogram, calibration and depth workflows

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

1. Load/capture a chessboard image. Set **columns and rows of inner corners**,
   measured grid spacing and units (for example 10 mm). A 10×7 square board has
   9×6 inner corners.
2. If necessary, undistort the image first using a matching lens profile.
3. Run Calibration grid. It detects corners, fits a planar homography, and anchors
   the selected `(origin_column, origin_row)` corner at physical `(0,0)`. +X follows
   increasing detected columns; +Y follows increasing detected rows. The overlay
   marks the origin. `reverse_order` reverses both corner axes.
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

An unmarked symmetric chessboard cannot uniquely identify a fixed physical corner
after arbitrary board rotation. The current origin is detector-relative and must
be checked using the overlay and configured orientation. A marked, uniquely
identifiable target is planned under the camera/robot milestone. No undistortion
is silently applied by the grid tool: calibrate and measure in the same image space.

## Lens distortion

Calibrate lens accepts a JSON `image_paths` list with at least three distinct grid
views of identical dimensions. Use substantially varied board orientations and
positions. It detects the grid in every image and solves a standard OpenCV pinhole
model. Outputs include camera matrix, distortion coefficients, image dimensions,
units, view count and reprojection RMS in pixels. Duplicated detected views and
failed detections are rejected; excessive RMS produces an inspection failure.
Pose diversity beyond exact-duplicate rejection remains the user's responsibility.

Connect the calibration output to Undistort lens along with the image. The tool
checks matrix validity, image dimensions and its reprojection-error limit. Save
and reload the lens profile to avoid solving the dataset on every runtime cycle.
The output retains the input dimensions and uses the original camera matrix.
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
