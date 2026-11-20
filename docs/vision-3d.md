# OpenCV 3D workflows

The **OpenCV** toolbox groups 3D tools by module. Start with **Camera Calibration
(calib)**, **Computational Geometry (geometry)**, **Stereo Correspondence (stereo)**,
or **RGB-D Processing (rgbd)**. Point-cloud file blocks are in the **Point clouds**
subgroups under **Custom > Input** and **Custom > Output**. The tested OpenCV
5.0.0 contrib build provides the dependencies for these tools.

See the [point-cloud API audit](opencv-scope-map.md#point-cloud-and-mesh-api-families)
for planned tools for imported inspection scans, depth-derived clouds and meshes.
The [native API crosswalk](opencv-scope-map.md#native-point-cloud-api-crosswalk)
identifies the Python bindings and where a tested fallback is needed.

## Try the included jobs

Open these projects with **File → Open…**.

| Project | What it demonstrates |
| --- | --- |
| [wide-angle stereo](<../examples/opencv tools/wide_angle_stereo/README.md>) | Fit fisheye and omnidirectional rigs, then validate held-out rectification and depth scale. |
| [stereo cloud](<../examples/opencv tools/stereo_cloud/stereo_cloud.json>) | Rectified 8 px disparity from a known baseline reconstructs a 3 m plane. |
| [RGB-D cloud](<../examples/demos/rgbd_cloud/rgbd_cloud.json>) | Metric depth, RGB color, pixel correspondence, XYZ filtering, voxel sampling and normals. |
| [fisheye calibration](<../examples/opencv tools/fisheye_calibration/README.md>) | Fit eight image views, validate a separate view and rectify with the fitted intrinsics. |
| [omnidirectional calibration](<../examples/opencv tools/omnidir_calibration/README.md>) | Fit the Mei lens and mirror parameter, check a held-out edge target, and rectify to a perspective view. |
| [3D pose](<../examples/opencv tools/pose_3d/pose_3d.json>) | Project known XYZ, recover a camera-relative pose, and triangulate the points. |
| [Sparse multi-view reconstruction](<../examples/opencv tools/sparse_multiview/README.md>) | Triangulate labelled features from four partial target views and refine their poses using two fixed anchors. |
| [structured light](<../examples/opencv tools/structured_light/structured_light.json>) | Generate Gray-code patterns and decode an identity camera/projector mapping. |
| [Phase-shift fringe replay](<../examples/opencv tools/phase_shift/README.md>) | Decode absolute subpixel projector X or Y from a six-exposure group with low-confidence gaps. |
| [Camera/projector reconstruction](<../examples/opencv tools/camera_projector/README.md>) | Fit a rig at different resolutions and reconstruct a held-out depth step. |
| [Robot-world and hand–eye](<../examples/opencv tools/robot_world/README.md>) | Jointly fit two poses and predict held-out board images with a fixed or robot-mounted camera. |
| [TSDF fusion](<../examples/opencv tools/tsdf_fusion/tsdf_fusion.json>) | Integrate a posed synthetic depth plane and inspect the reconstructed surface. |
| [TSDF mesh](<../examples/skimage tools/tsdf_mesh/README.md>) | Extract a metric triangle surface and preserve the gap in observed depth. |

Run once, then use **Results > 3D point-cloud viewer…**. Choose a retained cloud
output and adjust yaw, pitch and point size. Debug/Balanced retain actual cloud
results; Lean retains summaries in the editor while still passing the full cloud
inside the worker. The viewer is a CPU orthographic renderer that caps drawn
points; it does not modify or resample the processing output. **Render point
cloud** makes a regular preview image for the graph.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/stereo_cloud/stereo_cloud.json" --repeat 3 --profile lean
```

All examples use synthetic observations and require no cameras, projectors or robots.

## Camera models and coordinate frames

**Camera intrinsic model** defines focal lengths, principal point, size, distortion,
lens family, units and a frame name. Connect the existing Lens Calibration output
to its optional `calibration` port to reuse fitted intrinsics and name the camera
frame. The model is tied to one pixel grid; resized or rectified images need the
matching intrinsics. These wrappers use positive focal lengths and zero skew.
Pinhole distortion takes 0/4/5/8/12/14 coefficients; fisheye and omnidirectional
models take four, with `xi` for omnidirectional projection.

**Calibrate fisheye camera** fits intrinsics on the source grid with OpenCV 5.
Use checkerboard or circle-grid image paths, or connect a batch of identified
ChArUco observations. Each accepted camera is a `camera_model_v1` record with
four coefficients and zero skew. Its fit report includes the sensor area
spanned by the combined corner hull, per-view RMS, and target-to-camera
poses in target units. The RMS, worst-view limit and minimum hull coverage
gate the camera output. Save/load calibration profile blocks accept the camera
record. Keep a target image out of the fit and connect it and the fitted
camera to **Validate lens calibration**. That block fits a fresh target pose
with the intrinsics held fixed and reports the fisheye reprojection errors.
[Example and setup](<../examples/opencv tools/fisheye_calibration/README.md>)

**Calibrate omnidirectional camera** uses the same target inputs and fits the
Mei camera model, including mirror parameter `xi`. The fit report lists the
original indices selected by OpenCV and any omitted views. The default requires
all views. **Allow subset** accepts the retained group when it meets the minimum
view count, coverage and RMS limits. Save it as a camera profile or connect it
to **Validate lens calibration** or **Rectify pinhole / fisheye / omnidir**.
Held-out reprojection solves a fresh target pose from forward camera rays.
[Example and fit controls](<../examples/opencv tools/omnidir_calibration/README.md>)

Rigid poses use a `pose_3d_v1` record and a 4×4 **source-to-target** matrix:
`target_xyz = R @ source_xyz + t`. Rotation vectors use Rodrigues axis-angle in
radians, translations use the declared length units. Compose requires the first
target frame to match the second source frame; invert swaps both frames. These
are distinct from the 2D pose records used by pose-relative image regions.
Cloud transforms reject unit or frame mismatches instead of guessing conversions.

- **Project 3D points** handles pinhole, fisheye and omnidirectional projection.
  A 3×4 linear projection matrix is emitted only for undistorted pinhole geometry.
- **Undistort camera points** yields normalized rays or pixels. **Rectify pinhole /
  fisheye / omnidir** maps images to a perspective pinhole grid and emits new
  intrinsics. Nonlinear rectification starts a new image root coordinate frame.
- **3D object pose (PnP)** takes matched object/image points and pinhole intrinsics.
  It exposes PnP method, RANSAC, inlier/error thresholds, initial pose and LM/VVS
  refinement. A rejected fit does not emit a pose.
- **Fundamental matrix** and **Essential matrix / relative pose** estimate epipolar
  geometry. Relative pose uses one shared intrinsic model and returns **unitless**
  translation direction; it does not recover metric scale.
- **Triangulate 3D points** accepts corresponding undistorted points and projection
  matrices in the same world basis. It checks finite homogeneous coordinates,
  cheirality and reprojection error. The caller declares output units/frame from
  that projection basis.
- **ArUco 3D marker poses** detects markers and uses the specified physical edge
  length to return IDs, per-marker poses, reprojection errors and axes overlays.

**Versioned camera and rig profiles** use an explicit revision and check the
capture declaration and pixel grid before rectifying. Derived lens maps and
stereo reprojection preserve the selected revision.
[Setup and runnable examples](calibration-revisions.md)

## Review sparse stereo correspondences

**Check epipolar correspondences** accepts ordered left and right point
lists and a fundamental matrix for those views. It predicts the matching
line in each view and reports both perpendicular point-to-line distances
in pixels. A pair qualifies only when both distances meet the configured
tolerance. The **Inliers** list preserves the original pair order. Undefined
lines at an epipole are rejected and reported as null.

Use the **Minimum inlier ratio** to judge the pair set. Wire the source
images to view accepted pairs in green and rejected pairs in red. Each
overlay and graphics layer uses its own view's coordinate frame. A matrix
from **Fundamental matrix** carries the source point frames, so the review
can reject a mismatch such as using a raw-view matrix on rectified points.

**Score and refine epipolar correspondences** computes square-root Sampson
error in pixels for ordered stereo matches. Evaluate mode forwards the raw
pairs that meet the limit. Correct mode passes those pairs to OpenCV optimal
correction and checks the remaining residual, per-view displacement and
known image bounds. Undefined epipolar directions, including exact epipoles,
are rejected before correction. The report and **Inliers** mask follow input
order. **Indices** identifies the rows retained in the compact left and right
point outputs. Each output and preview keeps its own view's frame.
[Noisy stereo match example](<../examples/opencv tools/epipolar_refinement/README.md>)

## Stereo and depth

**Rectify uncalibrated stereo images** fits two pixel homographies from
matched points and a fundamental matrix. It accepts an optional inlier mask,
filters undefined and high-error correspondences, and uses the right image
dimensions as the default shared canvas. The report measures vertical fit
error and valid overlap. It rejects homographies with a projective horizon
crossing either source image. Each output image and support mask retains
its view's source mapping. **Common valid** is their intersection on the
left rectified grid. Use a calibrated stereo rig for metric reprojection.
[Held-out alignment and support example](<../examples/opencv tools/uncalibrated_stereo/README.md>)

**Calibrate stereo cameras** reports joint RMS and a capture-by-camera
breakdown for pinhole rigs. An optional per-view limit catches a single
bad capture with a low pooled error. **Validate stereo calibration** keeps
the intrinsics and relative pose fixed. It estimates a target pose from
one camera's observations and scores the projection in the other camera,
then swaps the roles. The report compares these cross-camera errors to the
single-camera pose fits for every held-out capture.
[Fit and held-out stereo example](<../examples/opencv tools/stereo_calibration_quality/README.md>)

**Calibrate multi-camera rig** fits three to eight pinhole cameras or three
to eight fisheye cameras. Supply synchronized target observations, camera
names and sizes, and a camera-by-capture visibility mask. A visible cell
uses the ordered target point slots. Use JSON null for a missing point and
retain at least six distinct correspondences in each usable view. Select
**Register cameras** initialization for partial views. Camera pairs connect
through shared captures, which may form a chain. Fit the intrinsics
from the same observations or connect fixed camera models. Output poses map
the first camera's coordinates into each named camera's coordinates in the
target's declared meters or millimeters.

**Validate multi-camera rig** keeps the rig fixed and uses independent
captures. For each co-visible capture it fits the target in one camera and
projects the target through the rig into the other cameras. Fit and holdout
reports include a camera-by-capture RMS table. Debug heatmaps show missing
cells in black. No rig is published when the fit exceeds its error limits.
[Three-camera chain example](<../examples/opencv tools/multicamera_rig/README.md>)

**Load coded rig captures** reads synchronized images and a surveyed ArUco
layout from a manifest. The layout can place markers of different sizes and
orientations in the target plane. The loader identifies corners by decoded ID,
reports duplicate and missing views, and preserves missing point slots. The
partial-view fit and held-out checks use each camera's observed corners.
[Custom coded target example](<../examples/opencv tools/coded_multicamera_rig/README.md>)

**Stereo BM / SGBM disparity** can derive a signed pixel search from near
and far depth limits using the calibrated rectification Q. The native
search count rounds up to a multiple of 16. Connecting rectification or
per-view support masks enables checks for complete search footprints.
The block reports the native ROI, the signed-disparity-safe rectangle,
and the final support fraction after mask holes. Pixels outside support
have invalid disparity, zero validity and zero confidence.
[Known-disparity example](<../examples/opencv tools/stereo_disparity_search/README.md>)

**Binary Census stereo disparity** compares local brightness order in
rectified images. It builds Census signatures with NumPy and sums the
Hamming distances with OpenCV box filtering. The pinned wheel exposes
constants for the contrib binary matcher, so this block uses an app-owned
integer-disparity implementation.

Use a signed manual interval of 2–128 candidates or derive it from
calibrated depth. A match must have full Census and aggregation support
across the search in both views. Best Cost, Second Best Cost and
Confidence describe forward matching on supported pixels. Confidence is
`(second - best) / second`, with tied costs assigned zero. The optional
reverse check also tests the winning correspondence. Use Valid Mask
to identify disparities retained after both checks.
[Known-shift comparison with SGBM](<../examples/opencv tools/binary_stereo/README.md>)

**Quasi-dense stereo disparity** grows OpenCV corner seeds into matching
pixel pairs. It reports signed horizontal disparity, vertical offsets,
eligible left support, and a validity mask in the rectified left grid.
Calibrated ROIs and optional per-view masks and regions constrain each
pair's correlation patch. A reverse pass can verify correspondence.
[Forward and reverse baseline examples](<../examples/opencv tools/quasi_dense_stereo/README.md>)

**Match stereo exposure and color** estimates a right-to-left gain or
gain plus bias per gray or RGB channel. Wire a reliable left-grid
floating-point disparity guide in pixel units. Finite disparities map
left X to right X as X minus disparity. For fractional X, both sampling
taps must have right-view support. The robust fit excludes near-saturated
pairs, rejects low signal variation and requires enough photometric
inliers to emit a corrected image. Apply it to the right rectified view
before running the final matcher. A supplied right mask and calibrated
right ROI restrict both reference sampling and correction. The output
preserves the right image's coordinate frame, encoding, depth and alpha.
[Calibrated color correction example](<../examples/opencv tools/stereo_color/README.md>)

**Remove disparity speckles** accepts float pixel disparity, or signed or
unsigned 16-bit fixed-point disparity with a configurable scale. OpenCV
stereo's raw values use 16 samples per disparity pixel. The optional input
mask and invalid sample setting exclude holes before filtering. Set an
invalid sample in the input's original units. Nonfinite input pixels are
always invalid.

The native filter groups 4-connected pixels whose neighbor differences
are within the configured threshold and removes groups at or below the
pixel-count limit. Set the size to zero to normalize samples and apply
validity without filtering. Float inputs are temporarily quantized for
connectivity. Retained float values keep their original precision, and input
integers convert to float pixels. The output uses NaN for invalid disparity
and exposes valid and newly removed masks on the same grid.
[Fixed-point and signed float example](<../examples/opencv tools/disparity_speckles/README.md>)

**Evaluate stereo disparity** compares an estimate and a reference on the
same pixel grid, with optional Region and Mask selection. Float disparities
are already in pixels. Set independent raw-samples-per-pixel scales for
integer maps and optionally exclude a finite invalid code in each map's
original units. Supply Estimate Valid and Reference Valid masks when
available. The default alignment checks known coordinate frames. Use
Pixel Grid to assert alignment for separately imported rasters with the
same dimensions.

Coverage is the fraction of reference-valid pixels with an estimate.
Bias is estimate minus reference. The bad-pixel fraction counts absolute
errors strictly above the threshold among evaluated pixels. The report
also includes the fraction of reference-valid pixels that were either
bad or missing. Coverage and bad-pixel limits gate the result independently.
Confidence is optional and does not change the overall evaluation. It
breaks the evaluated pixels into below-cutoff, at-or-above-cutoff and
unknown strata. Connect separate evaluators to compare regions in one run.

The block outputs the absolute error map and distinct evaluated, bad and
missing masks. The optional heatmap shows unscored pixels in black and
missing estimates in magenta. Feed it decoded monochrome reference
disparity. [Exact whole-field and regional example](<../examples/opencv tools/disparity_reference/README.md>)

**Calibrate wide-angle stereo** fits a same-size fisheye or omnidirectional
pair from matched target views. The fisheye fitter can retain or refine
supplied intrinsics. The omnidirectional fitter initializes both cameras
and reports which view pairs OpenCV retained. **Rectify stereo images**
accepts either resulting rig, produces a perspective pair and emits metric Q
for the existing disparity and cloud blocks. Fisheye exposes balance and FOV
scale. Omnidirectional rectification exposes the shared output focal length.
[Input format and examples](<../examples/opencv tools/wide_angle_stereo/README.md>)

**Reproject depth-backed region** transfers a source selection through a
rigid source-to-target camera pose. Each camera can be pinhole, fisheye
or omnidirectional. It lifts selected pixels using positive axial camera
Z, checks the inverse lens model by round trip, and projects them into
the target. Supply metric depths and named optical frames. The target
pixel keeps the nearest projected Z, with source scanline order breaking
exact ties. Output preserves the sampled footprint as a mask and an
optional exact raster region. It leaves unsampled pixels empty.

An optional target depth map rejects projections behind a measured
surface. Require Agreement also rejects projections with missing
reference depth or with positive clearance in front of it. Otherwise
these projections remain eligible and are marked unverified. The report
distinguishes clipped, colliding, occluded and unverified samples. It
also reports the fraction of selected source pixels retained, which can
decrease when the target view has lower magnification.
[Fisheye-to-omnidirectional example](<../examples/opencv tools/depth_region_projection/README.md>)

1. Define a rig with left/right intrinsic models and a left-to-right rigid pose,
   or fit it using **Calibrate stereo cameras** and corresponding calibration
   point arrays across at least three views.
2. **Rectify stereo images** produces remapped images and an explicit record with
   R1/R2, P1/P2, Q, valid ROIs, units and rectified frame. BM/SGBM expects a
   horizontally rectified pair.
3. **Stereo BM / SGBM disparity** exposes the search range, block size, native
   penalties/filters and optional contrib WLS. Its float disparity is in pixels
   (native fixed-point values are divided by 16); invalid values are NaN. Use
   the valid mask or the colored preview for inspection. WLS also emits a
   normalized confidence image. An optional cutoff excludes low-confidence
   pixels from downstream reconstruction. [Stereo confidence example](<../examples/opencv tools/stereo_confidence/README.md>)
4. **Disparity to 3D cloud** uses Q, accepts a mask/region and optional registered
   color, and emits a cloud in the first camera's rectified frame plus Z depth.
   Depth limits are in rig units. Use `input_scale=0.0625` for raw OpenCV int16
   disparity and `1` for this app's already-scaled disparity.

**Warp RGB-D by camera pose** reprojects a registered depth image and
optional color into a named target optical frame. The source and target use
the same undistorted pinhole intrinsics and image size, with positive focal
lengths and principal point. Supply a pose that maps source 3D coordinates to
target coordinates. A mask and region select source pixels. OpenCV floors
the projected coordinates and retains the nearest positive target Z at each
destination pixel. A hole has NaN depth, a zero validity mask and source
index −1. Valid source indices are zero-based and row-major. The optional
color keeps its encoding and precision using the same winning source pixels.
The emitted camera record and pixel grid identify the destination frame.
[RGB-D pose warp example](<../examples/opencv tools/rgbd_pose_warp/README.md>)

**Backproject depth pixels** takes a list of image XY requests and an
undistorted pinhole camera calibrated for the input depth grid. Nearest
sampling snaps to an integer pixel center, with ties rounded upward. Integer
sampling rejects fractional coordinates. An optional mask and region restrict
the sampled pixels. Each report row identifies the request index, the source
pixel and its row-major index when sampled, and the cloud row and XYZ when
valid. Depth is camera Z. XYZ uses the named optical frame, with X increasing
right and Y increasing down. Set the uint16 scale for your sensor. A zero,
nonfinite or nonpositive sample is invalid, and unsigned value 65535 is the
OpenCV invalid sentinel. The block gathers only the requested positions and
emits a sparse cloud. Tagged floating-point images retain their declared depth
units on input. Choose meters or millimeters for the cloud.
[Depth-step example](<../examples/opencv tools/sparse_depth_backprojection/README.md>)

**Decode and scale metric depth** converts uint16 measurements or existing
floating-point depth to meters or millimeters, with Float32 and Float64 output
and a separate validity mask. Untagged uint16 inputs use the configured scale
in meters per code and invalid code. Code zero is always invalid. Tagged
uint16 values carry the scale and invalid code from the encoder.

**Quantize metric depth to uint16** rounds positive metric values to the
nearest code, with exact ties going to the even code. Reserve 0 or 65535.
Choose whether ranges below or above the positive code interval should
clamp or become invalid. The out-of-range mask identifies both cases. The
report gives separate maximum absolute and RMS errors for in-range rounding
and all retained samples, including any clamped values. Decode the codes
before passing them to geometry blocks. The encoding metadata travels with
graph connections. Record it separately when exporting codes to a PNG.
[Metric depth round-trip example](<../examples/opencv tools/metric_depth_encoding/README.md>)

**Fill small depth holes** copies nearby valid samples into small invalid
regions and emits a mask identifying the estimates. Its default settings
preserve border-connected gaps. [Depth hole filling example](<../examples/custom tools/depth_hole_filling/README.md>)

**Smooth depth while preserving edges** uses a confidence-weighted bilateral
filter guided by the depth itself. The depth sigma is in meters. A matching
confidence image can reduce the weight of uncertain neighbors and invalidate
selected samples below the threshold. Existing holes stay invalid. A mask or
region limits writes, and trusted samples outside it can still support the
filter. The depth keeps its type, units, ticket and grid. The validity and
confidence-rejection masks use the same grid.
[Depth smoothing example](<../examples/opencv tools/depth_smoothing/README.md>)

**Clean time-of-flight depth** uses sensor confidence and a neighborhood on the
source depth grid to reject suspect samples. A depth within the agreement
distance of a trusted neighbor is protected from geometry rejection. Farther
and nearer neighbors can identify a mixed edge sample, and a coherent,
well-supported surface can identify a one-sided isolated outlier. Both
distances are in meters. A mask or region limits which samples can be rejected.
The cleaned depth invalidates rejected samples and keeps all other values,
their units and their spatial and capture metadata. Raw depth and the reason
codes remain available for review. Per-frame geometry is a heuristic for
artifacts such as multipath. Its thresholds require qualification on the
actual sensor and scene.
[Time-of-flight depth cleanup example](<../examples/custom tools/tof_depth_cleanup/README.md>)

**Fuse registered depth views** combines two to four positive axial-Z
rasters on one target camera grid. Register the views first. The default
alignment checks known pixel frames. Pixel Grid declares that separately
loaded rasters already share a grid. Known tickets must agree. Tagged
m/mm depth keeps its unit interpretation, and the result is in meters.

Each optional confidence image uses float values on [0, 1] or full-scale
uint8/uint16 values. Omitted confidence gives weight one. After rejecting
invalid depth and low confidence, the default Nearest policy averages
samples within tolerance behind the closest trusted depth. Consensus
chooses the largest group whose depth range fits the same tolerance.
Distinct groups tied for the largest size are left unresolved. Minimum
Support gates the winning group size. A mask or region limits the
evaluation, and Minimum Coverage gates the fraction that fuses.

The output confidence is the mean weight of contributing samples at
each fused pixel. Support Count records the size of the chosen group,
including groups that fall short of Minimum Support. Conflict Mask
marks incompatible eligible samples. Ambiguous Mask marks tied groups.
Rejected Views assigns bit values 1, 2, 4 and 8 to A, B, C and D where a
winner identifies outlying views. The report separates missing, tied
and insufficient support.
[Registered fusion and thin-foreground example](<../examples/opencv tools/registered_depth_fusion/README.md>)

For multiple stereo baselines, each Q-derived depth describes its own
rectified left camera. **Rectified stereo camera geometry** extracts
the rectified intrinsics from P1 and the pose back to the original left
frame using the inverse of R1. Use **Register RGB-D with confidence**
to map each result onto the same left image grid before fusion.
[Three-baseline consistency example](<../examples/opencv tools/multi_baseline_depth/README.md>)

**Evaluate camera depth** compares camera Z on aligned pixel grids.
Tagged depth uses its m/mm units. For untagged rasters, set the units
and, for raw uint16, the units per sample for each input. Zero, negative
and nonfinite depths are invalid. A region or selection mask limits the
comparison, and per-input validity masks can exclude untrusted samples.
Use the default Same Frame setting to check known pixel frames. Choose
Pixel Grid to assert alignment for independent imports.

The residual is estimate minus reference in millimeters, with NaN
outside valid overlap. Coverage divides overlap by the valid reference
count. Validity IoU divides overlap by the union of both validity masks.
The report includes bias, MAE, MSE, RMSE, and the fraction of overlapping
pixels strictly above the error tolerance. Coverage and error limits
control pass/fail independently. The optional magnitude heatmap shows
missed estimates in magenta and estimate-only samples in gray.
[Mixed-unit depth step example](<../examples/opencv tools/depth_reference/README.md>)

**Separate support plane and foreground** fits one dominant plane using valid
backprojected pixels. A mask or region can limit the fit to exposed support.
Classification still covers the whole image. Signed distances are in meters
perpendicular to the plane and positive on the camera side. The foreground-only
depth preserves the input type, units and grid. The block requires minimum
plane support before it emits classifications or foreground depth.
[Support-plane example](<../examples/opencv tools/depth_plane/README.md>)

**Measure depth components** takes axial Z and undistorted pinhole
intrinsics on the same grid. Apply near and far cutoffs and optionally
a Mask or Region. A foreground mask from **Separate support plane and
foreground** can restrict the input for a tilted work surface. Four- or
eight-neighbor connections can also require a maximum local Z step.
Zero step limit uses ordinary spatial connectivity.

Each retained component has an exact source label and a standard
collection record. Width and height measure the observed pixel
footprint along camera X and Y. Depth Span covers the observed range
of Z. Projected Area sums fronto-parallel pixel footprints at their
measured depths. Use **Filter object collection** with a custom metric,
such as `width_m`, and **Mask from filtered labels** with the corresponding
Labels output to rebuild the selected source mask.
[Touching-parts example](<../examples/opencv tools/depth_components/README.md>)

**Measure calibrated depth silhouette** uses a mask of one visible top face,
pinhole depth, and a metric support plane in the depth camera frame. It
back-projects usable boundary pixel footprints and measures their minimum-area
rectangle on the datum. Length is the longer side and width the shorter
side. Thickness is the median perpendicular height above the datum. The
block reports the exact invalid boundary selection and distinguishes missing
depth, rejected height and unusable corner rays. Face coverage, boundary
coverage and the image border policy determine whether dimensions are emitted.
[Tilted raised-parts example](<../examples/opencv tools/depth_silhouette/README.md>)

**Depth to point cloud** calls OpenCV's RGB-D projection using undistorted
pinhole intrinsics. `DepthArray` supplies its declared m/mm units. Untagged
integer depths use the configured units-per-sample; untagged floats use the
configured input units. Zero, negative and nonfinite values are omitted. XYZ
uses camera X right, Y down and Z forward, verified against synthetic pixels.
Registered color is optional; matching dimensions alone do not register it.

**Register depth to camera** uses explicit source/target intrinsics and a metric
source-to-target pose. It emits target-grid depth in meters. **Organized RGB-D
normals** uses native FALS, SRI or cross-product normals and retains valid pixel
correspondence. Sampling/point limits bound cloud outputs on large images.

**Recorded RGB-D** reads a lossless manifest with two ordered acquisition
streams using a declared shared clock. It emits a color image, metric depth,
optional depth-grid confidence, and one ticket per pair. Each frame is used
once. The report includes both nanosecond timestamps, signed skew, and
unmatched counts. The saved graph supports Run All and loop playback.

**Register RGB-D with confidence** projects depth samples onto the supplied
color camera grid using two pinhole models and the metric depth-to-color pose.
Depth must use undistorted intrinsics. A target lens distortion profile can
be used when projecting into raw color. The nearest positive camera Z wins
per pixel. Its source confidence travels with it. Low confidence leaves a
hole, even when a farther sample projects into the same pixel. The color
image is passed through and the output mask marks finite registered depth.
[Recorded RGB-D example](<../examples/opencv tools/recorded_rgbd/README.md>)

**Project cloud and check visibility** rasterizes metric cloud samples to the
target pinhole grid using calibrated intrinsics and an optional rigid transform.
It exports nearest camera Z, coverage and RGB when the cloud has colors.
The optional target depth compares predicted surface samples with observed Z.
The block keeps absent or farther-than-expected observations in the unverified
class. Its visible, occluded and unverified clouds preserve source attributes.
[Cloud visibility example](<../examples/opencv tools/cloud_visibility/README.md>)

## Point clouds and surface matching

`PointCloud` is a typed `cloud` port value with finite Nx3 XYZ, units, a named
frame, optional RGB bytes, unit normals, and optional source pixel indices/size.
It is usable inside custom blocks and ToolBlocks. It is separate from an RGB
image; the graph will not connect cloud output directly to an image input.
Ticket/recipe/outcome envelopes still work, and queue sizing counts array bytes.
Headless reports summarize clouds rather than including all XYZ samples.

Retained point clouds rotate in visible node thumbnails and in the inspector's
**Results** tab. For a block that also emits an image, select its cloud output
under **Block → Show advanced → Node preview** to put the cloud on the canvas.
Double-click a **Cloud** preview or click **Pop out…** to open its live 3D viewer.
Enable **Flyaround** to rotate continuously, and use **Yaw**, **Pitch**, and
**Point size** to adjust the view. The viewer is also available under
**Results → 3D point-cloud viewer…**. The previews use sampled points and are
available in Debug and Balanced.

- **Filter point cloud** selects XYZ bounds and can also use an image mask or
  region when pixel correspondence remains available. Regions retain source
  image frame checks.
- **Voxel-sample point cloud** uses a zero-anchored grid with uniform or per-axis
  spacing in the cloud's units. Each occupied voxel can retain its first
  source row, compute a centroid, or retain the measured row nearest its
  centroid. Equal distances select the earliest input row. Measured modes can
  expose original row indices and preserve the corresponding colors, normals
  and pixels. Centroid mode averages positions and colors. [Voxel sampling
  example](<../examples/custom tools/voxel_representatives/README.md>)
- **Measure point-cloud normals and curvature** uses bounded metric
  neighborhoods and local covariance eigenvectors. It orients valid normals
  toward or away from a viewpoint or uses a deterministic axis sign. It
  reports sparse and ambiguous neighborhoods, supports an eigenvalue-gap
  quality threshold, and passes curvature in the same row order as the
  accepted cloud. Source diagnostics retain all row positions.
  [Surface quality example](<../examples/custom tools/cloud_surface_quality/README.md>)
- **Fit point-cloud plane or sphere** uses seeded sample consensus and
  refines the best model using its support. The block classifies the entire
  scan at the selected metric tolerance and returns coefficients, signed
  residuals, and source-row IDs for inliers and outliers. Cloud outputs
  preserve source attributes and order. Sphere radius limits can constrain
  the search. Degenerate or insufficiently supported fits report a failure
  without emitting a model. [Plane and sphere example](<../examples/custom tools/cloud_model_fitting/README.md>)
- **Grow point-cloud surface regions** links metric XYZ neighbors that meet
  the unsigned normal-angle limit. It uses normals stored in the cloud, or
  estimates normals in bounded local PCA fits. Curvature can come from the
  same fits or a per-row input. Invalid or high-curvature points remain
  unassigned and cannot bridge regions. Labels, source row IDs, and both
  cloud subsets preserve source provenance. The pinned OpenCV 5 Python wheel
  uses the SciPy and NumPy fallback. [Touching-faces example](<../examples/custom tools/cloud_region_growing/README.md>)
- **Cluster point cloud by distance** groups scan rows linked by an
  inclusive XYZ radius in meters. It filters groups by point count and
  longest XYZ axis extent. Each accepted cluster includes bounds and original
  row indices. Optional labels align with the input rows and use 0 for
  rejected rows. Both cloud outputs preserve scan order and point attributes.
  Point, neighbor-pair and accepted-cluster limits bound the work and results.
  [Tray clustering example](<../examples/custom tools/cloud_clustering/README.md>)
- **Reject point-cloud outliers** splits a metric scan into retained and
  removed rows. Radius mode counts other rows within an inclusive distance
  in meters and retains rows at or above the minimum count. Statistical
  mode averages each row's K nearest other distances and keeps means at
  or below the global mean plus the selected multiple of population standard
  deviation. K adapts to the number of available rows. Coincident rows
  count separately. Both outputs preserve source order, units, frame,
  colors, normals and source pixels. The index size and per-query memory
  are bounded with an input point budget and batches.
  [Folded-panel example](<../examples/custom tools/cloud_outlier_rejection/README.md>)
- **Sample point cloud** chooses original rows by seeded random sampling
  without replacement or greedy farthest-point sampling. Set a maximum count
  or a capped fraction in the inspector. Farthest mode can stop when the next
  candidate's separation drops below a threshold in the cloud's units. It
  reports minimum retained spacing and covering radius. A work limit bounds
  the distance evaluations and produces a failed result with the partial
  sample if the requested stopping condition cannot be reached. The output
  indices match the cloud's sampling order and keep source colors, normals
  and pixels.
  [Cloud sampling example](<../examples/custom tools/cloud_sampling/README.md>)
- **Query point-cloud octree** finds exact nearest neighbors or the nearest
  hits inside an inclusive radius. Choose a fit or explicit cube in meters,
  then set its depth or maximum cell side. Cloud XYZ and query XYZ can be
  supplied in meters or millimeters. Per-query results are sorted by
  Euclidean distance and then original cloud row index. Radius results
  indicate when more hits exist than the result limit. The output cloud
  combines the selected rows in source order, preserving their colors,
  normals and source pixels. The index stores 36 bytes per retained point
  in its arrays. Build and search also use temporary memory. Query and point
  budgets bound their inputs, and the index is rebuilt on each invocation.
  [Octree query example](<../examples/custom tools/cloud_octree_query/README.md>)
- **Measure point cloud** reports bounds, extent, centroid and PCA axes.
- **Register clouds by local features** estimates the source-to-target pose
  for partly overlapping metric clouds. It compares neighbor distances and
  covariance shape in a configured radius, uses reciprocal descriptor
  matches, and fits a proper rigid transform with RANSAC. The block gates its
  pose by correspondence count, fit error and measured overlap in both
  directions. Its inlier records retain original point indices and available
  image pixels. Clouds must have matching m/mm units and similar local
  sampling. The pose can initialize ICP.
  [Partial-scan example](<../examples/opencv tools/cloud_feature_registration/README.md>)
- **Estimate point-cloud normals** uses contrib plane fitting. **Point-to-plane
  ICP** refines alignment of oriented clouds, optionally initialized with a
  known pose. It emits the transformed source cloud and native residual. Use
  an initial pose for separated clouds; ICP is local registration.
- **Train 3D PPF surface model** prepares a portable oriented-point model.
  **Match 3D PPF surface model** trains/caches the native detector for that model
  and emits ranked voted poses in scene coordinates. Downsample dense models
  before training. Use ICP to refine a selected candidate as needed.

PLY supports ASCII and binary little endian vertex data. PCD supports ASCII
XYZ, optional normals and packed RGB. OBJ supports ASCII XYZ, optional
normalized inline RGB on every vertex, and optional normals aligned by row
order. OBJ rejects malformed rows and non-point geometry. PLY and PCD drop
invalid XYZ rows. All three formats round-trip units and frame through
Dataflow comments. External files without those comments use inspector
settings. File and point limits apply. Save to a new path, which may be
project-relative. Image pixel mappings and arbitrary payload metadata remain
in memory. [OBJ and PLY example](<../examples/custom tools/cloud_obj_interchange/README.md>)

## Motion, fusion and structured light

**RGB-D frame odometry** wraps OpenCV 5 `ptcloud` two-frame motion with depth,
RGB or joint RGB-D modes. Supply matched intrinsics for both frames and, for
intensity modes, the corresponding registered images. Outputs map source-frame
points into the target frame in meters. The block does not track a global map
or accumulate trajectory state by itself.

**Fuse posed depth into a mesh** integrates camera Z into a bounded metric
distance field and uses scikit-image marching cubes on supported cells. It
accepts one depth image or a group of posed images. All eight corners of a
surface cell must satisfy the configured observation count. `TriangleMesh`
carries vertices, triangle indices, units and the world frame on a typed
`mesh` port. TSDF mesh extraction requires the optional `skimage` extra.
[TSDF mesh example](<../examples/skimage tools/tsdf_mesh/README.md>)

**Load triangle mesh** reads triangulated OBJ and PLY into the same port.
The mesh can retain vertex normals, RGB and two- or three-component texture
coordinates. The OBJ reader resolves signed indices and splits vertices at
normal or texture seams. PLY imports support ASCII and either binary byte
order. **Save triangle mesh** writes ASCII OBJ or binary little endian PLY
to a new path. Both formats preserve units and frame through Dataflow comments.
[Triangle interchange example](<../examples/custom tools/mesh_interchange/README.md>)

**Fuse posed depth frames (TSDF)** uses native OpenCV 5 `Volume`/`VolumeSettings`.
It builds a bounded dense volume per activation from a depth image or FlowGroup,
with one calibrated camera-to-world pose per frame. It emits a world-frame
oriented cloud and a raycast depth view from the final camera. Voxel size, volume
origin/resolution, truncation, weight and raycast step are exposed. This is
posed fusion, with no automatic tracking or persistent unbounded volume.
The CPU path is selected because local OpenCL volume extraction failed on this
wheel/driver; other operations' thread OpenCL settings are restored afterward.

**Track depth trajectory with TSDF** processes a batch of ordered, calibrated
metric depth planes. It estimates motion from the last accepted exposure
and checks the proposed pose on the fused TSDF surface. Rejected exposures
leave the pose and volume untouched. A bounded loss limit and final tracking
check control publication. Accepted depth packets and poses share a new
metric world frame for each completed segment.
[Recorded trajectory example](<../examples/opencv tools/depth_trajectory/README.md>)

**Generate Gray-code patterns** emits an ordered group and white/black frames.
**Decode Gray-code correspondences** accepts matching captured images, optional
shadow captures and selected camera pixels. It reports camera/projector pixel
pairs and validity, without guessing metric 3D. Use calibrated camera/projector
projection matrices and valid pairs for triangulation. Pattern projection,
trigger timing and physical capture are external to this synthetic example.

**Generate phase-shift patterns** emits a single-cycle coarse triplet and
a fine triplet in 120° steps for one projector axis. **Decode absolute
phase-shift coordinates** uses the coarse estimate to select the fine fringe
cycle per pixel. It emits subpixel coordinates and phase on the camera grid
with modulation, ambiguity and validity diagnostics. Absolute X and Y use
separate capture sequences and a stationary, aligned scene.
[Capture examples](<../examples/opencv tools/phase_shift/README.md>)

**Unwrap measured fringe phase** accepts a float single-channel wrapped
phase image in radians and an optional lit-pixel mask. OpenCV produces
the unwrapped phase and local inverse reliability. Four-connected
regions retain independent cycle counts. Wire known reference phases
at selected pixels to fix each region's integer offset. Relative mode
uses the first eligible pixel's wrapped cycle when a reference is absent.
Required mode flags missing references. Conflicting or inaccurate
references withhold the corresponding region. The output carries the
phase camera grid and records anchored and relative labels in its payload.
[Shadow and independent reference example](<../examples/opencv tools/phase_unwrap/README.md>)

**Track metric object silhouette** starts with a named metric triangle mesh,
an undistorted pinhole camera and an approximate object-to-camera pose.
The tracker searches normal to the predicted boundary, refines pose against
observed edges and checks the support and reprojection RMS. It emits a pose
in meters with the frame ticket. If tracking is lost, send a Boolean reset
and a current seed to resume. Use the compact report and selectable mesh
and matched-edge drawings to inspect the fit.
[Recorded silhouette example](<../examples/opencv tools/silhouette_tracking/README.md>)

**Train RGB-D pose template** teaches a named orientation from an isolated
object, aligned color and metric depth, and a known pose. Chain training
blocks to add views. **Find RGB-D template pose** searches the full calibrated
grid or a selected region, selects a trained rotation and recovers metric
translation from aligned depth. It checks appearance, anchor support and
reprojection before outputting the ticketed pose.
[Two-view LINEMOD example](<../examples/opencv tools/linemod_pose/README.md>)

**Calibrate camera and projector** fits the camera-to-projector transform
from matched target views on two pinhole grids. It can refine projector
intrinsics on their own grid while keeping camera intrinsics fixed.
**Reconstruct structured-light depth** takes dense X/Y phase maps or sparse
Gray-code pairs, and emits metric camera-Z, a point cloud, validity and
reprojection error. Triangulation checks positive depth at both devices,
ray angle, and error in each device's own pixel coordinates. Use the
single-file calibration blocks to save and reuse the resulting rig.
[Calibration and stepped-depth example](<../examples/opencv tools/camera_projector/README.md>)

## Planar fiducial targets

**Generate planar fiducial target** creates a single ArUco marker, an ArUco
grid or a ChArUco board from a predefined dictionary. Configure the marker
IDs, border width, physical lengths, DPI and white margin. The raster uses
integer pixel geometry. At the reported page size it preserves the configured
ArUco marker edge or the ChArUco square pitch. The report gives the effective
dimensions of all rasterized features in the selected physical units.

Export the raw image and set the full print dimensions using the reported
width and height. Use the measured dimensions of the printed board for
calibration. The runnable example verifies each layout using image detection,
including custom IDs and legacy ChArUco parity.
[Target generator example](<../examples/opencv tools/fiducial_targets/README.md>)

## Planar pose hypotheses

**Planar pose hypotheses** uses a calibrated pinhole camera and matched points
on a plane. Use general IPPE with at least four coplanar correspondences or
IPPE square with a centered, ordered marker square. It scores both native
solutions in the original distorted pixel grid, filters nonpositive depth,
and removes numerical duplicates. Each emitted candidate reports a rigid
object-to-camera transform in the supplied frames and camera units.

The block emits **Hypotheses** and flags **Ambiguous** when the top scores fall
within the configured RMS gap. **Pose** emits only when the ambiguity gate
allows a single transform. Connecting the source image adds drawable normals
to the preview. [Near-frontal and oblique example](<../examples/opencv tools/planar_pose_hypotheses/README.md>)

## Sparse multi-view reconstruction

Pack each calibrated pinhole view with its world-to-camera pose and landmark
IDs paired to distorted image pixels. Use three to 16 views sharing a world
frame and length unit, with up to 512 unique landmarks. A track contains the
visible observations for one ID. Undistorted bearings initialize each track
with multi-view direct linear triangulation. Tracks with weak parallax,
nonpositive depth, or excessive pixel error are reported and excluded.

Refinement minimizes distortion-aware reprojection residuals with a robust
loss. Points-only mode preserves every supplied pose. Joint mode holds two
anchor poses fixed and adjusts the other extrinsics. Use a known baseline and
shared, non-collinear tracks with enough angular separation. The cloud retains
the anchors' world frame and units, including unitless coordinates when no
metric baseline was supplied. Landmark IDs are stored alongside cloud rows in
metadata. This workflow takes supplied correspondences and initial camera
poses. [Four-view example and independent holdout](<../examples/opencv tools/sparse_multiview/README.md>)

**Recover metric scene scale** fits a positive scale, proper rotation and
translation from a unitless reconstruction to measured landmark coordinates.
Use at least three non-collinear anchors and one consistent camera gauge.
The solver pairs landmarks by ID, uses consensus to reject bad anchors and
reserves optional disjoint holdouts for the final acceptance check. It emits
the point cloud and optional world-to-camera poses in meters on the surveyed
frame, preserving the input intrinsics and reprojections.
[Scale recovery with independent metric targets](<../examples/opencv tools/scene_scale/README.md>)

- [OpenCV multi-view DLT reference](https://docs.opencv.org/5.0/extra_modules/triangulation.html)
- [SciPy sparse least-squares reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

## Robot-world calibration

**Calibrate robot-world and hand–eye** fits a transform from the moving object
to the gripper and a transform from the stationary object to the robot base.
For a fixed camera those objects are the target and camera respectively. For
a robot-mounted camera they are the camera and the target. Both arrangements
consume gripper-to-base and target-to-camera observations. Use multiple views
with rotations about at least two independent axes and one common length unit.
Ticketed pose groups pair by ticket and holdout tickets must be disjoint.

The solver initializes from Park relative motion, then jointly refines the two
rigid transforms against pose closure using SciPy least squares. It reports
translation RMS in millimeters and angular RMS in degrees, independently for
training and held-out samples. Both transforms are withheld if the solver fails
to converge or if either supplied split exceeds its configured limits.
[Fixed and robot-mounted camera examples](<../examples/opencv tools/robot_world/README.md>)

## Build and verification boundaries

**OpenCV 3D capabilities** reports bindings in the installed build. The Windows
OpenCV 5 contrib wheel exposes geometry, calibration, stereo, ptcloud and the
contrib modules used here, with several RGB-D functions directly under `cv2`.
This wheel lacks Python SFM/viz and the native hand–eye and robot-world solvers.
KinFu variants expose Python bindings with disabled constructors in this build.
The native TSDF block is qualified separately from KinFu. Robot-world results
come from offline synthetic fixtures. Physical hand–eye calibration,
camera/projector commissioning and arbitrary third-party rigs remain unverified.

- [OpenCV 5 module organization](https://github.com/opencv/opencv/wiki/OpenCV-5)
- [OpenCV 5 ptcloud API](https://docs.opencv.org/5.0/main_modules/ptcloud.html)
- [OpenCV RGB-D and volumetric reconstruction](https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html)
- [OpenCV contrib surface matching](https://docs.opencv.org/4.x/d9/d25/group__surface__matching.html)
- [OpenCV structured light](https://docs.opencv.org/4.x/d1/d90/group__structured__light.html)
