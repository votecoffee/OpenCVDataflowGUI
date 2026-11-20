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
uses the full ordered point set for the shared target. Each camera must be
connected through shared captures, which may form a chain. Fit the intrinsics
from the same observations or connect fixed camera models. Output poses map
the first camera's coordinates into each named camera's coordinates in the
target's declared meters or millimeters.

**Validate multi-camera rig** keeps the rig fixed and uses independent
captures. For each co-visible capture it fits the target in one camera and
projects the target through the rig into the other cameras. Fit and holdout
reports include a camera-by-capture RMS table. Debug heatmaps show missing
cells in black. No rig is published when the fit exceeds its error limits.
[Three-camera chain example](<../examples/opencv tools/multicamera_rig/README.md>)

**Stereo BM / SGBM disparity** can derive a signed pixel search from near
and far depth limits using the calibrated rectification Q. The native
search count rounds up to a multiple of 16. Connecting rectification or
per-view support masks enables checks for complete search footprints.
The block reports the native ROI, the signed-disparity-safe rectangle,
and the final support fraction after mask holes. Pixels outside support
have invalid disparity, zero validity and zero confidence.
[Known-disparity example](<../examples/opencv tools/stereo_disparity_search/README.md>)

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
  image frame checks. **Voxel-sample** can retain the first point per voxel or
  compute centroids. Centroids drop measured pixel indices and normals.
- **Measure point cloud** reports bounds, extent, centroid and PCA axes.
- **Estimate point-cloud normals** uses contrib plane fitting. **Point-to-plane
  ICP** refines alignment of oriented clouds, optionally initialized with a
  known pose. It emits the transformed source cloud and native residual. Use
  an initial pose for separated clouds; ICP is local registration.
- **Train 3D PPF surface model** prepares a portable oriented-point model.
  **Match 3D PPF surface model** trains/caches the native detector for that model
  and emits ranked voted poses in scene coordinates. Downsample dense models
  before training. Use ICP to refine a selected candidate as needed.

PLY read/write supports ASCII and binary little endian vertex data. PCD supports
ASCII XYZ, optional normals and packed RGB. Files round-trip units and frame
through dataflow header comments; files without those comments use inspector
settings. Invalid XYZ rows are dropped. Point/payload limits apply, existing
files are preserved, and paths may be project-relative. The interchange does
not export image pixel mappings, meshes or arbitrary payload metadata.

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
`mesh` port. **Save triangle mesh as PLY** writes a new bounded binary PLY.
The mesh path requires the optional `skimage` extra.
[Mesh example](<../examples/skimage tools/tsdf_mesh/README.md>)

**Fuse posed depth frames (TSDF)** uses native OpenCV 5 `Volume`/`VolumeSettings`.
It builds a bounded dense volume per activation from a depth image or FlowGroup,
with one calibrated camera-to-world pose per frame. It emits a world-frame
oriented cloud and a raycast depth view from the final camera. Voxel size, volume
origin/resolution, truncation, weight and raycast step are exposed. This is
posed fusion, with no automatic tracking or persistent unbounded volume.
The CPU path is selected because local OpenCL volume extraction failed on this
wheel/driver; other operations' thread OpenCL settings are restored afterward.

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
poses. The separate scene-scale item (S3-65) tracks scale recovery when camera
motion starts at unknown scale. [Four-view example and independent holdout](<../examples/opencv tools/sparse_multiview/README.md>)

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
