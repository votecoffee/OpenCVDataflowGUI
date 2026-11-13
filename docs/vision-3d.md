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
| [stereo cloud](<../examples/opencv tools/stereo_cloud/stereo_cloud.json>) | Rectified 8 px disparity from a known baseline reconstructs a 3 m plane. |
| [RGB-D cloud](<../examples/demos/rgbd_cloud/rgbd_cloud.json>) | Metric depth, RGB color, pixel correspondence, XYZ filtering, voxel sampling and normals. |
| [3D pose](<../examples/opencv tools/pose_3d/pose_3d.json>) | Project known XYZ, recover a camera-relative pose, and triangulate the points. |
| [structured light](<../examples/opencv tools/structured_light/structured_light.json>) | Generate Gray-code patterns and decode an identity camera/projector mapping. |
| [TSDF fusion](<../examples/opencv tools/tsdf_fusion/tsdf_fusion.json>) | Integrate a posed synthetic depth plane and inspect the reconstructed surface. |

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

## Stereo and depth

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

## Build and verification boundaries

**OpenCV 3D capabilities** reports bindings in the installed build. The Windows
OpenCV 5 contrib wheel exposes geometry, calibration, stereo, ptcloud and the
contrib modules used here, with several RGB-D functions directly under `cv2`.
It has no Python SFM/viz or hand-eye solver. KinFu variants have bindings but
their constructors are disabled in this build. The native TSDF block is tested
separately and does not require KinFu. Live robot integration, hand-eye calibration,
camera/projector commissioning and arbitrary third-party rigs remain unverified.

- [OpenCV 5 module organization](https://github.com/opencv/opencv/wiki/OpenCV-5)
- [OpenCV 5 ptcloud API](https://docs.opencv.org/5.0/main_modules/ptcloud.html)
- [OpenCV RGB-D and volumetric reconstruction](https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html)
- [OpenCV contrib surface matching](https://docs.opencv.org/4.x/d9/d25/group__surface__matching.html)
- [OpenCV structured light](https://docs.opencv.org/4.x/d1/d90/group__structured__light.html)
