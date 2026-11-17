# Rotating-camera panoramas

`cv.stitch_rotating_panorama` registers an unordered batch of 2–8 images using
OpenCV's native panorama camera estimator. It checks fitted rotations against
separately computed SIFT matches and robust pairwise homographies, then renders
the original pixels on a spherical, cylindrical or plane canvas.
The [eight projects](<../examples/opencv tools/rotating_panorama/README.md>) show
unordered overlap, input reordering, rejected and partial components, masks,
high-depth pixels and a contradictory view.

## Sources and registration

Connect a `flow.batch` group to Views. Sources must have matching pixel depth,
channel layout and encoding: gray, RGB or BGR. Optional mask and region groups
use the original source labels/order and compatible source grids. Their
intersection restricts both feature registration and final pixel support.

Unique batch labels identify cameras by default. Repeated labels, such as serial
captures from one camera, become `view_0`, `view_1`, etc. Source labels can instead
provide a unique list in input order. Anchor label overrides Reference index;
the anchor must survive registration. Input order does not imply overlap order.

Registration uses a shared selected-pixel luminance range to prepare bounded
8-bit work images. Registration megapixels controls their resolution. The
returned camera matrices are converted back using each actual rounded work-grid
scale. Original source pixels remain the rendering inputs, including uint16
and floating-point data. The work-image normalization is reported explicitly.

Native confidence controls OpenCV's component selection. The independent check
exposes feature count, mutual descriptor ratio/distance, homography inliers,
inlier fraction, RANSAC settings and bidirectional camera-model error. Error
thresholds are in original source pixels. A locally supported overlap that
contradicts the fitted rotation model rejects the fit. Too few matches do not
establish an overlap. The retained graph must connect to the anchor through
validated edges. Pair counts, inliers, residuals, native selection, retained
indices, dropped labels and all native camera estimates are available in Report.

This separate check can reuse image observations used by native fitting; it is
not a reserved validation set. Native high-level camera fitting and optional
wave correction are exposed now. Explicit solver/refinement choices, adjustable
intrinsic priors and reserved-observation validation remain in **S1-147**.

## Missing views and output validity

Component policy is explicit:

| Policy | Missing or disconnected views |
| --- | --- |
| Reject | Failed inspection; numerical image/layout outputs are withheld. |
| Fail partial | Render the anchored component with a failed judgment. |
| Allow partial | Render the anchored component; Report still says `complete: false`. |

At least two connected views are required. An absent anchor, insufficient
texture, native estimation failure, contradictory camera model or invalid plane
horizon produces a normal failed inspection with a reason and Report.
Bad configuration, invalid metadata, resource violations or native exceptions
use the common contained block-error path. Neither path emits stale images.

Image retains source depth/channel encoding and is zero outside Coverage.
Contributors counts supported samples at each output pixel. Interpolation uses
nearest/linear support or conservative cubic/Lanczos source-mask halos.
Float64 cubic remapping is rejected because of an observed native boundary
problem; linear or float32 is supported. Average blend combines supported raw
sample values; First prefers the earliest retained input. There is no exposure
compensation or radiometric calibration in these modes.

Coverage and Minimum covered fraction describe the observed rectangular canvas.
Padding is not a measured sample. Use Coverage for downstream measurements.
The optional connectivity Diagnostic is a raster chart on its own coordinate
grid. Source-boundary result graphics are drawable objects on Image, so they can
be combined with mapped measurements before the final filmstrip image. Graphics
disabled omits the chart and drawings while retaining numerical processing.

## Layout and source-point conversion

Layout is a JSON-compatible `rotation_panorama_v1` document containing:

- Projection, scale, origin, canvas size, anchor and a geometry-bound identity.
- Original input indices, resolved/original labels and source dimensions.
- Full-resolution pinhole matrices `K` and work-grid scales.
- `camera_to_anchor` before wave correction and `camera_to_panorama` afterward.
- Source ticket/spatial context, selected-pixel counts and native warp extents.

The panorama has an independent pixel coordinate root. A single source
homography cannot describe a spherical/cylindrical canvas. Wave correction
rotates the common panorama orientation; it does not establish a physical world
frame. Layout identity binds the complete declared geometry. The point mapper
checks its schema, finite matrices, dimensions and identity before use.

`cv.panorama_points` maps up to 100,000 source `[x,y]` measurements through Layout.
Select a camera with the optional Source label input, configured Source label,
original Source index, or the anchor. Compatible crop frames are converted to
full source pixels. Known incompatible roots or known different source tickets
are rejected; Ticket policy Ignore is an explicit override. Unknown point frames
must already use full source pixels. Missing ticket metadata cannot prove an
acquisition match.

Optional Source mask is on the full source grid; Mask is on the panorama grid,
normally Coverage. Outside policy filters unsupported points or rejects the
input. Points contains retained coordinates, Indices their original positions,
Valid one boolean per submitted point, and Count the retained total. Minimum
points sets the inspection judgment. Optional accepted/rejected markers belong
to the panorama grid and can be aggregated with stitcher graphics. Outputs use
the current layout/batch ticket and retain source-point ticket provenance;
obsolete source pyramid context is removed.

Layout is an inspectable current-run geometry record. Reusing a trained layout
to render future captures is not implemented by these blocks.

## Bounds and applicability

Default limits are 16,777,216 total source pixels, 0.2 megapixels per registration
image, 32,000,000 independent descriptor-pair comparisons, 8,388,608 canvas and
summed local-map pixels, and a 512 MiB array-work estimate. Shared source guards
also bound individual images. Pixel/work checks precede mask preparation and
native map allocation; rendering visits one source map at a time. Parameter
limits are documented in the [generated reference](block-parameters.md).
The estimate excludes graph copies and native estimator workspaces and is not
a process-memory or runtime guarantee.

Use static pinhole views rotating about one optical center with adequate
overlap and texture. Correct lens distortion upstream. Parallax, moving parts,
repetitive texture and lighting changes can defeat registration or its checks.
The controlled examples validate known focal length, relative rotation,
world-ray projection, support, point mapping and deliberate rejection. They do
not establish physical lens calibration, metric translation, robot extrinsics
or general-scene accuracy.
