# OpenCV coverage backlog

This inventory compares the built-in graph registry with the public Python
bindings in the pinned OpenCV 5 contrib wheel. Priorities reflect reuse across
industrial vision jobs and the workflows in the [roadmap](roadmap.md). They
are qualitative estimates. The [block parameter reference](block-parameters.md)
lists the tools that can be placed on a graph today. The
[roadmap](roadmap.md#vision-tool-priorities) ranks work by value to inspection
jobs and by how naturally inputs and results fit a graph.

## Stage 1 — Existing array tools and deferred work

The graph already covers common array arithmetic, range and mask operations,
channel routing, geometric transforms, reductions and extrema, matrix
solvers and decompositions, and reversible Fourier and cosine transforms.
[Measure image norms](../examples/inspection/image_norm/README.md) adds
absolute and reference-relative reductions across selected pixels, including
L1, L2, squared L2, infinity, Hamming and Hamming2.
[Combine Fourier spectra](../examples/image_processing/spectrum_product/README.md)
exposes complex bin multiplication and conjugation for custom filters,
circular convolution and circular correlation.
[Seeded random images and noise](../examples/image_processing/seeded_noise/README.md)
provide Gaussian and uniform fixtures and selected perturbations using a
private generator per activation.
[Matrix structure tools](../examples/numerical/matrix_structure/README.md)
construct diagonal matrices, extract diagonals and traces, and mirror
the upper or lower half of a square matrix.

The existing matrix blocks remain available for saved jobs and numerical
workflows. Further numeric table utilities and specialized array layouts
are deferred until an inspection use case needs them. User Python blocks
can handle specialized calculations with job-specific interfaces. The
binding inventory helps identify candidates and verify availability.

## P1 — Common tools and workflow gaps

| Rank | Outstanding capability | OpenCV primitives and graph work |
| --- | --- | --- |
| 1 | Inspection image stitching | [Stitch overlapping views](../examples/image_processing/stitch_image_pair/README.md) builds a two-view mosaic from a shared frame or wired registration, with coverage and overlap masks. [Stitch an ordered scan](../examples/image_processing/stitch_scan_sequence/README.md) automatically registers up to eight adjacent views and reports fit and overlap quality. [Correct exposure in an inspection scan](../examples/image_processing/stitch_scan_exposure/README.md) demonstrates overlap-based gain or offset correction and distance-weighted feathering. [Close the loop on an inspection scan](../examples/image_processing/stitch_scan_loop/README.md) constrains accumulated drift when the final view overlaps the first. [Choose a content-aware seam](../examples/image_processing/stitch_seam/README.md) uses bounded graph-cut selection on registered pairs or ordered scans, with automatic seam scaling for larger overlaps. Tiled rendering for mosaics beyond the current canvas and memory limits remains open. |
| 2 | Camera motion stabilization | [Camera stabilization](../examples/flow_control/camera_stabilization/README.md) estimates translation or similarity motion from background tracks, smooths or locks the trajectory, and emits a coverage mask and frame transforms. Longer recorded sequences, parallax evaluation, and offline look-ahead smoothing remain open. |
| 3 | Separate touching parts and isolate foreground | [Watershed separation](../examples/image_processing/watershed/README.md) includes distance peaks, editable point seeds, mask seeds, and a synthetic touching-parts example. Production calibration and validation fixtures remain open. |
| 4 | Reference comparison and defect scoring | The basic workflow is available, including histogram backprojection and a reusable registered defect inspection. [Flat-field correction](../examples/image_processing/flat_field/README.md) normalizes illumination using bright and optional dark references. Validated defect-specific inspection recipes remain open. |
| 5 | Multi-result template search and pick selection | Explicit visible-feature evidence and depth-based occlusion checks remain open. Multi-result matching and image-plane pick selection support score, pose angle, clearance and deterministic downstream ranking. [Subpixel refinement](../examples/pattern_matching/subpixel_template/README.md) is available for template centers and poses. |
| 6 | Calipers, widths, contours and shape fitting | Physical-unit tolerance recipes remain open. [Path intensity profiles](../examples/inspection/path_intensity_profile/README.md) sample open and closed paths with mask and region selection. [Contour profile tolerance](../examples/inspection/contour_profile/README.md) measures both directions between aligned outlines in pixels. Pose-relative paired-edge calipers now measure width, center offset and gaps across straight parallel scans. [Hull and convexity defect inspection](../examples/inspection/contour_defects/README.md), contour similarity, [contour approximation](../examples/geometry/contour_approximation/README.md), point-to-contour distance and dedicated shape fits are available. |
| 7 | File video and replay | Timed playback, live network streams, independent source readiness and deeper codec validation. Local video replay and writing are available with frame ranges, stride, frame tickets, timestamp provenance and stop/loop modes. See [video file workflows](video-files.md). |
| 8 | Industrial text and code workflows | Trainable OCR workflow assets, qualification of direct part marks on production images, and a separate barcode grading workflow. OCR and decoded-text checks are available alongside OpenCV QR/linear readers and optional ZXing Data Matrix and multi-format decoding. See [OCR workflows](ocr.md). |
| 9 | Color tolerance and coating checks | Production color-chart qualification remains open. [Multi-reference Lab classification](../examples/inspection/lab_color_classification/README.md) is available. Color-chart detection, fitted correction matrices, selected Lab statistics and CIE76/CIEDE2000 tolerance checks are available. See [color measurement](color-measurement.md). |
| 10 | Motion and conveyor tracking | PLC/camera hardware synchronization, timestamp-window pairing, calibrated world coordinates and production validation. Multi-part association and frame-latched encoder prediction support tracking beyond the camera view in image coordinates. CSRT/KCF track seeded boxes, and MOG2/KNN, sparse LK and Farneback provide motion primitives. |
| 11 | Robot and calibration workflows | Fixed-camera and robot-world/hand–eye calibration, frame composition tied to acquisition timestamps, held-out robot target checks and calibration revision selection. PnP and basic ArUco pose are available. [Validate lens calibration](../examples/calibration/validate_lens_calibration/README.md) checks fixed pinhole profiles on separate grid observations. [ChArUco board](../examples/calibration/charuco_board/README.md) emits identified board coordinates from partial views and feeds pose estimation. [ChArUco lens calibration](../examples/calibration/charuco_lens_calibration/README.md) fits a pinhole model from batches with different visible corners. [ChArUco calibration validation](../examples/calibration/validate_charuco_calibration/README.md) reports held-out errors by corner ID. [Hand–eye calibration](../examples/calibration/hand_eye_calibration/README.md) fits an eye-in-hand transform from paired poses and checks target consistency. |

## P2 — Common extensions

| Family | Outstanding capability |
| --- | --- |
| Image restoration | Blind deblurring and hard-shadow removal remain open. [Adaptive background correction](../examples/image_processing/adaptive_background/README.md) estimates smooth shading from one image. [Wiener deblur](../examples/image_processing/wiener_deblur/README.md) uses Gaussian, [linear motion](../examples/image_processing/motion_deblur/README.md), circular defocus, or custom point-spread functions. [TV-L1 denoising](../examples/image_processing/tvl1_denoising/README.md) supports single images and aligned observation batches. [Temporal non-local means](../examples/image_processing/temporal_nlm/README.md), single-frame non-local means, Telea/Navier–Stokes inpainting, [guided filtering](../examples/image_processing/guided_filter/README.md), and [domain transform filtering](../examples/image_processing/domain_transform_filter/README.md) are available. |
| Registration extensions | Sparse alignment diagnostics and perspective panorama workflows. Single-scale and multiscale ECC support intensity alignment with masks. |
| Feature detection and correspondence | Custom BRISK patterns remain open. [Descriptor radius matching](../examples/pattern_matching/descriptor_radius/README.md) retains multiple candidates per keypoint within a distance limit. [MSER](../examples/image_processing/mser_regions/README.md) detects stable grayscale and color regions. [Standard BRISK](../examples/pattern_matching/brisk_registration/README.md) supports Hamming matching. ORB/FAST and [standalone SIFT](../examples/pattern_matching/sift_registration/README.md) are available. [AKAZE](../examples/pattern_matching/akaze_registration/README.md) supports binary and float descriptors. ORB, SIFT, and AKAZE matching support ratio and cross-check modes, with homography inlier scoring downstream. [FLANN KD-tree matching](../examples/pattern_matching/flann_registration/README.md) accepts SIFT, [standalone KAZE](../examples/pattern_matching/kaze_registration/README.md), and float AKAZE descriptors. |
| Shape and topology | [Ballard generalized Hough](../examples/pattern_matching/generalized_hough/README.md) locates fixed-scale, fixed-orientation template shapes. [Guil generalized Hough](../examples/pattern_matching/generalized_hough_guil/README.md) searches rotation and scale and emits oriented poses. Ellipse/arc and additional polygon models, shape context and Zernike-style descriptors where available. [Contour hierarchy](../examples/geometry/contour_hierarchy/README.md) exposes parent and child links, nesting depth, and holes. [Rotated rectangle overlap](../examples/geometry/rotated_rectangle_overlap/README.md) fits part footprints and measures their intersection. [Convex polygon overlap](../examples/geometry/convex_overlap/README.md) measures continuous intersection, contact, and IoU from vertices or contour hulls. [Enclosing convex polygon](../examples/geometry/enclosing_polygon/README.md) fits bounded-side footprints and reports hull fill. [Delaunay triangulation](../examples/geometry/delaunay_triangulation/README.md) builds a planar mesh from point sites or measured centers. [Voronoi cells](../examples/geometry/voronoi_cells/README.md) partitions the image around sites and emits a selected-cell mask. [Minimum spanning tree](../examples/geometry/minimum_spanning_tree/README.md) finds minimum-cost connections in weighted graphs and point meshes. Connected components and Find blobs expose OpenCV's default, SAUF, BBDT, and Spaghetti labeling algorithms. |
| Segmentation and texture | SLIC/SEEDS/LSC superpixels, graph segmentation, selective search, ridge/Frangi-style response, local binary patterns, and co-occurrence matrices. [Pyramid mean-shift filtering](../examples/image_processing/mean_shift_filter/README.md) flattens local color variation before segmentation. [Gabor orientation bank](../examples/image_processing/gabor_orientation_bank/README.md) compares multiple angles and reports orientation confidence. [Gabor texture response](../examples/image_processing/gabor_texture/README.md) supports signed and quadrature responses with independently configurable orientations. [Pixel K-means](../examples/ml/kmeans/README.md) is available. |
| Spectral and intensity processing | [Linear channel transform](../examples/image_processing/channel_transform/README.md) mixes channels with an editable or wired matrix and optional per-channel offsets. [Color normalization](../examples/image_processing/normalize_color/README.md) supports shared or independent channel scales and optional mask or region selection. [DCT/IDCT](../examples/image_processing/dct_round_trip/README.md) supports reversible monochrome transforms. [DFT/IDFT and radial filtering](../examples/image_processing/fourier_filter/README.md) support band selection. [Hann window](../examples/image_processing/hann_window/README.md) exposes 2D or row tapering before a transform. [Per-channel intensity LUTs](../examples/image_processing/intensity_lut/README.md) support uint8 and uint16. [Intensity transforms](../examples/image_processing/intensity_transform/README.md) include log, exp, square root, power, and clipping. [Image arithmetic](../examples/image_processing/image_arithmetic/README.md) combines two images or an image and a scalar. [Pixel comparisons](../examples/image_processing/pixel_comparison/README.md) produce masks from relational tests. [Find nonzero pixels](../examples/image_processing/nonzero_pixels/README.md) extracts bounded coordinate lists and full-mask bounds. [Intensity profiles](../examples/inspection/intensity_profile/README.md) reduce rows or columns by channel and can return extrema coordinates. [Image summary statistics](../examples/inspection/image_statistics/README.md) measure masked channel statistics, arbitrary percentiles, and extrema locations. [Custom Fourier gain masks](../examples/image_processing/fourier_gain/README.md) support per-bin gains and conjugate pairing. [Fourier correlation](../examples/pattern_matching/fourier_correlation/README.md) measures linear or circular image displacement. [Tilted window statistics](../examples/image_processing/tilted_statistics/README.md) measure batches of 45-degree rectangles. [Local window statistics](../examples/image_processing/local_statistics/README.md) use summed-area tables for rectangular mean, standard deviation, and sample-count maps. |
| Image metrics | [Absolute image error](../examples/inspection/absolute_image_error/README.md) provides MAE, RMSE, maximum absolute error, and error maps. [Earth Mover’s Distance](../examples/inspection/earth_movers_distance/README.md) measures histogram shifts using bin-center distances. BRISQUE and other no-reference scores, perceptual image hashes, and defect score calibration remain open. |
| Motion and background | DIS, Dual TV-L1, DeepFlow and RLOF where bound, optical-flow consistency and visualization, Kalman filtering and tracking across missing detections. MOG2 and KNN provide adaptive foreground masking. |
| Calibration models and target tooling | Fisheye/omnidir, multi-camera extrinsics and sparse bundle adjustment where supported. [Circle grid calibration](../examples/calibration/circle_grid_calibration/README.md) supports symmetric and staggered targets for planar mapping and lens fitting. [Lens correction maps](../examples/calibration/lens_correction_maps/README.md) exposes reusable pinhole undistortion and rectification maps in the graph. [Lens map displacement](../examples/calibration/lens_map_displacement/README.md) measures and visualizes correction magnitude, direction, and source coverage. [Lens map files](../examples/calibration/lens_map_files/README.md) saves and reloads complete map pairs with output intrinsics. [Lens fit report](../examples/calibration/lens_fit_report/README.md) adds per-view reprojection RMS and native uncertainty estimates. |
| Depth and stereo | Edge-aware depth filtering, time-of-flight depth cleanup, alternative stereo algorithms and dense calibration evaluation. Stereo BM/SGBM supports [WLS confidence and left–right consistency](../examples/3d/stereo_confidence/README.md). [Depth hole filling](../examples/3d/depth_hole_filling/README.md) is available. |
| Point cloud and reconstruction | Surface/mesh output, organized segmentation and plane fitting, feature registration, multiple hypotheses and fitness ranking, occlusion/z-buffer visibility, SfM and more complete multi-frame pose/mesh workflows. Existing ICP, PPF, depth/point-cloud conversion and TSDF cover the baseline. |
| DNN task adapters | Multi-input model binding, dynamic shapes and batching, detection/rotated detection, NMS, semantic/instance masks, keypoints and OCR postprocessing, confidence calibration, portable preprocessing and CPU-vs-accelerator validation. Existing ONNX inference, classification and embeddings provide the base. |
| Classical ML | Persist and reuse fitted K-means centers for prediction, and add train/predict/export with SVM, KNearest, trees/forests, boosting, EM and other `ml` estimators. Add feature-table preparation and train/validation split. [Numeric K-means fitting](../examples/ml/kmeans/README.md) is available. |
| Media and image formats | [Bayer demosaicing](../examples/inputs/bayer_demosaicing/README.md) supports four tile patterns, three interpolation modes, and crop parity. [In-memory PNG and JPEG](../examples/outputs/in_memory_image_codecs/README.md) provide typed packets and explicit compression settings. [TIFF page replay](../examples/inputs/tiff_page_sequence/README.md) emits ticketed pages from a multipage file. [Multipage TIFF output](../examples/outputs/multipage_tiff_output/README.md) writes ordered lossless image batches. [Raw YUV decoding](../examples/inputs/raw_yuv/README.md) handles seven packed 8-bit 4:2:0 and 4:2:2 layouts. [Image metadata and EXIF orientation](../examples/inputs/exif_orientation/README.md) report decoded chunk types and selected EXIF fields and can orient JPEG/PNG inputs. Metadata writing, other sensor formats, and additional codecs and numerical formats remain open. |

[Feature covariance and Mahalanobis distance](../examples/numerical/feature_covariance/README.md) cover joint variation and numeric outlier checks.
[Solve linear system](../examples/numerical/linear_system/README.md) provides exact and least-squares
solutions with multiple observation columns, numerical rank, and residuals.
[Solve linear program](../examples/numerical/linear_program/README.md) optimizes continuous
nonnegative variables with linear constraints and emits solver status and constraint slack.
[Solve polynomial roots](../examples/numerical/polynomial_roots/README.md) supports analytic real
roots through degree three, iterative complex roots, and relative residual judgments.
[Multiply matrices](../examples/numerical/matrix_multiply/README.md) applies weighted real matrix
products with an optional addend and independent transposition controls.
[Invert matrix](../examples/numerical/matrix_inverse/README.md) supports LU, Cholesky and SVD
with numerical rank, reciprocal conditioning, and determinants for square matrices.
[Singular value decomposition](../examples/numerical/matrix_svd/README.md) exposes the spectrum,
thin or full factors, low-rank reconstruction and relative truncation error.
[Eigendecomposition](../examples/numerical/matrix_eigen/README.md) exposes symmetric and
nonsymmetric real eigenpairs, numerical basis rank, and residual checks.
[Check image samples](../examples/image_processing/image_validity/README.md) reports nonfinite and
out-of-range pixels with source-aligned masks, a first location, and a fraction limit.
[Replace nonfinite samples](../examples/image_processing/nonfinite_replacement/README.md) repairs
NaN and signed infinity by channel, preserves the image encoding and depth units,
and emits a mask of changed pixels.
[Join images](../examples/image_processing/join_images/README.md) builds horizontal or vertical
comparison views with padding and per-panel coordinate context.
[Paste image patch](../examples/image_processing/paste_patch/README.md) performs a masked
copy at a canvas offset and reports both rectangles and the copied pixels.
[Subpixel patch extraction](../examples/pattern_matching/subpixel_crop/README.md) samples a
rectangular patch at a fractional center and preserves coordinate provenance.
[Hough lines from points](../examples/geometry/point_set_lines/README.md) finds lines
from sparse coordinates and exposes vote counts for collection sorting.
[Hough polar lines](../examples/geometry/hough_polar_lines/README.md) supports binary
and intensity-weighted raster voting with masks and regions.
[Clip line segments](../examples/geometry/clip_line_segments/README.md) trims measured
lines to image or rectangular region bounds and measures the visible span.
[Hough circle collections](../examples/geometry/hough_circle_collections/README.md) expose
native strengths for Gradient and Gradient ALT and support radius filtering.
[Polar image warp](../examples/image_processing/polar_image_warp/README.md) unwraps circular
features onto linear or semilog grids, tracks source coverage, and can rewrap
an intact grid into its original frame.
[Tile image](../examples/image_processing/tile_image/README.md) repeats full images in a grid
with compact layout data and source-frame provenance.
[Transpose image](../examples/image_processing/transpose_image/README.md) swaps image axes
with exact main- or anti-diagonal reflection and coordinate mapping.
[Route image channels](../examples/image_processing/route_channels/README.md) copies, reorders,
and duplicates channels across matching input images, with optional zero fill.
[Labeled distance transform](../examples/image_processing/labeled_distance/README.md) computes
pixel distances, nearest-seed labels, and a mask for one selected zone.
[Masks from signed labels](../examples/image_processing/label_masks/README.md) shows exact
pixel comparisons and inclusive ranges on int32 segmentation outputs.
[Blend with weight maps](../examples/image_processing/blend_weight_maps/README.md) blends aligned
images using pixel weights and optional mask and region selection.
[Gram matrix](../examples/numerical/gram_matrix/README.md) computes row or column dot products
with optional centering, reusable offsets, and sample or population normalization.
[Find nearest rows](../examples/numerical/nearest_rows/README.md) adds exact top-K feature-table
search, candidate masks, mutual nearest matching, and numeric or packed-byte distances.
[Sort numeric table](../examples/numerical/numeric_table_sort/README.md) ranks whole records or
individual columns and rows, preserving input order for ties and returning source indices.
[Cartesian and polar fields](../examples/image_processing/polar_fields/README.md) convert in both directions and preserve angle units.
[Iterative phase correlation](../examples/pattern_matching/iterative_phase_correlation/README.md)
estimates subpixel translation and emits a correction for the moving image.
[Matched-feature translation](../examples/pattern_matching/translation_registration/README.md)
fits a translation with RANSAC or LMeDS and reports per-pair residuals.
[Packed remap coordinates](../examples/calibration/packed_remap/README.md) supports nearest
and subpixel fixed-point maps and retains lens correction provenance.
[Checkerboard capture quality](../examples/calibration/chessboard_quality/README.md)
measures edge width and brightness in both board directions.
[Paired Sobel and derivative-fed Canny](../examples/image_processing/canny_derivatives/README.md) expose signed gradients and accept custom derivative fields.
[Scale absolute to 8-bit](../examples/image_processing/scale_absolute/README.md) applies gain
and offset in stored units, saturates large magnitudes, and supports mask and
region selection.
[Principal components](../examples/ml/principal_components/README.md) supports fitted dimensionality reduction and reconstruction.

## P3 — Specialized families

| Family | Outstanding capability |
| --- | --- |
| Photo and exposure | Exposure alignment/merge, HDR/tone mapping, seamless clone, decolorization, stylization, motion deblur, `xphoto` white-balance variants and learning-based super-resolution. |
| Recognition | WeChat QR, QR encoding, multiple-code layout, cascade/HOG, face detection/recognition, saliency, line descriptors, image retrieval/vocabulary and OpenCV-specific OCR variants. |
| Sequence | Additional tracker families, background subtractors in `bgsegm`, motion history and advanced video stabilization. |
| Geometric utilities | Expanded robust estimators, graph matching, rectification variants, affine-invariant feature variants, panorama detail steps and coordinate-space diagnostics. |
| Diagnostics and visualization | Display overlays as reusable blocks, match/inlier overlays, density maps, 3D axes and camera frusta, disparity confidence views and inspection measurement plots. [Colorize scalar image](../examples/image_processing/colorize_scalar/README.md) provides built-in or custom palettes, fixed and automatic ranges, and invalid-pixel masking. |
| Contrib and domain tools | Fuzzy and bioinspired processing, structured-light modes beyond Gray code, saliency, detailed image hashing and optional SfM, mesh/visualization and hardware-specific modules where the Python build supplies them. |

## P4 — Infrastructure and build-dependent features

- CUDA/OpenCL/UMat acceleration, hardware decoder/encoder backends, camera
  device controls, DNN targets, and alternate runtimes require explicit build
  probes and end-to-end measurements.
- OpenCV.js/WebAssembly and compiled ToolBlocks belong to the
  [execution roadmap](opencv-backend.md).
- Drawing helpers, matrix constructors, getters/setters, serializers, backend
  controls and overloaded aliases generally belong inside higher-level nodes.
  Wheel bindings do not establish device support or calibration accuracy.

## Exact callable inventory

The [binding inventory](opencv-bindings.tsv) includes functions, classes and
public methods declared on those classes. Filter by priority and
`no static reference` to find candidates. `literal reference` means a
`cv2.…` expression occurs in application source. `method name only` means
the method name occurs somewhere in application source and may have a
different owner. `alias_of` identifies bindings that point to the same
Python object and the corresponding methods. The scan cannot decide whether a
reference is internal, configurable on a graph, dynamically resolved, or
equivalent to a tool already built from other primitives. The prioritized
tables above make that distinction for the main workflows.

Run `python scripts/opencv_binding_inventory.py` from the repository using its
pinned environment to refresh the TSV. Add `--check` to verify it. Constants,
private symbols and Python helper modules are omitted. OpenCV features absent
from this wheel must be evaluated against the intended distribution separately.
See the [OpenCV 5 documentation](https://docs.opencv.org/5.0/).

## Available code readers

**Read QR codes** supports multi, single and curved QR modes. **Read linear
barcodes** exposes native detector scales, gradient/downsampling thresholds
and optional super-resolution model files. Both take uint8 images with
optional masks and regions. Keep the full code and its quiet zone inside the
selection. Both return decoded text, symbology, original image-space corners,
center, count and an optional overlay. A minimum read count can fail the block
and report a reason. The automated fixtures decode QR and EAN-13. Supported
linear symbologies and decode behavior depend on the OpenCV implementation.
See the [OpenCV barcode documentation](https://docs.opencv.org/5.0/main_modules/objdetect_barcode.html).


## Available segmentation and comparison tools

**Seeded flood fill** returns a mask, area and bounds. It supports fixed
or floating intensity tolerances, 4/8-connectivity and a mask or region
that acts as a barrier. **Marker-controlled watershed** accepts a uint8
seed mask or a nonnegative int32 label image. It returns labels, a boundary
mask and a mask for one selected label or all foreground labels.
**Watershed seeds from distance peaks** converts a binary foreground mask
into int32 markers. It uses label 1 for known background, label 0 for
unassigned pixels and labels 2 and up for foreground peaks. Connect
`markers` to **Marker-controlled watershed**, select `labels` as the
marker mode and set the background label to 1. Connect the original
grayscale or color image to the watershed image input.
**Watershed seeds from points** and **Watershed seeds from masks** construct the
same label map from manually or programmatically selected cores. The watershed
block includes a color preview and a per-label foreground area table.
See the [OpenCV distance-transform watershed tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/distance_transformation/distance_transform.html).
**Morphological reconstruction** propagates a monochrome marker under
an upper image bound by dilation or above a lower bound by erosion.
It reports the number of geodesic steps and withholds the image when
the iteration limit is reached before convergence.
**GrabCut foreground** accepts a rectangle, a binary probable-foreground mask
or native four-state markers. It initializes a new model for each activation.
**Thin binary mask** exposes Zhang–Suen and Guo–Hall skeletonization.
The optional scikit-image blocks expose Zhang, Lee, Guo–Hall, and medial-axis
skeletonization, plus length-based pruning of terminal branches. See the
[skeleton pruning example](../examples/image_processing/skeleton_pruning/README.md).
**Fill holes in a binary mask** fills enclosed background components, with an
optional maximum hole area and configurable background connectivity.

**Compare image quality** compares two registered images using SSIM, PSNR or
MSE and an optional pass threshold. A region or mask selects pixels for the
score. SSIM uses neighborhoods from the full images, so nearby unselected
pixels can affect a selected pixel's similarity. SSIM and MSE return full
maps. The score uses equal weighting across selected pixels and channels.


## Available repeated template matching

**Find template instances** searches for fixed-scale, fixed-orientation
copies with a normalized matching method, minimum score, result limit,
overlap suppression and optional mask/region inputs. It returns ranked
bounds, centers, scores and a list of center poses. The pose list connects
directly to **Inspect each pose in parallel**. Open the [25-part example](../examples/pattern_matching/template_instances/template_instances.json)
or run it with `opencv-dataflow-run examples/pattern_matching/template_instances/template_instances.json`.


## Available histogram and shape measurements

**Compare histograms** accepts matching Histogram outputs and offers all six
OpenCV comparison metrics, optional count normalization, per-channel results
and a threshold on the mean or worst channel. Use identical bin edges and
ranges in both source histograms. **Histogram backprojection** maps one
modeled channel to a float32 image of per-pixel weights. It supports an
optional region or mask, uniform histogram bins, and peak, probability or
raw-count weighting. Convert the reference and sample channels to the same
numeric scale before comparing or backprojecting.
See the [OpenCV backprojection tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/histograms/back_projection/back_projection.html).

**Fit line, ellipse or enclosing circle** fits points or one detected object.
It reports image-space geometry and root mean squared distance to the fitted
boundary, with an optional residual tolerance. Circle mode returns the
smallest enclosing circle.
[Region from fitted shape](../examples/geometry/fitted_shape_region/README.md) turns
ellipse or enclosing-circle fits into reusable image selections with an axis margin.

**Compare contours (Hu moments)** measures similarity between ordered point
lists or selected contour/blob collection items. **Contour hull and convexity
defects** reports hull vertices, solidity and concavity depths from a
whole-pixel contour. **Test points against contour** classifies points as
inside, on the edge or outside, and can filter them by signed distance to the
boundary. See the [OpenCV shape-analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html).
In a two-object fixture with one hole, the pinned OpenCV 5.0.0
`findContoursLinkRuns` binding reported the hole contour without a parent link.
The standard contour path remains the supported option for hierarchy.
[Approximate convex contour](../examples/geometry/convex_contour_approximation/README.md)
exposes fixed-side reduction with an added-area budget and verifies enclosure.


## Available alignment and restoration

**Align images (ECC)** registers a sample to a reference with single-scale or
multiscale enhanced correlation. It exposes translation, rigid, affine and
homography models, optional masks for both frames, an optional initial
transform and a minimum coefficient. It emits the aligned image and the
sample-to-reference transform. The inverse mapping is available on its
geometry output.

Import the [Registered defect inspection ToolBlock](../examples/user_tools/registered_defect_inspection/registered_defect_inspection.toolblock.json)
to align a sample to a known-good reference, threshold the absolute
difference and report connected defects. Its optional region is in reference
coordinates. The included settings use translation, a residual threshold of
35 intensity levels and a minimum defect area of 12 pixels. Tune these
settings in the nested graph for your images.

**Non-local means denoise** offers channel-based and luminance/chroma modes.
A region or mask limits where denoised pixels are written. **Inpaint selected
pixels** reconstructs a masked area from neighboring pixels with Telea or
Navier–Stokes.


## Available conveyor pick selection

**Select pick candidate** accepts the match records from shape, texture or
fixed-scale template matching. It projects a configured part rectangle and
clearance margin around each pose. Candidates can be rejected for a low
score, an out-of-range angle, a center outside the pick region, a clearance
rectangle outside the image, or interference with another detected part.
An optional uint8 mask adds obstacle evidence outside the candidate's own
footprint. Other supplied detections contribute obstacle footprints even
when their scores or angles make them ineligible for selection.

The block returns one pose, eligible count, per-candidate rejection reasons
and an optional overlay. Ranking uses the configured image-space downstream
vector or the match score, with stable input order as the final tie breaker.
The [conveyor pick example](../examples/inspection/conveyor_pick/conveyor_pick.json) supplies a
synthetic obstacle mask and runs with
`opencv-dataflow-run examples/inspection/conveyor_pick/conveyor_pick.json`. The check measures visible
image-plane clearance. Depth, gripper reachability and swept-volume collision
checks remain separate work.


## Available bead calipers

**Paired-edge bead calipers** samples parallel intensity scans and finds
opposing edges around a bright or dark bead. Set the scan endpoints across
the expected bead and the span along it. An optional pose makes the scans
follow the part. Width and center offset are measured in image pixels
without a pose and in local pose pixels with a pose. Edge coordinates are
always reported in the working image.

The block checks minimum edge gradient, width, center offset, passing-scan
fraction and the longest run of missing scans. A mask or region must cover
the full scan, including any interpolation neighbors that contribute.
Missing scans, including scans outside the image or selection, count
toward the gap. The [caliper example](../examples/inspection/bead_calipers/bead_calipers.json) measures
a 126-pixel synthetic strip and runs with
`opencv-dataflow-run examples/inspection/bead_calipers/bead_calipers.json`.
