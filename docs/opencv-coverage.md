# OpenCV coverage backlog

This inventory compares the built-in graph registry with the public Python
bindings in the pinned OpenCV 5 contrib wheel. Priorities reflect reuse across
industrial vision jobs and the workflows in the [roadmap](roadmap.md). They
are qualitative estimates. The [block parameter reference](block-parameters.md)
lists the tools that can be placed on a graph today. The
[roadmap](roadmap.md#vision-tool-priorities) ranks work by value to inspection
jobs and by how naturally inputs and results fit a graph.

The [image, video and point-cloud coverage checklist](vision-coverage-checklist.md) selects
the stage 1–4 commitments from this backlog. It provides the denominator and
completed counts for roadmap percentages. The [OpenCV scope map](opencv-scope-map.md)
records module and subgroup dispositions, including point-cloud API availability.

## Stage 1 — Existing array tools and deferred work

The graph already covers common array arithmetic, range and mask operations,
channel routing, geometric transforms, reductions and extrema, matrix
solvers and decompositions, and reversible Fourier and cosine transforms.
[Measure image norms](<../examples/opencv tools/image_norm/README.md>) adds
absolute and reference-relative reductions across selected pixels, including
L1, L2, squared L2, infinity, Hamming and Hamming2.
[Combine Fourier spectra](<../examples/opencv tools/spectrum_product/README.md>)
exposes complex bin multiplication and conjugation for custom filters,
circular convolution and circular correlation.
[Fourier component reconstruction](<../examples/opencv tools/fourier_components/README.md>)
(S1-135) adds full, phase-only and zero-phase magnitude reconstruction for
grayscale/color images and typed spectra. It exposes phase validity, DC and
normalization controls, with independent impulse references and explicit
source-window and coordinate behavior.
[Seeded random images and noise](<../examples/custom tools/seeded_noise/README.md>)
provide Gaussian and uniform fixtures and selected perturbations using a
private generator per activation.
[Matrix structure tools](<../examples/opencv tools/matrix_structure/README.md>)
construct diagonal matrices, extract diagonals and traces, and mirror
the upper or lower half of a square matrix.

The existing matrix blocks remain available for saved jobs and numerical
workflows. Further numeric table utilities and specialized array layouts
are deferred until an inspection use case needs them. User Python blocks
can handle specialized calculations with job-specific interfaces. The
binding inventory helps identify candidates and verify availability.

## P1 — Common tools and workflow gaps

[Focus stacking and level selection](<../examples/opencv tools/focus_stack/README.md>)
(S1-138) now fuse registered image batches with local energy, hard or weighted
selection, captured-neighborhood coverage, ambiguity fallback and original
focus-level indices. Original color examples verify independent sharp pixels,
including upstream ECC registration on a shared sharp region. These diagnostics
do not establish physical depth or part visibility.

[Capture-stack flat-field references](<../examples/custom tools/flat_field_reference/README.md>)
(S1-137) now estimate bright/dark maps using mean or median, per-pixel capture
support, optional saturation/noise rejection and calibration selections. Floating
raw-unit references feed the existing correction block without rounding away
capture precision. Original RGB projects verify exact correction on held-out pixels.

[Binary mask boundary and area inspection](<../examples/opencv tools/mask_boundary/README.md>)
(S1-136) now measures directed and combined boundary distances, holes and disconnected
parts, tolerance bands and missing/extra area. Query selection retains full boundary
targets, while explicit border/empty-set policies prevent invented geometry. Original
tab/notch projects demonstrate independent pass/fail limits in the current pixel grid.

| Rank | Outstanding capability | OpenCV primitives and graph work |
| --- | --- | --- |
| 1 | Inspection image stitching | [Stitch overlapping views](<../examples/custom tools/stitch_image_pair/README.md>) builds a two-view mosaic from a shared frame or wired registration, with coverage and overlap masks. [Stitch an ordered scan](<../examples/custom tools/stitch_scan_sequence/README.md>) automatically registers up to eight adjacent views and reports fit and overlap quality. [Correct exposure in an inspection scan](<../examples/custom tools/stitch_scan_exposure/README.md>) demonstrates overlap-based gain or offset correction and distance-weighted feathering. [Close the loop on an inspection scan](<../examples/custom tools/stitch_scan_loop/README.md>) constrains accumulated drift when the final view overlaps the first. [Choose a content-aware seam](<../examples/custom tools/stitch_seam/README.md>) uses bounded graph-cut selection on registered pairs or ordered scans, with automatic seam scaling for larger overlaps. [Tiled inspection mosaic](<../examples/custom tools/tiled_scan_mosaic/README.md>) registers up to 64 views without a dense canvas and renders selected windows with coverage, overlap and source transforms. Priority blending and feather halos preserve window boundaries; the source batch remains bounded. |
| 2 | Camera motion stabilization | [Camera stabilization](<../examples/custom tools/camera_stabilization/README.md>) estimates translation or similarity motion from background tracks, smooths or locks the trajectory, and emits a coverage mask and frame transforms. [Recorded camera stabilization](<../examples/custom tools/recorded_camera_stabilization/README.md>) measures a real sequence on matched support and verifies recovery after coverage rejection. [Check stabilization on a second plane](<../examples/custom tools/stabilization_validation/README.md>) detects accumulated drift in a held-out inspection region on synthetic parallax. [Check stabilization on recorded depth variation](<../examples/custom tools/recorded_stabilization_validation/README.md>) verifies the gate on a recorded translation excerpt with depth-separated patches. [Smooth a recorded camera path](<../examples/custom tools/offline_camera_stabilization/README.md>) adds centered look-ahead for finite batches with per-frame coverage and transforms. [Continuous centered stabilization](<../examples/custom tools/continuous_camera_stabilization/README.md>) processes longer clips with bounded look-ahead and state, per-frame tickets/timestamps, scene segmentation and automatic tail flush. The source example has 144 lossless frames. |
| 3 | Separate touching parts and isolate foreground | [Watershed separation](<../examples/opencv tools/watershed/README.md>) includes distance peaks, editable point seeds, mask seeds, and a synthetic touching-parts example. Production calibration and validation fixtures remain open. |
| 4 | Reference comparison and defect scoring | The basic workflow is available, including histogram backprojection and a reusable registered defect inspection. [Flat-field correction](<../examples/custom tools/flat_field/README.md>) normalizes illumination using bright and optional dark references. Validated defect-specific inspection recipes remain open. |
| 5 | Multi-result template search and pick selection | Explicit visible-feature evidence and depth-based occlusion checks remain open. Multi-result matching and image-plane pick selection support score, pose angle, clearance and deterministic downstream ranking. [Subpixel refinement](<../examples/custom tools/subpixel_template/README.md>) is available for template centers and poses. |
| 6 | Calipers, widths, contours and shape fitting | [Line-pair metrology](<../examples/custom tools/line_pair_metrology/README.md>) measures corners, acute/obtuse angles and parallel gaps with source-frame alignment, physical workplane units, coverage and tolerance judgments. Planar workplane rectification supports physical-unit strip recipes. [Path intensity profiles](<../examples/custom tools/path_intensity_profile/README.md>) sample open and closed paths with mask and region selection. [Contour profile tolerance](../examples/applications/contour_profile/README.md) measures both directions between aligned outlines in pixels. Pose-relative paired-edge calipers now measure width, center offset and gaps across straight parallel scans. [Hull and convexity defect inspection](../examples/applications/contour_defects/README.md), contour similarity, [contour approximation](<../examples/opencv tools/contour_approximation/README.md>), point-to-contour distance and dedicated shape fits are available. |
| 7 | File video and replay | Timed playback, live network streams, independent source readiness and deeper codec validation. Local video replay and writing are available with frame ranges, stride, frame tickets, timestamp provenance and stop/loop modes. See [video file workflows](video-files.md). |
| 8 | Industrial text and code workflows | Trainable OCR workflow assets, qualification of direct part marks on production images, and a separate barcode grading workflow. OCR and decoded-text checks are available alongside OpenCV QR/linear readers and optional ZXing Data Matrix and multi-format decoding. See [OCR workflows](ocr.md). |
| 9 | Color tolerance and coating checks | Production color-chart qualification remains open. [Multi-reference Lab classification](../examples/applications/lab_color_classification/README.md) is available. Color-chart detection, fitted correction matrices, selected Lab statistics and CIE76/CIEDE2000 tolerance checks are available. See [color measurement](color-measurement.md). |
| 10 | Motion and conveyor tracking | PLC/camera hardware synchronization, timestamp-window pairing, calibrated world coordinates and production validation. Multi-part association and frame-latched encoder prediction support tracking beyond the camera view in image coordinates. CSRT/KCF track seeded boxes, and MOG2/KNN, sparse LK, Farneback and DIS provide motion primitives. |
| 11 | Robot and calibration workflows | Fixed-camera and robot-world/hand–eye calibration, frame composition tied to acquisition timestamps, held-out robot target checks and calibration revision selection. PnP and basic ArUco pose are available. [Validate lens calibration](<../examples/custom tools/validate_lens_calibration/README.md>) checks fixed pinhole profiles on separate grid observations. [ChArUco board](<../examples/opencv tools/charuco_board/README.md>) emits identified board coordinates from partial views and feeds pose estimation. [ChArUco lens calibration](<../examples/opencv tools/charuco_lens_calibration/README.md>) fits a pinhole model from batches with different visible corners. [ChArUco calibration validation](<../examples/custom tools/validate_charuco_calibration/README.md>) reports held-out errors by corner ID. [Hand–eye calibration](<../examples/opencv tools/hand_eye_calibration/README.md>) fits an eye-in-hand transform from paired poses and checks target consistency. |

## P2 — Common extensions

| Family | Outstanding capability |
| --- | --- |
| Image restoration | Blind deblurring (S1-84), hard-shadow removal (S1-145), and chromatic aberration correction (S1-146) remain open. [Adaptive background correction](<../examples/custom tools/adaptive_background/README.md>) estimates smooth shading from one image. [Wiener deblur](<../examples/custom tools/wiener_deblur/README.md>) uses Gaussian, [linear motion](<../examples/custom tools/motion_deblur/README.md>), circular defocus, or custom point-spread functions. [TV-L1 denoising](<../examples/opencv tools/tvl1_denoising/README.md>) supports single images and aligned observation batches. [Temporal non-local means](<../examples/opencv tools/temporal_nlm/README.md>), single-frame non-local means, Telea/Navier–Stokes inpainting, [guided filtering](<../examples/opencv tools/guided_filter/README.md>), and [domain transform filtering](<../examples/opencv tools/domain_transform_filter/README.md>) are available. |
| Registration extensions | Sparse alignment diagnostics and perspective panorama workflows. Single-scale and multiscale ECC support intensity alignment with masks. |
| Feature detection and correspondence | Qualify custom BRISK sampling layouts as extensions of S1-43, with persisted pattern metadata and a matching fixture. Native HOG emits one-window descriptors and cell direction graphics. [Descriptor radius matching](<../examples/opencv tools/descriptor_radius/README.md>) retains multiple candidates per keypoint within a distance limit. [MSER](<../examples/opencv tools/mser_regions/README.md>) detects stable grayscale and color regions. [Standard BRISK](<../examples/opencv tools/brisk_registration/README.md>) supports Hamming matching. [Affine-invariant SIFT/ORB](<../examples/opencv tools/affine_features/README.md>) add bounded tilt-roll sampling, source-space support ellipses and known-view repeatability checks. ORB/FAST and [standalone SIFT](<../examples/opencv tools/sift_registration/README.md>) are available. [AKAZE](<../examples/opencv tools/akaze_registration/README.md>) supports binary and float descriptors. ORB, SIFT, and AKAZE matching support ratio and cross-check modes, with homography inlier scoring downstream. [FLANN KD-tree matching](<../examples/opencv tools/flann_registration/README.md>) accepts SIFT, [standalone KAZE](<../examples/opencv tools/kaze_registration/README.md>), and float AKAZE descriptors. |
| Shape and topology | [Ballard generalized Hough](<../examples/opencv tools/generalized_hough/README.md>) locates fixed-scale, fixed-orientation template shapes. [Guil generalized Hough](<../examples/opencv tools/generalized_hough_guil/README.md>) searches rotation and scale and emits oriented poses. Ellipse/arc and additional polygon models, shape context and Zernike-style descriptors where available. [Intensity-weighted moments](<../examples/opencv tools/weighted_moments/README.md>) (S1-131) compare selected contrast-weighted and geometric centers/covariance, preserve numerical scale and frame context, and produce reusable poses with independent axis validity. [Contour hierarchy](<../examples/opencv tools/contour_hierarchy/README.md>) exposes parent and child links, nesting depth, and holes. [Rotated rectangle overlap](<../examples/opencv tools/rotated_rectangle_overlap/README.md>) fits part footprints and measures their intersection. [Convex polygon overlap](<../examples/custom tools/convex_overlap/README.md>) measures continuous intersection, contact, and IoU from vertices or contour hulls. [Enclosing convex polygon](<../examples/opencv tools/enclosing_polygon/README.md>) fits bounded-side footprints and reports hull fill. [Delaunay triangulation](<../examples/opencv tools/delaunay_triangulation/README.md>) builds a planar mesh from point sites or measured centers. [Voronoi cells](<../examples/opencv tools/voronoi_cells/README.md>) partitions the image around sites and emits a selected-cell mask. [Minimum spanning tree](<../examples/custom tools/minimum_spanning_tree/README.md>) finds minimum-cost connections in weighted graphs and point meshes. Connected components and Find blobs expose OpenCV's default, SAUF, BBDT, and Spaghetti labeling algorithms. |
| Segmentation and texture | Native Sobel-based ridge response supports bright and dark features. Multiscale Sato filtering emits ridge strength and the strongest Gaussian scale per pixel. Frangi filtering remains open. Gray-level co-occurrence measures directional contrast and other texture properties from selected pixel pairs. Local binary patterns provide scikit-image texture codes and histograms for OpenCV comparison. SLIC, SLICO, MSLIC, LSC, and SEEDS emit integer superpixel labels and boundary masks. Measure label regions turns label IDs into filterable area, centroid, bounds, extent, and mean-luminance records. Mask from filtered labels returns the original pixels for the retained IDs. Niblack, Sauvola, Wolf, and NICK support locally adaptive thresholding. Graph-based segmentation emits a label plane and boundaries. Selective Search emits candidate boxes that work with collection filtering, sorting, and drawing. [Pyramid mean-shift filtering](<../examples/opencv tools/mean_shift_filter/README.md>) flattens local color variation before segmentation. [Local structure tensors](<../examples/opencv tools/structure_tensor/README.md>) provide selected normal/tangent maps, eigenvalues, coherence and explicit validity with an axial summary. [Gabor orientation bank](<../examples/custom tools/gabor_orientation_bank/README.md>) compares multiple angles and reports orientation confidence. [Gabor texture response](<../examples/opencv tools/gabor_texture/README.md>) supports signed and quadrature responses with independently configurable orientations. [Pixel K-means](<../examples/opencv tools/kmeans/README.md>) is available. |
| Spectral and intensity processing | [Laplacian pyramids](<../examples/opencv tools/laplacian_pyramid/README.md>) (S1-124) expose signed floating/native CPU bands, level selection and odd-size reconstruction with optional floating-band gains. [Aligned image comparison](<../examples/opencv tools/aligned_image_comparison/README.md>) (S1-119) provides selected ECC/normalized correlation, per-channel scores, explicit constant handling and normalized residuals. [Linear channel transform](<../examples/opencv tools/channel_transform/README.md>) mixes channels with an editable or wired matrix and optional per-channel offsets. [Color normalization](<../examples/opencv tools/normalize_color/README.md>) supports shared or independent channel scales and optional mask or region selection. [DCT/IDCT](<../examples/opencv tools/dct_round_trip/README.md>) supports reversible monochrome transforms. [DFT/IDFT and radial filtering](<../examples/custom tools/fourier_filter/README.md>) support band selection. [Slanted-edge sharpness](<../examples/custom tools/slanted_edge_sharpness/README.md>) (S1-175) estimates ESF/LSF/SFR, MTF50/10 and edge width, with ROI/mask support, sampling controls and explicit failures for unsuitable edges. [Hann window](<../examples/opencv tools/hann_window/README.md>) exposes 2D or row tapering before a transform. [Per-channel intensity LUTs](<../examples/opencv tools/intensity_lut/README.md>) support uint8 and uint16. [Intensity transforms](<../examples/opencv tools/intensity_transform/README.md>) include log, exp, square root, power, and clipping. [Image arithmetic](<../examples/opencv tools/image_arithmetic/README.md>) combines two images or an image and a scalar. [Pixel comparisons](<../examples/opencv tools/pixel_comparison/README.md>) produce masks from relational tests. [Find nonzero pixels](<../examples/opencv tools/nonzero_pixels/README.md>) extracts bounded coordinate lists and full-mask bounds. [Intensity profiles](<../examples/custom tools/intensity_profile/README.md>) reduce rows or columns by channel and can return extrema coordinates. [Image summary statistics](<../examples/opencv tools/image_statistics/README.md>) measure masked channel statistics, arbitrary percentiles, and extrema locations. [Custom Fourier gain masks](<../examples/custom tools/fourier_gain/README.md>) support per-bin gains and conjugate pairing. [Fourier correlation](<../examples/opencv tools/fourier_correlation/README.md>) measures linear or circular image displacement. [Tilted window statistics](<../examples/custom tools/tilted_statistics/README.md>) measure batches of 45-degree rectangles. [Local window statistics](<../examples/opencv tools/local_statistics/README.md>) use summed-area tables for rectangular mean, standard deviation, and sample-count maps. |
| Image metrics | [Absolute image error](<../examples/custom tools/absolute_image_error/README.md>) provides MAE, RMSE, maximum absolute error, and error maps. [Earth Mover’s Distance](<../examples/opencv tools/earth_movers_distance/README.md>) measures histogram shifts using bin-center distances. BRISQUE scores image quality without a reference. Laplacian variance and Tenengrad check focus within an inspection region or mask. Binary perceptual hashes support near-duplicate checks. Other no-reference scores and defect score calibration remain open. |
| Motion and background | Dual TV-L1, DeepFlow, RLOF, Kalman filtering, and tracking across missing detections remain open. Farneback, DIS and dense-grid Lucas–Kanade can emit forward–backward consistency masks. Dense-grid Lucas–Kanade also reports interpolation support. MOG2, KNN, GSoC and LSBP provide adaptive foreground masking. |
| Calibration models and target tooling | Capture-set coverage (S3-127), calibrated measurement uncertainty (S3-128), wide-angle object pose (S3-130), independently calibrated relative pose (S3-131), fisheye/omnidir fitting, multi-camera extrinsics and sparse bundle adjustment remain open. [Circle grid calibration](<../examples/opencv tools/circle_grid_calibration/README.md>) supports symmetric and staggered targets for planar mapping and lens fitting. [Lens correction maps](<../examples/opencv tools/lens_correction_maps/README.md>) exposes reusable pinhole undistortion and rectification maps in the graph. [Lens map displacement](<../examples/custom tools/lens_map_displacement/README.md>) measures and visualizes correction magnitude, direction, and source coverage. [Lens map files](<../examples/custom tools/lens_map_files/README.md>) saves and reloads complete map pairs with output intrinsics. [Lens fit report](<../examples/custom tools/lens_fit_report/README.md>) adds per-view reprojection RMS and native uncertainty estimates. |
| Depth and stereo | Edge-aware depth filtering (S3-19), time-of-flight cleanup (S3-73), alternative stereo algorithms and dense calibration evaluation. Stereo BM/SGBM supports [WLS confidence and left–right consistency](<../examples/opencv tools/stereo_confidence/README.md>). [Depth hole filling](<../examples/custom tools/depth_hole_filling/README.md>) is available. |
| Point cloud and reconstruction | The [point-cloud API audit](opencv-scope-map.md#point-cloud-and-mesh-api-families) tracks native and fallback paths for import/export, indexing, sampling, segmentation, plane fitting, registration and hypothesis ranking, mesh I/O, rasterization and bounded fusion. Spatial axis fitting (S3-129) remains open. Existing ICP, PPF, depth/point-cloud conversion and TSDF cover the baseline. |
| DNN task adapters | Multi-input model binding, dynamic shapes and batching, detection/rotated detection, NMS, semantic/instance masks, keypoints and OCR postprocessing, confidence calibration, portable preprocessing and CPU-vs-accelerator validation. Existing ONNX inference, classification and embeddings provide the base. |
| Classical ML | Persist and reuse fitted K-means centers for prediction, and add train/predict/export with SVM, KNearest, trees/forests, boosting, EM and other `ml` estimators. Add feature-table preparation and train/validation split. [Numeric K-means fitting](<../examples/opencv tools/kmeans/README.md>) is available. |
| Media and image formats | [Source-over alpha composition](<../examples/opencv tools/alpha_composition/README.md>) (S1-132) handles straight/premultiplied layers, association-aware chaining, external alpha, typed/color conversions and selected background-preserving composition. [Paired-frame interpolation](<../examples/opencv tools/frame_interpolation/README.md>) (S1-130) uses bidirectional flow and validated inverse remaps, reports support and explicit hole fill, and carries compatible media/explicit timestamps and both source identities. [Packed MIPI sensor frames](<../examples/opencv tools/packed_raw_sensor/README.md>) (S1-127) unpack RAW10/12 with explicit strides, headers, bit alignment and optional pixel frames. Metadata-aware Bayer processing follows integer crops/flips/rotations. [High-bit-depth codecs](<../examples/opencv tools/high_bit_depth_codecs/README.md>) (S1-126) provide exact PNG/JP2 uint16 round trips, explicit lossless/lossy JP2 modes and PNG filter/buffer controls. [Floating-point codecs](<../examples/opencv tools/float_image_codecs/README.md>) preserve TIFF float32/64 precision and qualify Radiance RGBE loss; EXR remains unqualified on this build (S1-125 partial). [Bayer demosaicing](<../examples/opencv tools/bayer_demosaicing/README.md>) supports four tile patterns, three interpolation modes, and crop parity. [In-memory PNG and JPEG](<../examples/opencv tools/in_memory_image_codecs/README.md>) provide typed packets and explicit compression settings. [TIFF page replay](<../examples/opencv tools/tiff_page_sequence/README.md>) emits ticketed pages from a multipage file. [Multipage TIFF output](<../examples/opencv tools/multipage_tiff_output/README.md>) writes ordered lossless image batches. [Raw YUV decoding](<../examples/opencv tools/raw_yuv/README.md>) handles seven packed 8-bit 4:2:0 and 4:2:2 layouts. [Image metadata and EXIF orientation](<../examples/opencv tools/exif_orientation/README.md>) report decoded chunk types and selected EXIF fields and can orient JPEG/PNG inputs. Metadata writing, other sensor formats, and additional codecs and numerical formats remain open. |

[Feature covariance and Mahalanobis distance](<../examples/opencv tools/feature_covariance/README.md>) cover joint variation and numeric outlier checks.
[Solve linear system](<../examples/opencv tools/linear_system/README.md>) provides exact and least-squares
solutions with multiple observation columns, numerical rank, and residuals.
[Solve linear program](<../examples/opencv tools/linear_program/README.md>) optimizes continuous
nonnegative variables with linear constraints and emits solver status and constraint slack.
[Solve polynomial roots](<../examples/opencv tools/polynomial_roots/README.md>) supports analytic real
roots through degree three, iterative complex roots, and relative residual judgments.
[Multiply matrices](<../examples/opencv tools/matrix_multiply/README.md>) applies weighted real matrix
products with an optional addend and independent transposition controls.
[Invert matrix](<../examples/opencv tools/matrix_inverse/README.md>) supports LU, Cholesky and SVD
with numerical rank, reciprocal conditioning, and determinants for square matrices.
[Singular value decomposition](<../examples/opencv tools/matrix_svd/README.md>) exposes the spectrum,
thin or full factors, low-rank reconstruction and relative truncation error.
[Eigendecomposition](<../examples/opencv tools/matrix_eigen/README.md>) exposes symmetric and
nonsymmetric real eigenpairs, numerical basis rank, and residual checks.
[Check image samples](<../examples/custom tools/image_validity/README.md>) reports nonfinite and
out-of-range pixels with source-aligned masks, a first location, and a fraction limit.
[Replace nonfinite samples](<../examples/custom tools/nonfinite_replacement/README.md>) repairs
NaN and signed infinity by channel, preserves the image encoding and depth units,
and emits a mask of changed pixels.
[Join images](<../examples/custom tools/join_images/README.md>) builds horizontal or vertical
comparison views with padding and per-panel coordinate context.
[Paste image patch](<../examples/custom tools/paste_patch/README.md>) performs a masked
copy at a canvas offset and reports both rectangles and the copied pixels.
[Subpixel patch extraction](<../examples/opencv tools/subpixel_crop/README.md>) samples a
rectangular patch at a fractional center and preserves coordinate provenance.
[Hough lines from points](<../examples/opencv tools/point_set_lines/README.md>) finds lines
from sparse coordinates and exposes vote counts for collection sorting.
[Hough polar lines](<../examples/opencv tools/hough_polar_lines/README.md>) supports binary
and intensity-weighted raster voting with masks and regions.
[Clip line segments](<../examples/custom tools/clip_line_segments/README.md>) trims measured
lines to image or rectangular region bounds and measures the visible span.
[Hough circle collections](<../examples/opencv tools/hough_circle_collections/README.md>) expose
native strengths for Gradient and Gradient ALT and support radius filtering.
[Polar image warp](<../examples/opencv tools/polar_image_warp/README.md>) unwraps circular
features onto linear or semilog grids, tracks source coverage, and can rewrap
an intact grid into its original frame.
[Tile image](<../examples/custom tools/tile_image/README.md>) repeats full images in a grid
with compact layout data and source-frame provenance.
[Transpose image](<../examples/opencv tools/transpose_image/README.md>) swaps image axes
with exact main- or anti-diagonal reflection and coordinate mapping.
[Route image channels](<../examples/custom tools/route_channels/README.md>) copies, reorders,
and duplicates channels across matching input images, with optional zero fill.
[Labeled distance transform](<../examples/opencv tools/labeled_distance/README.md>) computes
pixel distances, nearest-seed labels, and a mask for one selected zone.
[Masks from signed labels](<../examples/custom tools/label_masks/README.md>) shows exact
pixel comparisons and inclusive ranges on int32 segmentation outputs.
[Blend with weight maps](<../examples/custom tools/blend_weight_maps/README.md>) blends aligned
images using pixel weights and optional mask and region selection.
[Gram matrix](<../examples/opencv tools/gram_matrix/README.md>) computes row or column dot products
with optional centering, reusable offsets, and sample or population normalization.
[Find nearest rows](<../examples/custom tools/nearest_rows/README.md>) adds exact top-K feature-table
search, candidate masks, mutual nearest matching, and numeric or packed-byte distances.
[Sort numeric table](<../examples/custom tools/numeric_table_sort/README.md>) ranks whole records or
individual columns and rows, preserving input order for ties and returning source indices.
[Cartesian and polar fields](<../examples/opencv tools/polar_fields/README.md>) convert in both directions and preserve angle units.
[Iterative phase correlation](<../examples/opencv tools/iterative_phase_correlation/README.md>)
estimates subpixel translation and emits a correction for the moving image.
[Matched-feature translation](<../examples/opencv tools/translation_registration/README.md>)
fits a translation with RANSAC or LMeDS and reports per-pair residuals.
[Packed remap coordinates](<../examples/opencv tools/packed_remap/README.md>) supports nearest
and subpixel fixed-point maps and retains lens correction provenance.
[Checkerboard capture quality](<../examples/custom tools/chessboard_quality/README.md>)
measures edge width and brightness in both board directions.
[Paired Sobel and derivative-fed Canny](<../examples/opencv tools/canny_derivatives/README.md>) expose signed gradients and accept custom derivative fields.
[Scale absolute to 8-bit](<../examples/opencv tools/scale_absolute/README.md>) applies gain
and offset in stored units, saturates large magnitudes, and supports mask and
region selection.
[Principal components](<../examples/opencv tools/principal_components/README.md>) supports fitted dimensionality reduction and reconstruction.

## P3 — Specialized families

| Family | Outstanding capability |
| --- | --- |
| Photo and exposure | Decolorization, motion deblur, `xphoto` white-balance variants and learning-based super-resolution. Exposure stacks support MTB alignment, Mertens fusion, Debevec and Robertson response calibration and HDR merge, and four tone-mapping operators. Seamless clone and edge-aware photo filters have blocks. |
| Recognition | WeChat QR, QR encoding, multiple-code layout, cascade and HOG detector workflows, saliency, line descriptors, image retrieval/vocabulary and OpenCV-specific OCR variants. Face-specific models are outside the inspection baseline. |
| Sequence | Additional tracker families, background subtractors in `bgsegm`, motion history and advanced video stabilization. |
| Geometric utilities | Expanded robust estimators, graph matching, rectification variants, affine-invariant feature variants, panorama detail steps and coordinate-space diagnostics. |
| Diagnostics and visualization | Display overlays as reusable blocks, density maps, 3D axes and camera frusta, disparity confidence views and inspection measurement plots. Review feature matches draws inlier and outlier layers on a comparison canvas. [Colorize scalar image](<../examples/opencv tools/colorize_scalar/README.md>) provides built-in or custom palettes, fixed and automatic ranges, and invalid-pixel masking. |
| Contrib and domain tools | Fuzzy and bioinspired processing, structured-light modes beyond Gray code, saliency, radial-variance and color-moment hashing, and optional SfM, mesh/visualization and hardware-specific modules where the Python build supplies them. |

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
**GrabCut seeds from masks** builds those markers from aligned binary masks.
Any nonzero pixel selects the class of its input. Unmarked pixels start in
the configured probable class. Definite masks override probable masks.
A foreground and background overlap at the same certainty is an error.
Connect **Markers** to GrabCut's **Markers** input and select **Labels** mode.
The builder reports each class count and can emit a color preview.
[OpenCV GrabCut reference](https://docs.opencv.org/5.0/main_modules/imgproc_segmentation.html)
**Find quadrilateral in mask** checks outer contours in descending area.
It requires a convex four-vertex approximation and can reject contours
near the image edge or whose area changes too much during approximation.
It outputs clockwise corners starting at the smallest x+y, with an upper
corner chosen on a tie. This is an image-space order. Use a separate
orientation cue when the part can appear upside down.
Connect **Corners** to **Transform from control points** with a known
destination rectangle, then warp the original image with **Warp perspective**.
Expose **Result graphics** and use **Overlay result graphics** to review
the numbered corners.
**Thin binary mask** exposes Zhang–Suen and Guo–Hall skeletonization.
The optional scikit-image blocks expose Zhang, Lee, Guo–Hall, and medial-axis
skeletonization, plus length-based pruning of terminal branches. See the
[skeleton pruning example](<../examples/custom tools/skeleton_pruning/README.md>).
**Fill holes in a binary mask** fills enclosed background components, with an
optional maximum hole area and configurable background connectivity.

**Compute image hash** wraps OpenCV average, perceptual, block-mean,
and Marr–Hildreth binary hashes. Use **Compare image hashes** to count
differing bits between matching methods and settings. The distance check
can accept near-duplicate frames. Crop upstream to compare one region,
and choose the bit limit using examples from the job.
[OpenCV image hashing API](https://docs.opencv.org/5.0/extra_modules/img_hash.html)

**Measure BRISQUE quality** uses a local OpenCV SVM model and its matching
range file. It reports a 0–100 score, with lower values indicating better
quality according to the model. An optional maximum score checks the
capture. Use an upstream crop to score an inspection window. Compare
scores for similar scenes and resolutions with the same model.
[OpenCV image quality API](https://docs.opencv.org/5.0/extra_modules/quality.html)

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
directly to **Inspect each pose in parallel**. Open the [25-part example](<../examples/opencv tools/template_instances/template_instances.json>)
or run it with `opencv-dataflow-run "examples/opencv tools/template_instances/template_instances.json"`.


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
[Region from fitted shape](<../examples/custom tools/fitted_shape_region/README.md>) turns
ellipse or enclosing-circle fits into reusable image selections with an axis margin.

**Compare contours (Hu moments)** measures similarity between ordered point
lists or selected contour/blob collection items. **Contour hull and convexity
defects** reports hull vertices, solidity and concavity depths from a
whole-pixel contour. **Test points against contour** classifies points as
inside, on the edge or outside, and can filter them by signed distance to the
boundary. See the [OpenCV shape-analysis reference](https://docs.opencv.org/5.0/main_modules/geometry_shape.html).

[Fit circle or arc](<../examples/opencv tools/circle_arc_fitting/README.md>)
(S1-134) fits geometric circles to ordered image points or selected contour/blob
boundaries, using normalized CPU least squares and optional RANSAC consensus.
It reports radial error, signed arc sweep, observed angular envelope, gap/order
checks, numerical conditioning and conditional radial-noise uncertainty. Short
or unstable support withholds primary geometry; arc order and RMSE have separate
judgments. Masks/regions, mapped image frames, tickets, original pixels and
pyramid context are retained. Geometric circles feed **Region from fitted shape**.
The original noisy/outlier and short-arc projects compare independent polar truth.
This solves a different problem from the native minimum enclosing circle.

In a two-object fixture with one hole, the pinned OpenCV 5.0.0
`findContoursLinkRuns` binding reported the hole contour without a parent link.
The standard contour path remains the supported option for hierarchy.
[Approximate convex contour](<../examples/opencv tools/convex_contour_approximation/README.md>)
exposes fixed-side reduction with an added-area budget and verifies enclosure.

[Rasterize contours with holes](<../examples/opencv tools/contour_rasterization/README.md>)
(S1-133) converts contour or blob collections to binary masks. Select by current
index or detection ID and keep descendant holes/islands. Independent subtrees
union; explicit solid and even-odd modes support other polygon semantics.
Missing or inconsistent hierarchy links raise validation errors. Filled and
outline modes expose binary connectivity, fixed-point precision, offsets,
mask/region clipping and optional pixel-count judgments. Collection origins,
frames, tickets and pyramid context are retained; the optional source image
keeps its pixels, depth and encoding. Original cutout references verify exact
topology and selected areas. This is a raster mask, not fractional alpha coverage.


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
The [conveyor pick example](../examples/applications/conveyor_pick/conveyor_pick.json) supplies a
synthetic obstacle mask and runs with
`opencv-dataflow-run examples/applications/conveyor_pick/conveyor_pick.json`. The check measures visible
image-plane clearance. Depth, gripper reachability and swept-volume collision
checks remain separate work.


## Available bead and mask clearance checks

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
toward the gap. The [caliper example](../examples/applications/bead_calipers/bead_calipers.json) measures
a 126-pixel synthetic strip and runs with
`opencv-dataflow-run examples/applications/bead_calipers/bead_calipers.json`.

**Mask clearance at reference** samples an exact Euclidean distance transform
at registered reference pixels. It reports missing foreground separately from
covered samples outside clearance limits. The distance is from pixel center to
nearest background pixel center, with a one-pixel background pad at image edges.
It estimates local margin in pixels around the expected path. Use paired-edge
calipers to measure edge-to-edge width. A mask or region limits sample locations
while retaining the full foreground neighborhood. The two output masks include
every affected sample. The review draws at most 256 points per status.

**Inspect path continuity** consumes an ordered Path intensity profile. It
checks a selected channel against an inclusive range, measures contiguous
out-of-range lengths in path pixels, and merges a gap across a closed path's
start. Each sample represents the interval between neighboring midpoints,
clipped at open endpoints. Unavailable samples split runs and are checked
separately. Gap lengths are estimates at the configured sampling resolution.
The gap list includes every run, while the overlay draws the 256 longest.

**Unwrap path strip** uses OpenCV remapping to straighten a band around an
open or closed path. Columns advance along the centerline, and rows sample
across its local normal. It emits a validity mask and X/Y source-coordinate
maps on a distinct strip grid. Invalid coordinates are -1 and invalid pixels
are zero. Closed paths include the closing segment without a duplicate end
column. Keep the strip narrow on tight bends to limit overlap. Along-step
spacing describes the centerline and cross-step spacing is in source pixels.

**Inspect strip width** measures the contiguous foreground run across each
column's nominal center. It needs an observed background sample on each side.
Masked samples and clipped edges are reported as unmeasurable, while observed
background at the center is reported as missing. Connect the strip sampling
data and validity mask for source-pixel widths. The width estimate is the
sample count times the cross-step. Coverage and width tolerance have separate
limits and ticket reasons.

**Map strip graphics to source** projects outlines, labels and markers
through the strip's X/Y maps into the original camera frame. It samples
outlines along their length and splits them at invalid map samples. The
output remains drawable objects for Overlay result graphics and filmstrip
review. Layers from a different frame or ticket and unsupported primitives
such as fills and arrows are counted as skipped.

**Group strip defects** turns adjacent columns of the same failure type
into runs with column ranges, lengths, and width extrema when available.
With strip sampling data, lengths follow the source centerline and runs of
the same type can span the start of a closed path. A minimum report length
filters short runs without changing the width verdict. Run graphics can be
mapped back onto the source image.

**Judge strip defects** checks the complete grouped report against per-kind
maximum run counts and lengths. Unlisted kinds have a zero-run limit. At
least one width must be measurable. Turn off **Enforce limits** on the width
block to let this policy supply the upstream verdict for Final ticket judgment.
Bind the policy's `limits` parameter to a versioned recipe field to switch
acceptance criteria without changing measurement or report settings.

**Rectify workplane** uses a planar calibration to produce a uniform physical
grid and a validity mask. Wire its scale to **Unwrap path strip** on that grid
and wire its validity mask to the strip's mask input. Strip width and defect
length then use the calibration's mm, cm, m or in units. The strip geometry
controls still use rectified pixels. Set the same units on **Judge strip
defects**. The perspective transform preserves the link to the camera grid
for overlays. This measurement applies on the calibrated plane.

**Validate planar calibration** measures held-out fiducial errors against a
fixed planar mapping. It checks both physical RMS and worst-point error and
emits the checked mapping only on pass. Connect that output to Rectify
workplane to withhold calibrated measurements when validation fails.

**Fit planar calibration** produces a planar mapping from ordered pixel and
physical references. It supports direct and robust homography fits and reports
fitting inliers and physical RMS. Configure ArUco IDs to establish point order.
Use distinct IDs for held-out validation before rectification.

**Inspect workplane rectangle** checks the long and short sides of a fitted
rectangle using the workplane's physical scale. Its optional long-axis check
is modulo 180 degrees and rejects ambiguous near-square fits. It withholds
numeric measurements if the footprint touches invalid coverage or the image
edge. The measured footprint can be drawn on the original camera image.

**Inspect planar point spacing** maps two detected points from the camera
grid through a planar calibration and checks their physical distance. It
withholds the distance when a usable pair is unavailable. It draws the
endpoints and dimension on the camera image without rectifying the image.
Both points must lie on the calibrated plane.

**Inspect workplane ellipse** checks the fitted minor and major diameters,
their difference, and the contour-to-ellipse RMS error in physical units.
Wire an ellipse from the shape fit and the image, scale and validity mask
from the same workplane. It withholds measurements if the fitted footprint
touches the workplane edge or invalid coverage.
