# Image, video and point-cloud coverage checklist

This is the implementation scope for stages 1–4 of the [roadmap](roadmap.md).
The [coverage backlog](opencv-coverage.md) keeps broader candidates and background.
The [OpenCV scope map](opencv-scope-map.md) ties this checklist to the upstream
source modules and the installed Python namespaces. It records the graph
capabilities assigned to each family and why other APIs are deferred. Earlier
checklist IDs retain their stage.
The native API column names the main functions or building blocks for each job.
Each row represents one independently verifiable capability, even when several
native calls or graph blocks implement it.

<!-- coverage-summary:start -->

| Stage | Complete | Partial | Planned | Total | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. Core and baseline 2D | 99 | 20 | 56 | 175 | 56.6% |
| 2. ML and DNN | 9 | 11 | 42 | 62 | 14.5% |
| 3. 3D imagery and point clouds | 33 | 27 | 71 | 131 | 25.2% |
| 4. Selected contrib | 20 | 3 | 79 | 102 | 19.6% |
| **Overall** | 161 | 61 | 248 | 470 | 34.3% |

<!-- coverage-summary:end -->

## Counting rules

- **Complete** means a built-in graph path exists for the stated scope with a
  reproducible example or automated fixture. Completed rows give the block IDs
  that anchor the baseline. This is coverage of the declared graph capability
  on the supported OpenCV 5.0.0 contrib build.
- **Partial** means a relevant path is shipped and the row names the extension
  still needed. **Planned** means the target capability has yet to ship.
- Divide **Complete** by the total rows in each stage. The overall number is
  total completed rows divided by total rows. Partial work earns zero in both
  calculations. Stable IDs stay with their capabilities. When scope changes,
  document the added, removed, or split IDs and the corresponding denominator
  change in the commit.
- Scope includes images, video, image-derived descriptors, camera geometry, and
  depth, reconstructed surfaces, and imported inspection point clouds. General matrix
  operations, generic table ML, GUI/runtime infrastructure, PLC and robot
  transport, hardware qualification, and application-specific job packages
  are outside this denominator.
- Assign overlaps to one stage. Stage 2 owns learned vision and image-feature
  models, stage 3 owns camera geometry, depth and point clouds, and stage 4 owns the remaining
  selected extra modules. For build-dependent rows, confirm the target Python
  binding and verify the supported fallback or backend before marking complete.
- The [raw Python binding inventory](opencv-bindings.tsv) helps discover APIs
  and aliases. This checklist defines the denominator for progress reporting.

Run `python scripts/vision_coverage.py` to calculate progress. Use `--update`
after changing the checklist to refresh the table, and `--check` to verify
the saved counts, scope IDs and declared Python bindings. The [block parameter reference](block-parameters.md) lists current ports
and configuration. For planned or partial rows, the last column gives the
fixture or result needed to close the item.

## Stage 1 — OpenCV core and baseline 2D

[Stage tracker](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) ·
[Sequence and measurement follow-up](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-01 | Tiled inspection mosaic | Registered tile windows, warpPerspective, feather halos and coverage masks | Complete | cv.register_scan_layout, cv.render_mosaic_tile. Register up to 64 views and render individual bounded windows beyond dense-canvas limits. A 12-view example and dense-render comparisons verify coverage, overlap, pixel depth, tile boundaries and source transforms; allocation guards are tested on a large virtual canvas. |
| S1-02 | Continuous centered stabilization | calcOpticalFlowPyrLK, estimateAffinePartial2D, warpAffine, bounded look-ahead | Complete | sequence.stabilize_stream. A 144-frame example and longer synthetic streams verify fixed look-ahead/state bounds, original tickets and timestamps, coverage, reset, scene boundaries and automatic tail flush, including chained ToolBlocks and acquisition queues. CPU fixtures match the batch smoother across supported pixel depths. |
| S1-03 | Local still-image replay and save | imread, imwrite | Complete | input.file, output.image. Read typed image files, replay a list or folder with filename tickets, and save the result. |
| S1-04 | Image encode/decode in memory | imencode, imdecode | Complete | cv.encode_image, cv.decode_image. PNG/JPEG buffer round trips with compression settings and spatial provenance. |
| S1-05 | Multipage TIFF replay and save | imdecodemulti, imencodemulti on file-backed buffers | Complete | input.tiff_file, output.tiff_file. Ordered, ticketed page ranges and lossless batch output. |
| S1-06 | JPEG/PNG metadata and orientation read | imreadWithMetadata, EXIF orientation transforms | Complete | input.file. Report metadata and exercise the saved-image orientation fixture. |
| S1-07 | Image metadata write and buffer round trip | imwriteWithMetadata, imencodeWithMetadata, imdecodeWithMetadata | Planned | Round-trip supported metadata types through JPEG/PNG files and buffers. Verify orientation semantics and reject unsupported chunks explicitly. |
| S1-08 | Raw sensor color decoding | demosaicing, cvtColor | Complete | cv.demosaic_bayer, input.raw_yuv, cv.decode_yuv. Bayer phase at a crop and packed YUV layouts. |
| S1-09 | Camera image capture | VideoCapture.open, read, release | Complete | input.camera. Graph camera frame contract with backend and format settings. Physical backend qualification is tracked under acquisition. |
| S1-10 | Video file replay and recording | VideoCapture, VideoWriter | Complete | input.video_file, output.video. Frame bounds, stride, stop/loop, frame timestamps, and a local round trip. |
| S1-11 | Timestamp-paced file video | VideoCapture timestamps and replay clock | Partial | input.video_file exposes media timestamps. Verify paced playback with a fake clock, variable gaps, stride, and prompt Stop/Resume. |
| S1-12 | Network video input | VideoCapture with a supported network backend | Planned | Replay a local test stream and verify reconnect, bounded read failure, and clean stop. |
| S1-13 | Color conversion and pixel depth | cvtColor, convertTo, convertScaleAbs | Complete | cv.gray, cv.convert, cv.scale_absolute. Grayscale, color-space/depth conversion, and signed magnitude display. |
| S1-14 | Channel split, merge and linear mixing | split, merge, mixChannels, transform | Complete | cv.split_channels, cv.merge_channels, cv.route_channels, cv.channel_transform. Verify monochrome planes and multi-source channel order. |
| S1-15 | Image arithmetic and comparison | add, subtract, multiply, divide, absdiff, compare, inRange | Complete | cv.image_arithmetic, cv.absdiff, cv.compare_pixels, cv.in_range. Test mixed depths, zero divisors, masks, and integer labels. |
| S1-16 | Image masking, bitwise operations and blending | bitwise_and/or/xor/not, addWeighted, blendLinear | Complete | cv.apply_mask, cv.bitwise, cv.blend, cv.blend_weight_maps. Selected processing and per-pixel weights. |
| S1-17 | Intensity transforms and normalization | normalize, LUT, log, exp, pow | Complete | cv.normalize, cv.intensity_lut, cv.intensity_transform, cv.gamma. Masked transforms and channel-wise 8/16-bit LUTs. |
| S1-18 | Image statistics and nonfinite samples | meanStdDev, minMaxLoc, norm, findNonZero, checkRange-style validation | Complete | cv.image_statistics, cv.image_norm, cv.find_nonzero_pixels, inspect.image_validity, cv.replace_nonfinite. Report extrema, selected values, invalid pixels and replacements. |
| S1-19 | Histogram, comparison and backprojection | calcHist, compareHist, calcBackProject | Complete | inspect.histogram, inspect.compare_histograms, inspect.histogram_backproject. Per-channel summaries, binwise distances and selected intensity distributions. |
| S1-20 | Local and directional statistics | integral, integral2, tilted integral, sampled reductions | Complete | cv.local_statistics, cv.tilted_statistics, cv.intensity_profile, cv.path_intensity_profile. Window and path statistics with sample counts. |
| S1-21 | Contrast equalization and scalar palettes | equalizeHist, createCLAHE, applyColorMap | Complete | cv.equalize, cv.clahe, cv.colorize_scalar. Global/local contrast and scalar previews with validity masks. |
| S1-22 | Gaussian smoothing | GaussianBlur | Complete | cv.gaussian_blur. Verify kernel, sigma, border and selected processing. |
| S1-23 | Custom convolution and sharpening | filter2D, sepFilter2D, unsharp blending | Complete | cv.filter2d, cv.separable_filter, cv.unsharp. Verify editable kernels and source encoding. |
| S1-24 | Derivatives, edges and focus | Sobel, Scharr, Laplacian, spatialGradient, Canny | Complete | cv.gradient, cv.spatial_gradient, cv.canny, cv.canny_derivatives, inspect.focus. Signed responses and selected focus scores. |
| S1-25 | Fixed, automatic and adaptive thresholds | threshold, thresholdWithMask, adaptiveThreshold | Complete | cv.threshold, cv.auto_threshold, cv.adaptive_threshold. Fixed, Otsu, Triangle, mean and Gaussian workflows. |
| S1-26 | Morphology and reconstruction | getStructuringElement, erode, dilate, morphologyEx | Complete | cv.morphology, cv.morph_reconstruct. Standard or editable kernels and constrained reconstruction. |
| S1-27 | Flood fill, holes and nearest seeds | floodFill, distanceTransform, distanceTransformWithLabels | Complete | inspect.flood_fill, inspect.fill_mask_holes, cv.distance_transform, cv.distance_labels. Mask fill and labeled distance maps. |
| S1-28 | Foreground and touching-part segmentation | grabCut, watershed, seed masks and points | Complete | inspect.grabcut, inspect.watershed, inspect.distance_markers, inspect.watershed_point_seeds. Known foreground and separated-object fixtures. |
| S1-29 | Image color clustering | kmeans | Complete | inspect.kmeans_image. Cluster image pixels and report label and quantized image outputs. |
| S1-30 | Blob and component measurements | connectedComponentsWithStats, SimpleBlobDetector | Complete | inspect.components, inspect.blobs, inspect.simple_blob_detector. Area, centroid, outline, and configurable filtering. |
| S1-31 | Contour retrieval, holes and approximations | findContours, contourArea, arcLength, approxPolyDP, approxPolyN | Complete | inspect.contours, inspect.contour_hierarchy, inspect.approximate_contour, inspect.approximate_convex_contour. Retain nesting and bounded polygon fits. |
| S1-32 | Contour comparison, containment and hulls | moments, HuMoments, matchShapes, convexHull, convexityDefects, pointPolygonTest | Complete | inspect.compare_contours, inspect.contour_points, inspect.contour_hull, inspect.contour_profile. Shape comparison and defects with measurable tolerances. |
| S1-33 | Shape fitting and overlap | fitLine, fitEllipse/AMS/Direct, minEnclosingCircle, minAreaRect, rotatedRectangleIntersection, intersectConvexConvex | Complete | inspect.fit_shape, inspect.bounding_rotated_rectangle, inspect.rotated_rectangle_overlap, inspect.convex_polygon_overlap. Fits, intersections and IoU. |
| S1-34 | Line and circle detection | HoughLines/P/PointSet and accumulator variants, HoughCircles, createLineSegmentDetector | Complete | inspect.hough_polar_lines, inspect.hough_lines, inspect.hough_point_lines, inspect.hough_circles, inspect.line_segments. Geometry, strengths where native, and overlays. |
| S1-35 | Calipers and oriented edge checks | Sobel or sampled derivatives, remap | Complete | inspect.oriented_edges, inspect.bead_calipers, cv.unwrap_path_strip, inspect.strip_width. Edge direction, paired widths and curved-path width. |
| S1-36 | Image resize and integer crops | resize and integer ROI extraction | Complete | cv.resize, cv.crop. Verify interpolation, supported sample types and coordinate round trips. |
| S1-37 | Image composition | hconcat, vconcat, repeat, masked copy | Complete | cv.join_images, cv.tile_image, cv.paste_patch. Bound composition size and retain panel or placement coordinates. |
| S1-38 | Affine and projective warps | getAffineTransform, getPerspectiveTransform, warpAffine, warpPerspective, perspectiveTransform | Complete | cv.control_point_transform, cv.warp_affine, cv.warp_perspective, cv.rectify_quadrilateral, cv.transform_points. Defined transform direction and output grid. |
| S1-39 | Floating-point image remapping | remap | Complete | cv.remap. Verify dense X/Y maps, interpolation, borders and selected writes. |
| S1-40 | Gaussian image pyramids | pyrUp, pyrDown | Complete | cv.pyramid_up, cv.pyramid_down, cv.image_pyramid, cv.pyramid_level. Select a level and track source coordinates. |
| S1-41 | Discrete Fourier transform round trip | dft, idft | Complete | cv.dft, cv.idft. Recover images from complex coefficients and verify reconstructed samples. |
| S1-42 | Corner and feature detection | goodFeaturesToTrack, cornerHarris, cornerMinEigenVal, cornerSubPix, FAST | Complete | inspect.corners, inspect.corner_response, inspect.refine_corners, inspect.fast. Response maps, refined points and selections. |
| S1-43 | Binary and float local descriptors | ORB, SIFT, AKAZE, KAZE, BRISK | Complete | inspect.orb, inspect.sift, inspect.akaze, inspect.kaze, inspect.brisk. Descriptor data linked to pixel coordinates. |
| S1-44 | Descriptor matching and review | BFMatcher, FlannBasedMatcher, knnMatch, radiusMatch | Complete | inspect.match_sift, inspect.match_orb, inspect.match_flann, inspect.match_descriptor_radius, inspect.review_matches. Ratio, cross-check, radius and inlier review. |
| S1-45 | Template matching with multiple instances | matchTemplate, local-peak selection, subpixel refinement | Complete | inspect.match_template, inspect.template_instances. Multiple candidates with scores, locations and configurable suppression. |
| S1-46 | Trainable shape and texture matching | Contour geometry, ORB/SIFT descriptors, homography | Complete | pattern.train_shape, pattern.match_shape, pattern.train_texture, pattern.match_texture. Reusable models and image poses. |
| S1-47 | Generalized Hough shape search | createGeneralizedHoughBallard, createGeneralizedHoughGuil | Complete | inspect.hough_template, inspect.guil_template. Fixed or rotation/scale-aware search with votes. |
| S1-48 | Registered reference alignment | phaseCorrelate, findTransformECCMultiScale, findTransformECCWithMask, estimateAffine2D, estimateAffinePartial2D, findHomography | Complete | inspect.phase_correlation, inspect.phase_correlation_iterative, inspect.align_ecc, inspect.affine, inspect.homography, inspect.translation. Robust alignment with residual or quality checks. |
| S1-49 | Bounded scan stitching | Local registration, warpPerspective, seam and exposure blending | Complete | cv.stitch_image_pair, cv.stitch_scan_sequence. Overlapping pairs and ordered short scans with masks, drift correction, and exposure compensation. |
| S1-50 | Recorded and streaming motion baseline | calcOpticalFlowPyrLK, estimateAffinePartial2D, warpAffine | Complete | sequence.stabilize_camera, sequence.stabilize_batch. Causal and finite centered correction with coverage and validation. |
| S1-51 | Sparse and dense optical flow | calcOpticalFlowPyrLK, calcOpticalFlowFarneback, DISOpticalFlow | Complete | sequence.track_points, sequence.lk_grid_flow, sequence.farneback, sequence.dis. Flow and consistency checks. |
| S1-52 | Baseline background subtraction | createBackgroundSubtractorMOG2, createBackgroundSubtractorKNN, temporal median | Complete | sequence.mog2, sequence.knn_background, sequence.temporal_median. Learning and foreground masks on replay. |
| S1-53 | Seeded object and conveyor tracking | TrackerCSRT, TrackerKCF, image-space association | Complete | sequence.track_csrt, sequence.track_kcf, sequence.track_conveyor. Seeded boxes and multi-part image/encoder tracking. |
| S1-54 | Track across missed image detections | KalmanFilter and gated assignment | Partial | sequence.track_conveyor predicts with encoder motion. Add a reusable image-center predictor with confidence, bounded missed frames and crossing-part tests. |
| S1-55 | Reference-based image error | absdiff, norm, PSNR | Complete | inspect.image_quality. Selected MAE, RMSE, maximum absolute error and PSNR on aligned frames. |
| S1-56 | Illumination and color inspection | cvtColor, channel gains, flat-field/shading correction, Lab difference | Complete | calibration.color_balance, cv.flat_field, cv.adaptive_background, inspect.measure_lab, inspect.compare_lab. Selected Lab statistics and tolerance. |
| S1-57 | Masked image inpainting | inpaint | Complete | cv.inpaint. Verify Telea and Navier–Stokes reconstruction over a selected region. |
| S1-58 | Deblurring with a known blur kernel | DFT-based Wiener inverse | Complete | cv.wiener_deblur. Gaussian, defocus, line-motion and editable point-spread functions. |
| S1-59 | Bracketed exposure stack alignment | createAlignMTB | Complete | cv.align_exposures. Recover known shifts and preserve common image support. |
| S1-60 | Native flat QR decoding | QRCodeDetector.detectAndDecode and detectAndDecodeMulti | Complete | inspect.read_qr. Decode single or multiple flat codes and report image-space outlines. |
| S1-61 | Visible support for 2D pick candidates | Projected template support, masked comparisons and contour coverage | Partial | inspect.select_pick ranks score, angle and clearance. Verify a visible-support score on clear, touching, and partly covered parts. |
| S1-62 | Image and pose-relative selections | Rectangle, circle and polygon rasterization, Boolean masks | Complete | region.shape, mask.shape, region.combine, region.pose, region.fitted_shape. Apply editable and measured selections to image operations. |
| S1-63 | Object collection filtering and sorting | Contour/component measurements, stable ranking and source overlays | Complete | collection.filter, collection.sort, collection.clip_lines, collection.draw. Retain IDs through size, orientation, shape and position filters. |
| S1-64 | Stable intensity regions | MSER.detectRegions | Complete | inspect.mser. Stable grayscale and color regions as image coordinates and filterable bounds. |
| S1-65 | Planar neighborhoods and cell masks | Subdiv2D Delaunay and Voronoi | Complete | inspect.delaunay_triangulation, inspect.voronoi_cells. Recover fixture neighbor geometry and select a cell mask. |
| S1-66 | Mean-shift color smoothing | pyrMeanShiftFiltering | Complete | cv.mean_shift_filter. Flatten local color variation while retaining boundaries for segmentation. |
| S1-67 | Multi-reference color classification | Lab statistics and CIE76/CIEDE2000 distances | Complete | inspect.classify_lab. Rank configured reference colors and reject out-of-tolerance samples. |
| S1-68 | Compare reference edge maps | distanceTransform and symmetric edge tolerances | Complete | inspect.compare_edge_maps. Report missing and added edges after allowing bounded position error. |
| S1-69 | Map selections through nonlinear lens correction | remap and lens map validity | Partial | region.transform handles affine/projective mapping and calibration.undistort_lens_points maps points. Carry an entire source selection into the corrected image and verify a masked measurement through a nonlinear warp. |

### Geometry, filtering and photo operations

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-70 | Image borders and padding modes | copyMakeBorder | Complete | cv.copy_border. Verify constant and reflected padding and shifted source coordinates. |
| S1-71 | Fractional patch sampling | getRectSubPix-style bilinear sampling | Complete | cv.crop_subpixel. Report fractional origin and a coverage mask on known shifted patches. |
| S1-72 | Arbitrary-angle image rotation | getRotationMatrix2D, warpAffine | Complete | cv.rotate. Exercise fitted and clipped canvases with aligned downstream selections. |
| S1-73 | Exact right-angle rotations and transpose | rotate, flip, transpose | Complete | cv.rotate90, cv.flip, cv.transpose_image. Preserve exact samples, labels and coordinate mapping. |
| S1-74 | Box smoothing and local sums | blur, boxFilter | Complete | cv.box_blur, cv.box_filter. Select normalized or sum output, depth and border behavior. |
| S1-75 | Median smoothing | medianBlur | Complete | cv.median_blur. Suppress impulse noise with an odd kernel. |
| S1-76 | Bilateral smoothing | bilateralFilter | Complete | cv.bilateral. Smooth selected noise while retaining intensity edges. |
| S1-77 | Gabor texture response | getGaborKernel, filter2D | Complete | cv.gabor_response, cv.gabor_bank. Compare signed and quadrature responses, strongest angle and confidence. |
| S1-78 | Gradient magnitude and angle fields | cartToPolar, polarToCart | Complete | cv.cart_to_polar, cv.polar_to_cart. Recover a known signed vector field with explicit angular units. |
| S1-79 | Discrete cosine transform round trip | dct, idct | Complete | cv.dct, cv.idct. Recover an image from typed coefficients with measured error. |
| S1-80 | Spectral filtering and correlation | mulSpectrums, createHanningWindow, spectral gain masks | Complete | cv.combine_fourier_spectra, cv.hann_window, cv.filter_fourier_spectrum, cv.mask_fourier_spectrum, inspect.fourier_correlation. Verify periodic suppression and signed displacement. |
| S1-81 | Exposure fusion to display range | createMergeMertens | Complete | cv.fuse_exposures. Fuse aligned brackets without a response curve and compare selected detail. |
| S1-82 | HDR response and radiance reconstruction | createCalibrateDebevec, createCalibrateRobertson, createMergeDebevec, createMergeRobertson | Complete | cv.calibrate_debevec, cv.calibrate_robertson, cv.merge_debevec, cv.merge_robertson. Compare radiance against a known exposure sequence. |
| S1-83 | HDR tone mapping | createTonemap, createTonemapDrago, createTonemapReinhard, createTonemapMantiuk | Complete | cv.tonemap_hdr. Convert HDR radiance using the selected operator and verify output range. |
| S1-84 | Estimate an unknown blur kernel | Blind point-spread estimation and regularized DFT restoration | Planned | Recover motion direction or blur extent from a controlled degraded image and check sensitivity to noise. |
| S1-85 | Photo detail and edge-aware smoothing | detailEnhance, edgePreservingFilter | Complete | cv.detail_enhance, cv.edge_preserving_photo. Verify the available photo smoothing modes and detail enhancement. |
| S1-86 | Photo stylization and sketching | stylization, pencilSketch | Complete | cv.stylize_photo, cv.pencil_sketch. Expose image parameters and both sketch outputs. |
| S1-87 | Seamless image patch composition | seamlessClone | Complete | cv.seamless_clone. Verify masked placement and preserved surroundings on a synthetic patch. |
| S1-88 | Illumination and texture correction | illuminationChange, textureFlattening | Planned | Reduce a known glare gradient or texture while retaining a measured boundary inside the mask. |
| S1-89 | Masked color transfer | colorChange | Planned | Adjust a selected surface's color while retaining original surroundings and edge structure. |
| S1-90 | Contrast-preserving decolorization | decolor | Planned | Separate equal-luminance color features and compare against ordinary grayscale. |
| S1-91 | Non-local means denoising | fastNlMeansDenoising, fastNlMeansDenoisingColored, multi-frame variants | Complete | cv.non_local_means, cv.temporal_nlm. Verify single-frame and aligned-stack restoration with bounded memory. |
| S1-92 | TV-L1 denoising | denoise_TVL1 | Complete | cv.denoise_tvl1. Verify noise reduction on single images and aligned observations. |
| S1-93 | Image remapping to polar and log-polar coordinates | warpPolar and inverse map | Complete | cv.warp_polar. Verify circular geometry and source coverage through rewrap. |
| S1-94 | Packed remap conversion and reuse | convertMaps, remap | Complete | cv.pack_remap_map, cv.remap_packed. Compare fixed-point and floating-map output and retain calibration provenance. |
| S1-95 | Warp a video frame using a dense flow field | remap and flow-to-sample coordinates | Planned | Warp a frame through known motion and report invalid, occluded and uncovered pixels. |

### Selections, measurement, video and decoding

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-96 | Independent RGB and alpha channel handling | split, merge, extractChannel, insertChannel | Complete | cv.split_alpha_channels, cv.merge_alpha_channels, cv.extract_channel, cv.insert_channel. Verify exact BGRA round trips and replacement. |
| S1-97 | Label-plane region measurement and mask recovery | Per-label area, centroid, extents and selection | Complete | inspect.measure_label_regions, inspect.mask_from_labels. Recover precisely the selected label pixels. |
| S1-98 | Foreground seed authoring from masks | GrabCut label masks and watershed marker labels | Complete | inspect.grabcut_mask_seeds, inspect.watershed_mask_seeds. Check overlapping and invalid seeds and recover intended labels. |
| S1-99 | Detect a quadrilateral from a segmentation mask | findContours, approxPolyDP | Complete | inspect.find_quadrilateral. Verify ordered corners and rejection of unsupported shapes. |
| S1-100 | Bounded-side convex enclosure | minEnclosingTriangle, minEnclosingConvexPolygon | Complete | inspect.bounding_convex_polygon. Report containment and added area against the input hull. |
| S1-101 | Image-space regions and pose transforms | Homogeneous 2D image geometry and mask rasterization | Complete | pose.define, pose.from_feature, pose.offset, region.transform, region.filter_points, region.measure. Verify offset axes and region coverage. |
| S1-102 | Clearance and discontinuity on an image path | distanceTransform, ordered path sampling | Complete | inspect.mask_clearance, inspect.path_continuity. Distinguish isolated misses and persistent gaps, including on a closed path. |
| S1-103 | Curved-strip defect extraction and source graphics | remap, contiguous spans and mapped geometry | Complete | inspect.strip_defects, inspect.judge_strip_defects, inspect.map_strip_graphics. Measure defect spans and map them back onto the source. |
| S1-104 | Registration of unordered rotating views | Stitcher or explicit feature registration and panorama projection | Planned | Build a panorama with known overlapping views and report disconnected or inconsistent views. |
| S1-105 | Deterministic image and noise generation | Seeded Gaussian and uniform noise | Complete | input.random_image, cv.add_image_noise. Verify reproducible frames and selected noise with preserved sample types. |
| S1-106 | Image annotation and reusable overlays | polylines, markers and text rasterization | Complete | cv.annotation_layer, cv.overlay_result_graphics. Compose selected image-frame layers and preserve geometry on resampling. |
| S1-107 | Stateful image references | Ticketed image memory, running background estimate | Complete | sequence.image_memory, sequence.background. Verify learning, reset, and reference reuse across replay frames. |
| S1-108 | Region tracking by color distribution | meanShift, CamShift, histogram backprojection | Planned | Track an object across translation and scale change with a lost-track rule. |
| S1-109 | Robust keypoint-free motion refinement | VariationalRefinement | Planned | Refine an initial flow on a known motion field and verify residuals and invalid borders. |
| S1-110 | Video seeking by media time | VideoCapture CAP_PROP_POS_MSEC and timestamped replay | Planned | Seek by time on fixtures with known frame timing and report the effective position. |
| S1-111 | Animated image decoding and encoding | imreadanimation, imwriteanimation, imdecodeanimation, imencodeanimation | Planned | Round-trip a supported animation with per-frame durations, frame order and format checks. |
| S1-112 | Native QR encoding | QRCodeEncoder | Planned | Generate plain and multi-segment labels where supported and round-trip them through the native reader. |
| S1-113 | Compare decoded text to an expected value | Image-derived text normalization and comparison | Complete | inspect.verify_text. Verify expected strings, mismatch reasons and ticket propagation after decoding. |
| S1-114 | Multi-format barcode decoding | ZXing image decoding adapter | Complete | inspect.read_zxing. Decode formats including Data Matrix with a repeatable fixture when the optional decoder is installed. |
| S1-115 | Internal and external camera capture settings | VideoCapture property readback | Partial | input.camera exposes backend and capture settings. Report negotiated resolution, rate, format and unsupported requests using a controlled source fixture. |
| S1-116 | Robust differential exposure compensation | Local brightness and contrast agreement on aligned images | Planned | Verify a registered reference under changing illumination and report exposure-adjusted and unadjusted defect residuals. |
| S1-117 | Multi-view exposure seams and blending | detail exposure compensators, GraphCutSeamFinder, MultiBandBlender | Partial | cv.stitch_scan_sequence uses gain/offset, feathering and graph-cut seams. Verify multiband blending across overlapping tiles with bounded memory and preserved image depth. |
| S1-118 | Resample coverage masks conservatively | Area sampling, explicit coverage threshold | Planned | Resample a validity mask to a different pixel grid and verify partially covered pixels and source provenance. |
| S1-119 | Correlation-based image comparison | computeECC and masked normalized correlation on aligned image pairs | Planned | Compare supported image encodings with constant-region handling, per-channel results and a defect fixture. |
| S1-120 | Transport distance between image histograms | EMD on image histograms | Complete | inspect.compare_histograms. Compare histogram signatures using the native transport distance and known distributions. |
| S1-121 | Curved QR decoding | QRCodeDetector.detectAndDecodeCurved | Partial | inspect.read_qr exposes curved mode. Qualify a known wrapped label and verify source and rectified module coordinates. |
| S1-122 | Native linear barcode decoding | barcode.BarcodeDetector | Complete | inspect.read_barcode. Verify EAN or UPC text, pose and outline. |
| S1-123 | Variable-frame-rate and unreliable video timing diagnostics | VideoCapture properties and timestamp checks | Partial | input.video_file labels decoder timestamps and estimated timing. Qualify variable-frame-rate and missing or incorrect metadata fixtures. |

### Precision, multiscale processing and video fidelity

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-124 | Laplacian image pyramids | pyrDown, pyrUp and detail Laplacian-pyramid routines | Planned | Expose signed residual levels and verify reconstruction across odd image dimensions and preserved source frames. |
| S1-125 | Floating-point image codec round trips | imread, imwrite and format-specific float encoding | Planned | Verify float TIFF and Radiance HDR precision and channel order. Qualify OpenEXR on a build that provides it. |
| S1-126 | Lossless high-bit-depth compressed images | JPEG 2000 and PNG codec options | Planned | Verify exact 16-bit round trips at supported settings and reject silent conversion or unavailable codecs. |
| S1-127 | Packed high-bit-depth sensor frames | RAW10/RAW12 unpacking and demosaicing | Planned | Decode known sensor byte patterns with stride, Bayer phase and bit-depth checks. |
| S1-128 | Affine-invariant image feature extraction | AffineFeature.create with a supported feature detector | Planned | Measure correspondence repeatability under known view tilts and preserve point scale, orientation and source frame. |
| S1-129 | Local structure tensor and dominant orientation | cornerEigenValsAndVecs | Planned | Expose eigenvalue and orientation maps with controlled borders and verify edges, corners and low-confidence flat areas. |
| S1-130 | Motion-compensated frame interpolation | Forward/backward flow and remap | Planned | Create intermediate frames for known motion, report occlusion coverage and preserve media timing. |
| S1-131 | Intensity-weighted center and orientation | moments with intensity weights and region selection | Planned | Compare geometric and intensity-weighted centers and axes on a known asymmetric feature. |
| S1-132 | Per-pixel alpha composition | Straight and premultiplied alpha, source-over composition | Planned | Verify transparent boundaries and float or integer results on foreground/background pairs. |
| S1-133 | Rasterize selected contours with holes | drawContours, fillPoly and contour hierarchy | Planned | Convert selected contour collections into masks with topology intact and verify source-grid alignment. |
| S1-134 | Ordered circle/arc fitting and angular extent | Least-squares or robust circle/arc fitting from ordered image edge points | Planned | Fit known partial circles amid noise and outliers. Report center, radius, residual, angular coverage and uncertainty, and check outlier bias and short-arc degeneracy. Minimum enclosing circles do not establish this fit. |
| S1-135 | Reconstruct features by local frequency and phase | DFT, magnitude and phase of image bands | Partial | cv.dft emits complex spectra. Qualify phase-only or magnitude-only reconstruction with declared channel and normalization behavior. |
| S1-136 | Binary mask boundary distances | distanceTransform and directed boundary comparisons | Partial | inspect.contour_profile measures matched contour boundaries. Qualify a source-grid mask comparison with tolerance bands and missing-area statistics. |
| S1-137 | Build a flat-field reference from image stacks | Selected frame averaging or median | Planned | Estimate dark and bright references from repeated captures and verify correction on a held-out image. |
| S1-138 | Focus stacking and per-pixel focus selection | Local focus response, image alignment and masked blending | Planned | Fuse different focal planes on a synthetic or recorded stack and verify the focus-index and coverage maps. |
| S1-139 | Log-polar rotation and scale registration | warpPolar and phaseCorrelate | Planned | Recover rotation and scale before translation and verify ambiguity handling on symmetric parts. |
| S1-140 | Controlled video encoder qualification | VideoWriter, fourcc and decoder round trip | Partial | output.video writes tested local streams. Check representative lossless and lossy codecs for frame count, rate, color and invalid backend selection. |
| S1-141 | Debounce foreground across consecutive frames | Temporal mask accumulation and connected components | Planned | Suppress single-frame foreground spikes and retain a sustained moving feature with bounded state and reset behavior. |
| S1-142 | Synthetic fiducial-free image distortion fixtures | Geometric remap and seeded perturbations | Planned | Generate known shear, perspective and lens-like distortions and emit ground-truth maps for inspection regression. |
| S1-143 | Image contrast and visibility between two regions | Selected image statistics and contrast-to-noise ratio | Planned | Report signal/background contrast, noise and invalid selections on a controlled inspection target. |
| S1-144 | Unicode image annotation | FreeType or Qt glyph rasterization and frame-aware graphics | Partial | cv.annotation_layer draws existing image labels. Verify multilingual OCR or inspection text renders on the annotated output image with preserved coordinates. |

### Optical corrections and panorama refinement

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-145 | Hard-shadow detection and compensation | Illumination models, color ratios, masks and guided boundary handling | Planned | Replay the same part with a cast-shadow edge. Report the shadow mask and verify defect contrast and boundary preservation on both views. |
| S1-146 | Correct lateral chromatic aberration | correctChromaticAberration, loadChromaticAberrationParams | Planned | Correct known color fringes with declared lens settings and measure channel alignment and invalid border coverage. |
| S1-147 | Estimate and refine camera parameters for panoramas | detail rotation estimators, bundle adjustment and wave correction | Planned | Fit a rotating-camera image set with known focal length and yaw. Verify overlap consistency, residuals, and bad-view rejection. |
| S1-148 | Fill stabilization borders for review | Video stabilization inpainting or neighboring-frame fill | Planned | Fill uncovered display pixels from aligned neighbors where possible and preserve a separate measured-coverage mask for inspection. Verify both outputs on known jitter. |

### Buffer I/O, exposure and binary shape tools

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-149 | Multipage images on memory buffer ports | imdecodemulti and imencodemulti | Partial | input.tiff_file and output.tiff_file use native memory codecs internally. Expose ordered, typed pages on a buffer port and verify a lossless TIFF round trip and bounded decode. |
| S1-150 | Moving-region rejection in HDR brackets | Exposure alignment, consistency masks and supported HDR merge | Planned | Use brackets with one moving part. Report rejected support and verify that static detail remains usable without treating reconstructed pixels as measured. |
| S1-151 | Binary hit-or-miss morphology | morphologyEx with MORPH_HITMISS and a ternary kernel | Planned | Expose foreground, background and don't-care kernel cells. Verify corners or junctions and reject incompatible encodings. |
| S1-152 | Memory-efficient contour extraction | findContoursLinkRuns | Planned | Compare contours and hierarchy with the supported dense-mask reference on large sparse masks. Verify the native binary input and topology limits. |

### Codec preflight, random access and video tracking

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-153 | Probe image codec availability and page inventory | haveImageReader, haveImageWriter and imcount | Planned | Verify readable formats, encoder availability, page counts and invalid file diagnostics before decoding large inputs. |
| S1-154 | Read and preserve optical-flow fields in files | readOpticalFlow and writeOpticalFlow | Planned | Round-trip a known signed flow field, validate the frame association and unsupported or invalid samples, and compare loaded motion against a computed reference. |
| S1-155 | Reuse precomputed optical-flow pyramids | buildOpticalFlowPyramid with optional derivatives | Planned | Verify level geometry, border and derivative controls, and parity against direct tracking. Qualify the Python representation or keep the cache inside the tracker. |
| S1-156 | Seeded MIL video tracking | TrackerMIL | Planned | Track an annotated part from a configured box and verify initialization, localization, failure reporting and reset over a recorded clip. |
| S1-157 | Decode video from a bounded encoded stream | VideoCapture and IStreamReader where supported | Planned | Verify a recorded in-memory stream with known timestamps and frames. Qualify the backend for partial reads, limits and deterministic teardown. |
| S1-158 | On-demand multipage image access | ImageCollection or indexed page decode | Partial | input.tiff_file replays selected page ranges. Qualify random page access on a large stack with bounded caching, identical decoded samples and stable page provenance. |

### Image smoothing, sequence statistics and code workflows

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-159 | Large-radius image smoothing | stackBlur | Planned | Compare against Gaussian smoothing on noisy, high-resolution images. Measure edge retention and runtime across odd kernel sizes and report the supported encodings. |
| S1-160 | Temporal per-pixel noise and covariance | accumulate, accumulateSquare, accumulateProduct and accumulateWeighted | Planned | Compute masked mean, variance and paired-channel or paired-source covariance from a replay stack. Verify sample counts, overflow resistance, reset and known noise levels. |
| S1-161 | ArUco-backed QR reading | QRCodeDetectorAruco | Planned | Compare both native QR backends on cluttered, rotated and small labels. Verify payload, rectified patch and corners on the original image, including multiple candidates. |
| S1-162 | Structured-append QR payloads | QRCodeEncoder.encodeStructuredAppend and a qualified decoder | Planned | Generate and recover a split payload in sequence, even when symbols are presented out of order. Report missing, duplicated and mismatched fragments without treating a partial payload as valid. |
| S1-163 | Keypoint repeatability and match accuracy | evaluateFeatureDetector, computeRecallPrecisionCurve or validated equivalents | Planned | Evaluate two registered images with known common support. Report repeatability and correspondence counts, then precision and recall versus match threshold with a held-out transform. |

### Reusable mosaic layouts and registered replay

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-164 | Reuse a calibrated mosaic layout | Stitcher.composePanorama and persisted camera or tile transforms | Partial | cv.register_scan_layout emits versioned geometry and cv.render_mosaic_tile reuses it without feature fitting, validating source order and grids. Qualify reuse with missing views identified by stable source IDs, and verify seams, coverage and source-coordinate mapping; reject incompatible grid changes. |
| S1-165 | Replay registered views on a common canvas | detail.Timelapser, detail.TimelapserCrop and common valid support | Planned | Render each view as a separate timestamped frame on the same grid. Verify union and intersection crops, invalid borders, and per-frame source-to-canvas maps. |

### Reference frequency response

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-166 | Measure image frequency response against a reference | dft and divSpectrums with a supported-band mask | Planned | Estimate relative gain and phase from aligned captures of a known target. Verify on a known blur or transfer function and suppress bins with insufficient reference energy. |

### Video trajectories and restoration

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-167 | Persist and replay a video motion trajectory | videostab.FromFileMotionReader and ToFileMotionWriter or a typed trajectory record | Partial | sequence.stabilize_camera emits a transform. Save a versioned trajectory with frame timestamps, direction and validity, then apply it to the same clip without refitting. Verify transformed coordinates, mismatched clips and reset. |
| S1-168 | Restore blurred frames from aligned neighbors | videostab.WeightingDeblurer or a tested equivalent | Planned | Use a recorded or synthetic clip with alternating sharp and motion-blurred frames. Verify sharper detail without introducing duplicate moving parts, preserve original coverage and report which neighbors contributed. |
| S1-169 | Correct rolling-shutter wobble in video | videostab.MoreAccurateMotionWobbleSuppressor or a tested equivalent | Planned | Reduce line bending on a clip with known row-dependent distortion. Verify near-edge geometry, valid image support and behavior when there is insufficient motion evidence. |

### Foreground references across restarts

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-170 | Restore a validated foreground reference after restart | BackgroundSubtractor.getBackgroundImage and versioned reference images or warmup replay | Partial | sequence.mog2 and sequence.knn_background emit background images. Save a camera-bound reference or verified warmup frames, reload them after a restart, and compare the first eligible foreground result. Reject mismatched calibration, size or encoding and keep known foreground out of relearning. |

### Track prediction and detector scheduling

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-171 | Predict oriented image tracks through missed detections | KalmanFilter, an unscented filter or a tested nonlinear predictor | Partial | sequence.track_conveyor predicts position from encoder motion. Predict image-space center, scale and orientation with uncertainty through gaps. Verify angle wraparound, timestamp intervals, reacquisition and explicit expiry on an annotated clip. |
| S1-172 | Refresh tracked objects with a periodic full-frame detector | DetectionBasedTracker or a graph scheduler with local tracking | Planned | Run full-frame detection at a configured cadence and refine tracks locally between refreshes. Verify recovery from drift, discovery of entering parts, stale-track expiry and the detector call budget on replay. |

### Image profile resampling

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-173 | Resample measured image profiles onto a common pitch | signal.resampleSignal with verified antialiasing or a tested equivalent | Partial | cv.path_intensity_profile supplies image-space profiles and sample coordinates. Align two profiles to a common pixel or physical pitch. Verify amplitude, peak position, antialiasing, invalid gaps and source-coordinate provenance for upsampling and decimation. |

### Image metrology and edge sharpness

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-174 | Measure corners and angles from fitted image lines | fitLine with frame-aware line intersection and angle measurement | Partial | inspect.fit_shape fits individual image lines. Measure the vertex and included angle of a line pair, preserve source frames or physical workplane units, and reject parallel or poorly conditioned intersections on known corner fixtures. |
| S1-175 | Measure sharpness from a slanted image edge | Subpixel edge sampling, edge-spread derivative and dft | Planned | Estimate an edge-spread curve and spatial-frequency response from a selected edge. Verify known blur levels, sample pitch, edge polarity and windowing, and report insufficient edge support or clipping. |

## Stage 2 — ML and DNN for images and video

[Stage tracker](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) ·
[Current ML workflows](ml.md)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-01 | Single-image ONNX inference | dnn.readNetFromONNX, blobFromImage, Net.setInput, Net.forward | Complete | ml.dnn. Local model load and a reproducible CPU inference fixture. |
| S2-02 | Named model outputs | Net.getUnconnectedOutLayersNames, Net.forward | Complete | ml.dnn. Select the named outputs and preserve tensor shapes for downstream decoders. |
| S2-03 | Multi-input inference | Net.setInput by name, typed image/tensor inputs | Partial | ml.dnn binds one input. Run a reproducible model with two independently wired inputs and reject missing or mismatched tensors. |
| S2-04 | Batched and dynamic-shape image inference | dnn.blobFromImages, backend shape validation | Partial | ml.dnn supports single-image inference. Verify mixed batch sizes, supported dynamic dimensions, result order, and per-image tickets. |
| S2-05 | Portable model preprocessing and execution receipt | dnn preprocessing, dnn.getAvailableTargets | Partial | ml.dnn stores input preprocessing and backend settings. Publish a revisioned model/preprocessing contract with fixture results and an explicit fallback report. |
| S2-06 | Image classification | Tensor softmax and top-K label decoding | Complete | ml.classify. Decode class scores, confidence, and labels. |
| S2-07 | Image embeddings | Output tensor vectorization and optional normalization | Complete | ml.tensor_vector. Carry a normalized image feature vector to search and matching. |
| S2-08 | Object detection and NMS | dnn.NMSBoxes, common detection tensor adapters | Planned | Decode boxes, classes and confidence from a pinned detector, apply class-aware NMS and undo letterboxing. |
| S2-09 | Rotated object detection | dnn.NMSBoxesRotated, oriented box adapter | Planned | Decode angle, dimensions and pose, and verify rotated suppression on overlapping parts. |
| S2-10 | Semantic segmentation | Per-pixel argmax, resize, remap | Planned | Decode a known model into a label plane and masks on the source grid, including ignored classes. |
| S2-11 | Instance segmentation | Box and mask tensor decoding, NMS, resize | Planned | Emit a distinct source-aligned mask and confidence for each retained detection. |
| S2-12 | Keypoint and heatmap decoding | Heatmap peaks, minMaxLoc, subpixel refinement | Planned | Recover ordered image-space landmarks and their confidence from a pinned model and a known landmark fixture. |
| S2-13 | CTC text recognition from image crops | Perspective crop, DNN inference, CTC decode | Complete | ml.recognize_text. Decode rectified crops with blank and repeat handling. |
| S2-14 | SIFT LightGlue matching | SIFT, official PyTorch LightGlue adapter, geometric verification | Complete | ml.lightglue_features, ml.lightglue. Offline model fixture checks matches, transform and score. |
| S2-15 | SuperPoint LightGlue adapter | SuperPoint detection/descriptor inference, matched LightGlue weights | Planned | Verify repeatable keypoint and match coordinates on the same image pair used by SIFT. |
| S2-16 | DISK LightGlue adapter | DISK detection/descriptor inference, matched LightGlue weights | Planned | Verify descriptor shapes and pose accuracy on a pinned local model and pair. |
| S2-17 | ALIKED LightGlue adapter | ALIKED detection/descriptor inference, matched LightGlue weights | Planned | Verify model input requirements and pose accuracy on a pinned local model and pair. |
| S2-18 | Image embedding retrieval | Annoy build, query, save and load | Complete | ml.annoy_build, ml.annoy_search, ml.annoy_save, ml.annoy_load. Index, rank, and persist image-derived embeddings. |
| S2-19 | Reusable image color or descriptor clusters | kmeans with persisted centers, nearest-center prediction | Partial | inspect.kmeans_image fits current-frame colors. Reuse a trained palette or visual vocabulary on separate captures and verify label consistency after reload. |
| S2-20 | SVM image classification | ml.SVM train, predict and model I/O | Planned | Train and reload a classifier on HOG or other image descriptors and verify held-out images with an explicit class mapping. |
| S2-21 | K-nearest image classification | ml.KNearest train and findNearest | Planned | Classify held-out image descriptors with neighbor distances and an unknown-class cutoff. |
| S2-22 | Random forest image classification | ml.RTrees train, predict and getVotes | Planned | Train on measured image features and verify held-out labels and reported class votes after reload. |
| S2-23 | Image classifier and detector evaluation | Image-level split, confusion matrix, IoU-based matching | Planned | Use labeled images to report per-class errors and confidence thresholds. Keep frames from one sequence together when splitting. |

### Model adapters and image-derived ML

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-24 | Image letterboxing and geometry round trip | Model input resize, border fill and source/network transforms | Complete | ml.dnn. Verify normalization, channel order, mask handling, pad sizes and inverse coordinates with known pixels. |
| S2-25 | SSD-style detector tensor adapter | Decode class, score and corners from DetectionOutput records | Planned | Decode a pinned SSD-style model and reject invalid coordinates and labels. |
| S2-26 | YOLO-style detector tensor adapter | Decode box/class outputs and apply class-aware NMS | Planned | Verify a pinned YOLO export, explicit axis layout, thresholds and inverse letterboxing. |
| S2-27 | Object-detection box fusion across image tiles | Model coordinate transforms, IoU and cross-tile suppression | Planned | Merge overlapping tile detections onto one source grid and verify boundary objects occur once. |
| S2-28 | Detection tracking across frames | Image detection association and optional re-identification embeddings | Planned | Verify stable object IDs and count identity switches and fragmented tracks on clips with occlusions, crossing parts and re-entry. |
| S2-29 | Prompted image segmentation | Image embedding and point/box prompt model inputs | Planned | Use one pinned ONNX model to verify prompt coordinate mapping and output masks on a controlled image. |
| S2-30 | Image segmentation over tiles | Per-tile inference and overlap-aware label or score fusion | Planned | Match full-frame labels on a shared fixture and preserve object edges across seams. |
| S2-31 | Text-region decoding from a probability map | Threshold, contour candidates and rotated rectangles | Complete | ml.text_regions. Verify oriented text boxes and coordinates on the original image. |
| S2-32 | Text detection with a pinned network | DNN text detector and DB-style score-map decoding | Partial | ml.text_regions decodes probability maps. Qualify a pinned detector with oriented labels and split/merged text fixtures. |
| S2-33 | OCR recognition across an image batch | Rectified crops, batch inference and CTC sequence mapping | Partial | ml.recognize_text handles rectified regions. Verify batched results retain region order and individual confidence. |
| S2-34 | OCR vocabulary and constrained decoding | CTC prefix search, alphabet and optional lexicon | Planned | Compare blank handling, repeats and industrial text constraints on known logits and image samples. |
| S2-35 | Learned single-image super-resolution | dnn_superres.DnnSuperResImpl | Planned | Pin a model and compare detail, noise and dimensional error against interpolation on downsampled images. Check for false or erased defects against a native-resolution reference. |
| S2-36 | Learned image restoration | ONNX denoising or deblurring model and overlap-safe output mapping | Planned | Run a pinned restoration model on degraded/reference pairs and report improvement and introduced artifacts. |
| S2-37 | Learned monocular depth | ONNX dense depth inference and explicit relative/metric output | Planned | Verify model scale conventions, source alignment, invalid samples and error on known depth imagery. |
| S2-38 | Learned stereo disparity | Paired-image ONNX inference and disparity scaling | Planned | Verify a known disparity plane and preserve masks, left/right frame and pixel units. |
| S2-39 | Learned optical flow | Paired-frame ONNX flow inference | Planned | Verify displacement direction, scale and valid coverage on known frame shifts. |
| S2-40 | Learned video object tracker | TrackerNano, TrackerVit, TrackerDaSiamRPN or equivalent local ONNX model | Planned | Pin model assets and verify init, update, loss and reset on annotated video. |
| S2-41 | Image-descriptor dimensionality reduction | PCACompute/PCAProject on extracted image features | Partial | cv.fit_pca and cv.transform_pca implement numeric PCA. Qualify a persisted descriptor pipeline on held-out images and track the feature model revision. |
| S2-42 | Texture and color mixture classification | ml.EM, per-pixel feature mixtures | Planned | Fit a probabilistic surface classifier from image samples, score held-out pixels and persist the model. |
| S2-43 | Image feature outlier detection | Image descriptors, covariance and Mahalanobis distance | Partial | cv.feature_covariance and inspect.mahalanobis_distance score feature tables. Qualify an image-only pipeline with known-good variation and held-out defects. |
| S2-44 | Bag-of-visual-words image retrieval | SIFT/ORB quantization and per-image visual-word histograms | Planned | Build image-level signatures from a trained codebook and verify ranking on held-out views. |
| S2-45 | Object landmark inference with model-specific skeleton | Keypoint decoding, confidence and source-coordinate graph | Planned | Decode and draw a configured skeleton and verify missing landmarks are suppressed. |

### Classical detection and model qualification

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-46 | HOG sliding-window object detection | HOGDescriptor.detectMultiScale with configured detector weights | Planned | Detect objects at multiple scales with per-window confidence and verify suppression on known image boxes. |
| S2-47 | Cascade-based image object detection | CascadeClassifier.detectMultiScale | Planned | Load a local configured cascade and verify detections, scale controls and output scores on a labeled fixture. |
| S2-48 | Rotated crop classification | Model preprocessing from measured image poses | Partial | ml.dnn consumes an image or selected region. Qualify independent oriented part crops with source geometry and per-part ticket retention. |
| S2-49 | Image-based regression | OpenCV SVM regression or ONNX scalar decoder | Planned | Estimate a continuous image-derived measurement on held-out captures and report residuals with model revision. |
| S2-50 | Image-model anomaly maps | Local embeddings and nearest-reference or model-specific anomaly scores | Planned | Train a known-good image reference set, localize anomalies and verify threshold decisions on independent defect images. |
| S2-51 | Detect image defects under input shifts | Model evaluation across exposure, blur and camera pose | Planned | Replay controlled image perturbations and report confidence, localization and false rejects against the reference results. |
| S2-52 | Learned instance tracking masks | Image segmentation plus temporal association | Planned | Track per-instance masks through overlapping or occluded video frames with ticket and ID continuity. |

### Native learned matching and retrieval

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-53 | Native ONNX LightGlue execution | LightGlueMatcher.create, setPairInfo and match | Partial | ml.lightglue has a PyTorch path. Qualify the native ONNX path on a pinned image pair with explicit keypoint/image context, scores, rejected matches, and pose parity. |
| S2-54 | Native OpenCV embedding index | ANNIndex build, knnSearch, save, load and setOnDiskBuild | Partial | ml.annoy_search, ml.annoy_build use a separate Annoy package. Qualify native index persistence and bounded on-disk build, compare rankings by metric against exact search, and verify stable image IDs. |

### DNN performance and detection qualification

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-55 | Profile image model layers | dnn.Net.getPerfProfile and getLayerNames | Planned | Qualify per-layer and total inference timings on a pinned model. Record engine and target, distinguish fused layers from unavailable timings, and verify predictable behavior when profiling is disabled. |
| S2-56 | Score-decay suppression for overlapping detections | dnn.softNMSBoxes | Planned | Verify linear and Gaussian score decay on overlapping parts. Keep class IDs, updated scores and original indices aligned and compare against hard suppression on adjacent detections. |
| S2-57 | Qualify quantized image model inference | ONNX QuantizeLinear/DequantizeLinear models and DNN engine selection | Planned | Compare a pinned integer-quantized vision model with its floating-point reference. Verify scale and zero-point handling, classification or detection tolerances, and diagnostics for an unsupported operator or execution target. |

### Multi-scale image reconstruction

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-58 | Produce several super-resolution scales from one inference | dnn_superres.DnnSuperResImpl.upsampleMultioutput or named DNN outputs | Planned | Qualify a pinned model with at least two named scales. Verify Python output delivery or a tested equivalent, output order, source-coordinate transforms, and feature localization against the native-resolution reference. |

### Train and qualify image classifiers and detectors

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-59 | Calibrate classifier confidence and abstain on uncertain images | ml.NormalBayesClassifier.predictProb, classifier scores and a held-out calibration set | Planned | Map scores from image descriptors to calibrated class probabilities using a separate split. Verify class order, uncertainty rejection and probability reliability on held-out known and ambiguous parts. |
| S2-60 | Learn an image anomaly score from known-good examples | ml.SVM with ONE_CLASS and a verified decision-score convention | Planned | Train and reload a model on known-good image descriptors. Verify inlier and outlier decisions and raw score direction on independent normal and defective parts. Preserve preprocessing and the threshold revision. |
| S2-61 | Train a HOG sliding-window detector from image crops | HOGDescriptor.compute, linear ml.SVM and HOGDescriptor.setSVMDetector | Planned | Train from labeled windows, mine false positives from separate backgrounds, and reload the detector vector with its HOG geometry. Verify score orientation and recall on held-out full images. |
| S2-62 | Consolidate repeated detections by spatial votes | groupRectangles and class-aware grouping | Planned | Merge repeated candidates into a representative box and vote count. Verify grouping tolerance, singleton policy, stable source IDs and separation of adjacent parts and different classes. |

## Stage 3 — 3D imagery and point clouds

[Camera and robot calibration](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2) ·
[3D expansion](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6) ·
[Current 3D workflows](vision-3d.md)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-01 | Detect and identify planar ArUco markers | aruco.ArucoDetector.detectMarkers | Complete | inspect.aruco_markers. Identify marker IDs and ordered image corners. |
| S3-02 | Checkerboard or circle-grid pinhole calibration | calibrateCameraExtended, projectPoints | Complete | calibration.lens, calibration.validate_lens. Fit intrinsics and distortion and check independent target reprojections. |
| S3-03 | Pinhole remapping and aligned image points | initUndistortRectifyMap, undistortPoints, remap | Complete | calibration.lens_maps, calibration.undistort_lens_points, calibration.undistort. Verify map/image dimensions, coverage and corrected points. |
| S3-04 | Fit fisheye intrinsics from target images | fisheye.calibrate, fisheye.projectPoints | Partial | 3d.camera and 3d.rectify_camera consume existing fisheye intrinsics. Fit a profile from observed targets and check held-out reprojection and valid coverage. |
| S3-05 | Fit omnidirectional intrinsics | omnidir.calibrate, omnidir.projectPoints | Partial | 3d.camera and 3d.rectify_camera consume an existing model. Estimate intrinsics and xi and verify held-out edge observations. |
| S3-06 | Planar registration and physical workplane | findHomography, perspectiveTransform, warpPerspective | Complete | calibration.fit_planar, calibration.validate_planar, calibration.rectify_workplane. Independent marker checks and a uniform physical image grid. |
| S3-07 | Physical dimensions from calibrated 2D images | Planar point mapping and scaled shape fitting | Complete | calibration.inspect_planar_distance, inspect.workplane_rectangle, inspect.workplane_ellipse, inspect.strip_width. Ticketed dimensions and tolerance reasons with source-aligned graphics. |
| S3-08 | Versioned camera and rig profiles | Versioned intrinsics, map and extrinsic records tied to an image grid | Partial | input.calibration, output.calibration, input.lens_maps and output.lens_maps cover profiles and maps. Add explicit revision selection and reject mismatched capture or grid metadata. |
| S3-09 | Project camera points and rays | projectPoints, undistortPoints and lens-family variants | Complete | 3d.project, 3d.undistort_points, 3d.rectify_camera. Pixel projection and ray/rectification fixtures for declared lens families. |
| S3-10 | Pinhole object and fiducial pose | solvePnP, solvePnPRansac, solvePnPRefineLM/VVS, Rodrigues | Complete | 3d.pnp, 3d.aruco_pose. Known pinhole poses, inlier and reprojection checks, and camera-relative frames. Wide-angle object pose remains S3-130. |
| S3-11 | Shared-pinhole relative pose and epipolar geometry | findFundamentalMat, findEssentialMat, recoverPose, computeCorrespondEpilines | Complete | 3d.fundamental, 3d.essential_pose, 3d.check_epipolar. Relative pose uses one shared pinhole camera model; fundamental fitting and epipolar checks operate on image correspondences. Reject outliers and report residuals. Separate camera models remain S3-131. |
| S3-12 | Two-view sparse triangulation | triangulatePoints | Complete | 3d.triangulate. Recover 3D positions from two supplied projection matrices and paired image points with cheirality and reprojection bounds. More than two views remain S3-33. |
| S3-13 | Stereo rig calibration and rectification | stereoCalibrate, stereoRectify, initUndistortRectifyMap | Complete | 3d.stereo_calibrate, 3d.stereo_rectify. Matched pinhole target observations and a rectified pair with projection and Q matrices. |
| S3-14 | Wide-angle stereo fitting and rectification | fisheye.stereoCalibrate/stereoRectify, omnidir.stereoCalibrate and rectification maps | Planned | Fit and rectify a wide-angle rig and check held-out vertical disparity and depth scale. |
| S3-15 | Stereo disparity | StereoBM, StereoSGBM | Complete | 3d.disparity. Known pixel shift, invalid mask and fixed-point scaling. |
| S3-16 | Stereo confidence and filtering | ximgproc.createRightMatcher, createDisparityWLSFilter, left–right consistency | Complete | 3d.disparity. Verify confidence on known correspondences and occlusion. |
| S3-17 | Disparity-to-depth reprojection | reprojectImageTo3D | Complete | 3d.reproject. Known baseline and depth plane, retained pixel positions and masks. |
| S3-18 | RGB-D depth and color registration | registerDepth, intrinsic and pose transforms | Complete | image.rgbd, 3d.register_depth. Pair color and depth and transform metric depth to the declared camera grid. |
| S3-19 | Depth cleanup and edge-aware smoothing | Invalid-depth masks, hole fill, joint bilateral or guided filtering | Partial | 3d.fill_depth_holes fills small gaps. Validate a depth-aware smoother at a foreground edge with invalid samples and confidence. |
| S3-20 | Organized depth normals | RgbdNormals | Complete | 3d.depth_normals. Verify normal orientation, pixel correspondence and invalid samples on depth planes. |
| S3-21 | PLY and ASCII PCD cloud interchange and preview | PLY, ASCII PCD, point attributes and orthographic preview | Complete | input.cloud, output.cloud, 3d.cloud_render. Round-trip imported or camera-derived clouds and display a bounded sample. |
| S3-22 | Camera surface registration | Point-to-plane ICP, point-pair feature detector | Complete | 3d.icp, 3d.ppf_train, 3d.ppf_match. Align two camera-derived surfaces and verify the recovered pose and score. |
| S3-23 | Depth plane and foreground separation | Robust plane fitting on backprojected pixels, organized masks | Planned | Fit a plane under clutter and emit signed depth residuals and foreground masks on the camera grid. |
| S3-24 | Image-space visibility of reconstructed surfaces | Project points and z-buffer on target intrinsics | Planned | Identify occluded versus visible surface samples and compare the predicted mask with known scene depth. |
| S3-25 | RGB-D odometry | rgbd.Odometry | Complete | 3d.odometry. Estimate relative camera motion from two aligned RGB-D frames with a reproducible fixture. |
| S3-26 | Posed depth fusion | OpenCV TSDF integration | Complete | 3d.tsdf_fuse. Integrate posed depth frames with explicit bounds and recover a surface cloud. |
| S3-27 | Mesh extraction from fused depth | TSDF surface extraction or marching cubes | Partial | 3d.tsdf_fuse produces a surface cloud. Verify mesh faces, units, and a bounded export on known geometry. |
| S3-28 | Gray-code structured light | structured_light.GrayCodePattern.generate and getProjPixel | Complete | 3d.graycode, 3d.graycode_decode. Known projector pixel mapping and shadow mask. |
| S3-29 | Phase-shift structured light | structured_light.SinusoidalPattern generate and phase decoding | Planned | Decode phase and unwrap a known camera/projector target, including low-confidence regions. Verify binding behavior on the supported build. |
| S3-30 | Camera-projector calibration and reconstruction | Projector as inverse camera, stereo calibration and triangulation | Partial | 3d.graycode_decode supplies correspondences. Validate calibrated depth against a known stepped target from an offline capture or simulation. |
| S3-31 | Eye-in-hand calibration from camera observations | calibrateHandEye, target reprojection and pose composition | Complete | calibration.hand_eye, 3d.pnp, 3d.compose. Offline target-to-camera and gripper-to-base fixture with held-out target consistency. |
| S3-32 | Fixed-camera and robot-world calibration | Eye-to-hand frame arrangement and joint robot-world/hand–eye solver | Partial | calibration.hand_eye supports eye-in-hand. Verify fixed-camera and joint solutions against held-out image-derived target observations. Confirm Python solver availability or validate a tested offline solver. |
| S3-33 | Sparse multi-view reconstruction | Feature tracks, recoverPose, multi-view triangulation, bundle adjustment | Partial | 3d.essential_pose and 3d.triangulate handle a pair. Verify tracks across three or more supplied or estimated camera views with missing observations and stable landmark IDs. Measure reprojection and scale behavior after refinement; qualify a tested triangulator because native sfm bindings are absent from the pinned wheel. |

### Calibration, stereo and depth extensions

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-34 | Detect checkerboards and assess capture quality | findChessboardCorners/SB, cornerSubPix | Complete | calibration.grid, calibration.chessboard_quality. Check corner spacing, blur and exposure before forwarding views. |
| S3-35 | Detect symmetric and staggered circle grids | findCirclesGrid, SimpleBlobDetector | Complete | calibration.grid. Verify ordered circle-grid observations with independent synthetic targets. |
| S3-36 | Detect partial ChArUco targets | aruco.CharucoDetector | Complete | calibration.charuco. Recover board and pixel correspondence by ID from partially visible views. |
| S3-37 | Generate planar fiducial targets | aruco.generateImageMarker, Board.generateImage | Planned | Render configurable marker and board layouts and verify a detection round trip with declared physical geometry. |
| S3-38 | ChArUco pinhole calibration with independent verification | calibrateCameraExtended from per-view ID subsets | Complete | calibration.lens, calibration.validate_lens, calibration.charuco. Fit from differing visible corner sets and verify errors by ID. |
| S3-39 | Inspect lens-map displacement | Source/remap point displacement magnitude and mask | Complete | calibration.lens_map_displacement. Verify distortion vectors and map validity against known displacement. |
| S3-40 | Camera matrix optimization for useful output support | getOptimalNewCameraMatrix, initUndistortRectifyMap | Complete | calibration.lens_maps. Compare cropped and extended fields of view and report output intrinsics. |
| S3-41 | Fixed and projective planar pixel-to-world mapping | Homogeneous pixel-to-physical transform | Complete | calibration.pixel_to_physical. Map known ordered landmarks without resampling the image. |
| S3-42 | Multiple planar pose hypotheses | solvePnPGeneric with IPPE or other supported methods | Planned | Retain plausible poses with reprojection scores and test near-ambiguous planar views. |
| S3-43 | Sparse epipolar error and correspondence refinement | sampsonDistance, correctMatches | Planned | Verify geometric residuals and correction on noisy point pairs, including degenerate epipoles. |
| S3-44 | Uncalibrated stereo rectification | stereoRectifyUncalibrated, warpPerspective | Planned | Rectify two image grids from image matches and verify vertical residual and valid support. |
| S3-45 | Stereo calibration residuals by camera and capture | stereoCalibrateExtended, projectPoints | Partial | 3d.stereo_calibrate reports fit RMS. Verify per-view/per-camera residuals against known held-out target data. |
| S3-46 | Optimize stereo search range and valid ROI | getValidDisparityROI, search-range policy | Planned | Constrain matching to shared support and verify invalid border bands and a known disparity range. |
| S3-47 | Disparity speckle removal | filterSpeckles | Planned | Remove isolated inconsistent disparity islands with explicit invalid-value and disparity-scale handling. |
| S3-48 | Evaluate stereo disparity against a reference | Bad-pixel rates, absolute error and confidence slices | Planned | Compare a disparity result to ground truth on valid pixels and report per-region error and coverage. |
| S3-49 | Decode, sample and report metric depth | Depth units and integer pixel sampling | Complete | input.depth_file, inspect.depth_sample. Report invalid samples as missing and preserve image locations and units. |
| S3-50 | Camera-derived cloud transform and measurement | Rigid XYZ transform with frame and unit checks | Complete | 3d.cloud_transform, 3d.cloud_measure. Verify known camera pose, physical extents and output frame. |
| S3-51 | Unorganized cloud normals from reconstructed points | Local neighborhood normals | Complete | 3d.cloud_normals. Recover normals of a known camera-derived surface with explicit neighbor support. |
| S3-52 | Project camera clouds into a depth or color image | Intrinsic projection and a depth buffer | Partial | 3d.cloud_render provides an orthographic preview. Emit metric camera-aligned depth and coverage using declared intrinsics and occlusion rules. |
| S3-53 | Evaluate camera depth against a reference | Unit-normalized depth error, validity overlap | Planned | Compare known depth planes and depth steps and report residual maps and missing regions. |
| S3-54 | Calibrate a rig with three or more image streams | calibrateMultiview, calibrateMultiviewExtended | Planned | Recover a known same-lens-model rig from partial views with explicit visibility masks. Verify per-view residuals, independent reprojections and frame closure. |
| S3-55 | Reproject a depth-backed region across wide-angle camera views | Camera rays, metric depth and target-camera projection | Planned | Map a physical patch between wide-angle cameras and verify it lands on the same scene area with coverage. |
| S3-56 | Range image fusion with depth confidence | Weighted depth accumulation in camera coordinates | Planned | Fuse registered depth views and verify confidence, occlusion boundaries and preserved thin objects. |
| S3-57 | Depth-derived foreground component measurement | Organized depth thresholding and connected components | Planned | Separate image-space parts at different depths and report physical size and source masks. |
| S3-58 | Local 3D features and correspondence from reconstructed views | Keypoint neighborhoods, geometric correspondence and robust alignment | Planned | Register partly overlapping reconstructed surfaces and report accepted correspondences and alignment error. |
| S3-59 | Metric depth backprojection | Intrinsics and depth-to-point projection | Complete | 3d.depth_cloud. Verify XYZ, invalid depth and per-pixel color/provenance from an aligned image. |
| S3-60 | Reconstructed cloud selection and voxel sampling | Spatial bounds, valid indices and voxel pooling | Complete | 3d.cloud_filter, 3d.cloud_voxel. Verify retained camera correspondence and bounded point density. |
| S3-61 | Binary stereo correspondence | stereo.StereoBinaryBM or StereoBinarySGBM | Planned | Requires a build with the Python binding. Compare a known disparity image to the regular BM/SGBM path. |

### Depth sequences and additional camera models

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-62 | Color and depth from a synchronized recorded stream | Timestamp pairing, explicit registration and depth confidence | Partial | image.rgbd pairs typed color and depth. Verify recorded streams with staggered timestamps and independent camera grids. |
| S3-63 | Stereo exposure and color consistency | Pairwise color correction on rectified correspondence support | Planned | Reduce photometric mismatch on a calibrated pair and compare valid disparity coverage. |
| S3-64 | Recover calibrated object dimensions from a depth silhouette | Depth backprojection and physical contour measurement | Planned | Measure known raised parts and report thickness, width and invalid boundary samples. |
| S3-65 | Scene-scale recovery from multi-view observations | Multi-view triangulation with known metric anchors | Partial | 3d.essential_pose supplies unit-scale motion. Verify metric recovery and error with held-out 3D targets. |
| S3-66 | Stereo-to-depth consistency across multiple baselines | Stereo reprojection and cross-view depth checks | Planned | Fuse valid baselines while identifying occlusion or outlier disparity on an independently measured depth fixture. |
| S3-67 | Phase unwrapping for measured fringe images | phase_unwrapping.HistogramPhaseUnwrapping and getInverseReliabilityMap | Planned | Recover a known wrapped surface phase. Verify a shadow mask, per-pixel inverse reliability and separately referenced connected regions. |
| S3-68 | Camera pose tracking against an object silhouette | rapid silhouette trackers and projected object geometry | Planned | Track a known metric object in video with reprojection checks and explicit loss and reset behavior. |
| S3-69 | Three-dimensional model pose from image and depth templates | linemod image gradients and depth normal templates | Planned | Recover an object pose from matched RGB-D templates and verify confidence and image registration. |
| S3-70 | Multi-view calibration with coded custom targets | ccalib and observed camera-target correspondences | Planned | Qualify a custom target across cameras with held-out frame closure. Confirm the relevant Python APIs or use a tested solver. |
| S3-71 | Depth-fusion trajectory estimation | KinFu or equivalent depth tracking and fusion | Partial | 3d.odometry and 3d.tsdf_fuse support pairwise motion and posed depth. Verify drift, failure recovery and a bounded trajectory on a recorded sequence. |
| S3-72 | Quasi-dense stereo correspondence | stereo.QuasiDenseStereo | Planned | Grow correspondences from stereo feature matches and compare disparity and support against known image pairs. |

### Depth cleanup, point-cloud processing and mesh workflows

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-73 | Time-of-flight depth cleanup | Depth confidence, neighborhood checks and edge-aware filtering | Planned | Use recorded depth with a range step and low-confidence samples. Flag flying pixels and multipath-like outliers while preserving real discontinuities and raw sample provenance. |
| S3-74 | Sparse pixel-to-point backprojection | depthTo3dSparse | Planned | Backproject selected image points without constructing a full cloud. Verify XYZ, units, frame, invalid points, and original pixel indices. |
| S3-75 | Normalize and quantize a metric depth image | rescaleDepth and typed depth conversion | Planned | Round-trip integer and float depth with explicit scale, invalid codes, clipping and quantization error. |
| S3-76 | Warp a registered RGB-D frame by camera pose | warpFrame | Planned | Transform depth and color from a known moving view and verify destination depth, mask, and occlusion. |
| S3-77 | Point-cloud octree and spatial queries | Octree.createWithDepth, createWithResolution, KNNSearch and radiusNNSearch | Planned | Index an imported or reconstructed cloud with explicit metric bounds. Compare query indices and distances against brute-force fixtures and verify memory bounds. |
| S3-78 | Random and farthest-point cloud sampling | randomSampling, farthestPointSampling or validated fallbacks | Planned | Verify seeded reproducibility, sample spacing, point budget, and retained attributes. The pinned wheel lacks these geometry bindings. |
| S3-79 | Representative-point voxel sampling | voxelGridSampling or equivalent voxel selection | Partial | 3d.cloud_voxel supports first point and centroid. Verify nearest-original-point representative selection, per-axis spacing, and associated attributes. The native geometry function is unavailable in the pinned Python wheel. |
| S3-80 | Radius and statistical cloud outlier rejection | Spatial queries and local neighbor statistics | Planned | Recover a known surface from sparse noise and report removed indices, density settings, and retained edge detail. |
| S3-81 | Euclidean cloud clustering | Radius-neighbor connected components in XYZ | Planned | Separate several imported parts with known clearances, retain source IDs, and reject clusters by physical size or count. |
| S3-82 | Normal-based cloud region growing | RegionGrowing3D or equivalent neighborhood segmentation | Planned | Separate touching surfaces by normal and curvature with stable labels. Qualify an alternate build or a tested fallback because the pinned wheel lacks the native binding. |
| S3-83 | Robust plane and sphere fitting on unorganized clouds | SACSegmentation RANSAC models or equivalent fitting | Planned | Fit known metric planes and spheres amid outliers and report coefficients, inliers, residuals, and failed degeneracy. The pinned wheel lacks the native class. |
| S3-84 | Local cloud curvature and normal quality | normalEstimate or equivalent neighbor PCA | Partial | 3d.cloud_normals estimates normals. Verify curvature, invalid neighborhoods and normal orientation on planes and spheres. Use a validated fallback on the pinned wheel. |
| S3-85 | Native OBJ point-cloud interchange | loadPointCloud and savePointCloud | Partial | input.cloud and output.cloud support PLY and PCD. Qualify the native OBJ path with aligned XYZ, normal and color fields and compare PLY behavior with the current path. |
| S3-86 | Triangle mesh interchange | loadMesh and saveMesh | Planned | Round-trip supported triangulated OBJ/PLY files with topology, optional vertex fields, units and frame. Reject invalid indices and unsupported layouts. |
| S3-87 | Project a mesh to color and depth images | triangleRasterize, triangleRasterizeColor, triangleRasterizeDepth | Planned | Rasterize known overlapping triangles with near/far clipping and culling. Verify camera convention, metric depth, colors and masks. |
| S3-88 | Cloud-to-cloud dimensional comparison | Bidirectional nearest-neighbor distances and overlap checks | Planned | Compare aligned measured and reference clouds, report signed distances where normals are reliable, and verify overlap and tolerance failures. |
| S3-89 | Compare an inspection scan to a reference mesh | Closest triangle distances and local reference normals | Planned | Measure a calibrated surface deviation on known geometry, with boundary and unsupported areas reported separately. |
| S3-90 | ICP overlap and local registration diagnostics | Point-to-plane ICP and paired inlier residuals | Partial | 3d.icp returns a pose and residual. Verify overlap, inlier counts and rejection of poor or ambiguous alignment with a known partial-overlap pair. |
| S3-91 | Refine and rank multiple surface-match hypotheses | PPF3DDetector candidates and ICP refinement | Partial | 3d.ppf_match returns candidates ranked by votes. Verify residual- and coverage-based ranking and suppress duplicate poses in cluttered and partially occluded scenes. |
| S3-92 | Merge registered point-cloud scans | Rigid transforms, frame reconciliation and overlap-aware sampling | Planned | Merge multiple views in a named frame and verify unit conversion, per-scan provenance, attribute consistency and duplicate suppression. |
| S3-93 | Fuse registered color into a TSDF | Volume ColorTSDF, integrateColor and fetchPointsNormalsColors | Planned | Recover a known colored surface from posed RGB-D frames and verify color/depth alignment, confidence, and memory limits. |
| S3-94 | Raycast a fused volume into a new camera view | Volume.raycastEx and raycastExColor | Partial | 3d.tsdf_fuse raycasts the last input view. Verify an independent target pose and intrinsics with explicit depth, normal, color and validity outputs. |
| S3-95 | Bounded hashed and growing TSDF volumes | Volume HashTSDF, setEnableGrowth and visible-block diagnostics | Planned | Reconstruct a longer sequence that exceeds a fixed volume, cap growth, and verify geometry and eviction or rejection behavior. |
| S3-96 | Binary and organized PCD interchange | PCD binary layouts, dimensions and point fields | Partial | input.cloud and output.cloud support ASCII PCD. Verify binary layouts, declared organization, and invalid samples with external round trips. |
| S3-97 | Generate a surface mesh from an organized depth grid | Adjacent depth samples and discontinuity-aware triangulation | Planned | Build triangles for known depth surfaces while leaving holes at depth jumps, preserving camera provenance and valid normals. |
| S3-98 | Oriented cloud extents and physical pose | Principal axes or fitted oriented box on XYZ points | Partial | 3d.cloud_measure reports bounds and PCA axes. Verify rotated object dimensions, ambiguous orientations, and a frame-aware pose. |
| S3-99 | Inspection attributes on imported point clouds | Typed per-point intensity, confidence and source-index payloads | Planned | Preserve declared attributes through import, filtering, sampling, merging and export. Verify missing attributes and original scan indices. |

### Scan replay, cloud metrology and mesh inspection

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-100 | Find multiple planes on an organized depth grid | findPlanes with a metric point grid and organized normals | Planned | Recover two or more known planes, return coefficients and per-pixel labels, and preserve invalid depth and unmatched foreground. |
| S3-101 | Estimate a metric transform from matched 3D points | estimateAffine3D and estimateTranslation3D with explicit model choice | Planned | Recover a transform from matched cloud landmarks with outliers. Distinguish rigid, similarity and affine records, enforce units and reject degenerate correspondences. |
| S3-102 | Project registered image texture onto a mesh | Camera projection, visibility and UV mapping | Planned | Transfer colors from a known calibrated view and verify texel positions, occlusion, invalid support and retained mesh geometry. |
| S3-103 | Bounded tiled processing for large inspection clouds | Spatial tiling with query halos and streaming reduction | Planned | Process a recorded or synthetic scan beyond the single-cloud limit. Verify cross-tile neighborhood equivalence, output order, preserved IDs, and peak memory. |
| S3-104 | Replay recorded point-cloud sequences | File-backed frame groups and scan timestamp provenance | Partial | input.cloud loads an individual scan. Replay an ordered manifest with explicit frames, units, timestamps and bounds, and verify malformed or missing scans and stop behavior. |
| S3-105 | Virtual point-cloud sections and profiles | Plane-aligned slab selection and metric profile sampling | Planned | Extract sections at known locations and report distances, widths and coverage without losing point or frame provenance. |
| S3-106 | Surface flatness and height against a datum | Plane residuals and signed metric deviation statistics | Planned | Evaluate a known planar reference and several raised or warped scans. Emit deviation and coverage maps with documented datum and tolerance semantics. |
| S3-107 | Triangle mesh validity and measurement | Topology checks and geometric reductions | Planned | Report bad indices, nonmanifold edges, holes, area and volume on known fixtures. Emit volume only where topology makes it meaningful. |
| S3-108 | Mesh simplification for bounded review | Qualified decimation with error and attribute checks | Planned | Reduce a dense scan mesh to a requested face budget and verify geometric deviation, preserved boundaries and attribute integrity. |
| S3-109 | Refine imperfect calibration-target geometry | calibrateCameraROExtended | Planned | Fit known small target-coordinate deviations and verify held-out reprojection, scale anchors and reported target uncertainty. |
| S3-110 | Marker-aware and oversized checkerboard detection | findChessboardCornersSBWithMeta | Partial | calibration.grid detects checkerboards. Verify native metadata for a partially visible oversized board and recover stable target ordering and origin. |
| S3-111 | Reproducible perturbed cloud fixtures | ppf_match_3d.addNoisePC and getRandomPose or a seeded equivalent | Planned | Generate clouds with known pose and noise and report ground truth. Verify seed isolation and use the fixtures for sampling, registration and tolerance tests. |
| S3-112 | Organize a scanner cloud into a range image | Calibrated angular projection and range conflict resolution | Planned | Map a known rotating or angular scan onto a declared grid, retaining intensity, source indices, invalid bins and multiple-return handling. |

### Additional multi-camera and point-cloud geometry

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-113 | Register cameras with non-overlapping views | registerCameras and registerCamerasExtended | Planned | Recover the transform from two fixed cameras seeing different known parts of one calibration target. Verify mixed pinhole/fisheye models, per-view residuals and held-out frame closure. |
| S3-114 | Rectify three collinear cameras | rectify3Collinear or a validated equivalent | Planned | Rectify a measured three-camera rig to common epipolar rows and verify depth across both baselines. The pinned Python wheel lacks this direct binding. |
| S3-115 | Cylindrical surface and bore measurement from a point cloud | Robust 3D cylinder fit with radial residuals | Planned | Recover axis and diameter on known outer and inner surfaces with partial coverage and outliers. Report usable angular and axial support, units and degenerate fits. Use a qualified solver because OpenCV's native sample-consensus models cover planes and spheres. |

### Projector geometry, camera profiles and fiducials

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-116 | Inverse rectification for projector patterns | initInverseRectificationMap and remap | Planned | Warp a known pattern into a calibrated projector grid. Verify forward and inverse correspondence, visible support and that projected features land on the expected target positions. |
| S3-117 | Inspect a camera projection matrix | decomposeProjectionMatrix and calibrationMatrixValues | Planned | Recover intrinsics, camera center and orientation from a known camera projection matrix. Report field of view, principal point and focal length with explicit pixel or sensor units, and verify reprojection and degenerate input rejection. |
| S3-118 | Recover planar camera motion from image pairs | decomposeHomographyMat and filterHomographyDecompByVisibleRefpoints | Planned | Use a calibrated image pair and planar matches to return valid motion and plane hypotheses. Reject negative-depth candidates and preserve translation scale ambiguity unless a metric anchor is provided. |
| S3-119 | Create and qualify custom ArUco dictionaries | aruco.extendDictionary and Dictionary | Planned | Persist a seeded dictionary with declared bit size, ID ordering and Hamming separation. Verify save/load and render/detect round trips and the configured bit error tolerance. |
| S3-120 | Recover partially visible ArUco boards | aruco.ArucoDetector.refineDetectedMarkers and Board.matchImagePoints | Planned | Recover valid rejected candidates on an occluded GridBoard, preserve per-ID correspondences and compare board pose against known geometry without accepting markers from another board. |

### External disparity and deforming surfaces

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-121 | Filter independently supplied stereo disparity | ximgproc.createDisparityWLSFilterGeneric with optional paired disparity | Partial | 3d.disparity filters the disparity it computes. Accept an external disparity with explicit fixed-point scale and invalid samples. Verify guided edge preservation, occlusion support and confidence where a right disparity is available. |
| S3-122 | Diagnose binary stereo correspondence costs | Census variants and Hamming cost or a tested equivalent | Planned | Compare correspondence costs across disparities at known textured, flat and occluded pixels. Report ambiguity and verify the same rectified grid, descriptor window and validity support. The pinned wheel lacks direct Census bindings. |
| S3-123 | Reconstruct nonrigid surfaces from depth sequences | dynafu.DynaFu or a qualified equivalent | Planned | Recover a known deforming surface from recorded depth, report reconstruction error and tracking loss, and verify reset and cloud provenance. The pinned wheel exposes the class but its constructor rejects this build configuration. |

### Wide-angle and external reconstructions

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-124 | Reconstruct metric surfaces from wide-angle stereo | omnidir.stereoReconstruct | Planned | Use a calibrated wide-angle pair with known depth. Verify projection mode, disparity scale and invalid support, and compare colored XYZ output and image provenance against the reference. |
| S3-125 | Import an external sparse reconstruction | sfm.importReconstruction or a validated Bundler reader | Planned | Load known camera intrinsics, extrinsics and sparse points together. Verify camera-to-world convention, scale status and held-out reprojection, and reject unsupported formats explicitly. |
| S3-126 | Generate and detect random-pattern calibration targets | randpattern.RandomPatternGenerator and RandomPatternCornerFinder or a tested equivalent | Planned | Render a repeatable target with known physical size, recover labeled correspondences from a cropped or tilted view, and verify their target coordinates and held-out reprojection. |

### Calibration evidence and spatial measurements

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-127 | Assess calibration capture coverage and parameter stability | calibrateCameraExtended diagnostics and target observation geometry | Partial | calibration.lens reports per-view error and parameter deviations and rejects duplicate views. Assess image coverage and target-pose diversity across a capture set. Flag weak geometry and compare fitted parameters against repeat captures and held-out image points. |
| S3-128 | Estimate uncertainty on calibrated image measurements | projectPoints Jacobians and covariance propagation or measured perturbation trials | Planned | Propagate declared pixel, pose and calibration uncertainty into physical dimensions or target positions. Preserve units and parameter correlations, report modeling assumptions, and compare predicted intervals with repeated synthetic and recorded observations. |
| S3-129 | Fit a spatial line or axis to measured 3D points | fitLine on XYZ with a robust distance choice | Planned | Fit a known axis amid noise and outliers, report direction, anchor, residuals and retained point IDs, and preserve physical units and frame. Verify direction-sign ambiguity and reject coincident or otherwise underconstrained samples. |

### Lens-aware object and relative pose

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-130 | Estimate object pose with wide-angle lens models | fisheye.solvePnP, fisheye.solvePnPRansac and qualified normalized-ray fallback | Partial | 3d.pnp accepts pinhole cameras while 3d.project and 3d.undistort_points understand fisheye and omnidirectional models. Qualify lens-aware pose and outlier rejection against known targets at image centers and edges. Reproject with the source lens model and verify pixel residuals, solver criteria, units and frames. Declare the supported ray domain for an omnidirectional fallback. |
| S3-131 | Recover relative pose from separately calibrated views | findEssentialMat/recoverPose with per-camera intrinsics and distortion, or lens-normalized correspondences | Partial | 3d.essential_pose uses a shared pinhole camera. Accept independent camera models and image grids, and verify unequal focal lengths, distortion, a mixed-lens pair, outliers, cheirality and baseline direction. Keep translation scale unknown without a metric anchor. |

## Stage 4 — Selected contrib functions for images and video

[OpenCV extra modules](https://docs.opencv.org/5.0/) ·
[Contrib source](https://github.com/opencv/opencv_contrib)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-01 | Guided filtering | ximgproc.guidedFilter | Complete | cv.guided_filter. Verify edge preservation with aligned guidance and selected writes. |
| S4-02 | Locally adaptive threshold formulas | ximgproc.niBlackThreshold | Complete | cv.local_threshold. Compare Niblack, Sauvola, Wolf and NICK on uneven illumination. |
| S4-03 | SLIC and SLICO superpixels | ximgproc.createSuperpixelSLIC | Complete | cv.slic_superpixels. Verify compactness and boundary labels for the SLIC and SLICO variants. |
| S4-04 | Graph-based image segmentation | ximgproc.segmentation.createGraphSegmentation | Complete | cv.graph_segments. Verify integer label boundaries on a known region fixture. |
| S4-05 | Native ridge response | ximgproc.RidgeDetectionFilter_create | Complete | cv.ridge_response. Dark and bright narrow structures on a known fixture. |
| S4-06 | Binary thinning | ximgproc.thinning | Complete | inspect.thin_mask. Zhang–Suen and Guo–Hall centerline masks. |
| S4-07 | Structural similarity | quality.QualitySSIM_compute | Complete | inspect.image_quality. Compare aligned images with a similarity map and selected reductions. |
| S4-08 | Gradient-based image quality | quality.QualityGMSD_compute | Planned | Compare spatial degradation of aligned images, emit the native map and verify score ordering with blur and noise. |
| S4-09 | Binary perceptual hashes | img_hash.AverageHash/PHash/BlockMeanHash/MarrHildrethHash | Complete | cv.image_hash, inspect.compare_image_hashes. Exact and near-duplicate comparison with declared hash type. |
| S4-10 | Color and radial perceptual hashes | img_hash.ColorMomentHash, RadialVarianceHash | Planned | Preserve native descriptor encoding and verify distance for color moments and similarity for radial variance. Check score direction and thresholds on rotated, resampled and changed fixtures. |
| S4-11 | Fit a color correction model | ccm.ColorCorrectionModel fit | Complete | calibration.fit_ccm. Fit a correction on declared chart samples and report residuals. |
| S4-12 | Gray-world white balance | xphoto.createGrayworldWB | Planned | Verify neutral patch recovery and saturation handling. |
| S4-13 | Fast global edge-preserving smoothing | ximgproc.createFastGlobalSmootherFilter | Planned | Remove small intensity variations while preserving measured edges on a selected image. |
| S4-14 | Joint bilateral filtering | ximgproc.jointBilateralFilter | Planned | Verify image detail transfer using an aligned guide and typed image output. |
| S4-15 | Rolling guidance filtering | ximgproc.rollingGuidanceFilter | Planned | Compare removal of small texture against preservation of larger part boundaries. |
| S4-16 | Extended inpainting for defect cleanup | xphoto.inpaint | Planned | Fill a masked defect using the supported algorithm and verify known background recovery. |
| S4-17 | DCT denoising | xphoto.dctDenoising | Planned | Quantify noise reduction and edge retention on a seeded color or grayscale fixture. |
| S4-18 | Fast line detection | ximgproc.createFastLineDetector | Planned | Verify segment positions and orientation on straight edges and retain collection metadata. |
| S4-19 | Compact binary descriptors | xfeatures2d.BriefDescriptorExtractor, FREAK | Planned | Compute descriptors at wired image keypoints and check Hamming matching across a registered image pair. |
| S4-20 | Dense local descriptors | xfeatures2d.DAISY | Planned | Sample image descriptors on a controlled grid and verify matching under known displacement. |
| S4-21 | GSoC adaptive foreground | bgsegm.createBackgroundSubtractorGSOC | Complete | sequence.gsoc_background. Check warmup and detection on an empty-conveyor replay. |
| S4-22 | Dual TV-L1 flow | optflow.createOptFlow_DualTVL1 | Planned | Report flow and confidence on a known moving patch and compare with the existing flow baseline. |
| S4-23 | DeepFlow | optflow.createOptFlow_DeepFlow | Planned | Verify displacement and occlusion behavior on a short recorded or synthetic sequence. |
| S4-24 | Dense RLOF | optflow.createOptFlow_DenseRLOF | Planned | Verify displacement, support mode and forward/backward validity on a translated texture. |
| S4-25 | WeChat QR decoding | wechat_qrcode.WeChatQRCode | Planned | Compare small and low-contrast labels with the native path. Verify multiple symbols and pin model assets. |
| S4-26 | HOG image descriptors | HOGDescriptor.compute | Complete | inspect.hog. Describe a configured image window and visualize cell orientations. |

### Filtering, descriptors and shape extensions

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-27 | Domain-transform filtering | ximgproc.dtFilter | Complete | cv.domain_transform_filter. Verify the supported modes and guidance image alignment. |
| S4-28 | Adaptive MSLIC superpixels | ximgproc.createSuperpixelSLIC with MSLIC | Complete | cv.slic_superpixels. Verify adaptive labels and reject unsupported single-seed configurations before native calls. |
| S4-29 | LSC superpixels | ximgproc.createSuperpixelLSC | Complete | cv.lsc_superpixels. Report boundary recall against a known outline. |
| S4-30 | SEEDS superpixels | ximgproc.createSuperpixelSEEDS | Complete | cv.seeds_superpixels. Verify requested granularity, iteration controls and label recovery. |
| S4-31 | Selective Search object proposals | ximgproc.segmentation.createSelectiveSearchSegmentation | Complete | inspect.selective_search. Compare proposal modes with known image regions. |
| S4-32 | BRISQUE no-reference quality | quality.QualityBRISQUE_create | Complete | inspect.brisque_quality. Compare sharp, blurred and compressed versions of the same fixture with pinned model assets. |
| S4-33 | LSBP adaptive foreground | bgsegm.createBackgroundSubtractorLSBP | Complete | sequence.lsbp_background. Verify learning, warmup and motion detection on replay. |
| S4-34 | Detect color charts in a camera image | mcc.CCheckerDetector | Complete | calibration.detect_color_chart. Detect the target and return its ordered patch locations. |
| S4-35 | Apply a fitted color correction model | ccm.ColorCorrectionModel inference | Complete | calibration.apply_ccm. Verify source encodings and corrected chart samples with a reusable model. |
| S4-36 | Fast bilateral solver | ximgproc.createFastBilateralSolverFilter | Planned | Refine a sparse image signal with confidence weights and guide edges. Verify wheel dependencies and compare against a known target. |
| S4-37 | Adaptive manifold filtering | ximgproc.amFilter | Planned | Compare edge preservation and selected-region behavior with the bilateral baseline. |
| S4-38 | Weighted median filtering | ximgproc.weightedMedianFilter | Planned | Remove impulse outliers with an aligned guide and verify edge localization. |
| S4-39 | Anisotropic diffusion | ximgproc.anisotropicDiffusion | Planned | Preserve an intensity boundary across configured iterations while reducing seeded noise. |
| S4-40 | L0 image smoothing | ximgproc.l0Smooth | Planned | Reduce low-amplitude texture while retaining larger boundaries, with stable output depth and range. |
| S4-41 | Extended gradient operators | ximgproc GradientDeriche and GradientPaillou families | Planned | Verify signed axis response and localization on a step-edge fixture. |
| S4-42 | Contour ellipse detection | ximgproc.findEllipses | Planned | Detect multiple rotated ellipses in clutter with scores and oriented image geometry. |
| S4-43 | Fast Hough line transform | ximgproc.FastHoughTransform | Planned | Verify transformed bin geometry and recover lines with known angle and offset. |
| S4-44 | Structured edge detection | ximgproc.createStructuredEdgeDetection | Planned | Use a revisioned local model and compare detected boundaries with marked reference edges. |
| S4-45 | EdgeBoxes object proposals | ximgproc.createEdgeBoxes | Planned | Use oriented edge maps to rank region proposals and verify overlap against known objects. |
| S4-46 | Line feature descriptors and matching | line_descriptor.BinaryDescriptor and BinaryDescriptorMatcher | Planned | Match line features across a known camera motion and retain segment endpoints and native distances. |
| S4-47 | Grid motion statistics match filtering | xfeatures2d.matchGMS | Planned | Filter outlier image matches before pose fitting and compare against unfiltered geometry. |
| S4-48 | Additional binary patch descriptors | xfeatures2d LATCH, BEBLID and TEBLID | Planned | Compare repeatability and descriptor distance against a known transformed image pair. |
| S4-49 | Scale-aware interest points | xfeatures2d.HarrisLaplaceFeatureDetector | Planned | Locate the same corners through a scale change and preserve size and response. |
| S4-50 | Shape context contour comparison | createShapeContextDistanceExtractor | Planned | Match similar contours despite small deformation and report a distance with known variants. |
| S4-51 | Hausdorff contour distance | createHausdorffDistanceExtractor | Planned | Verify configured rank, translation sensitivity and comparison against known offsets. |
| S4-52 | Native shape-based nonrigid registration | createThinPlateSplineShapeTransformer | Planned | Fit paired image contours, warp a reference image and verify interpolated landmark error. |

### Saliency, motion, photo and text extensions

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-53 | Static saliency image maps | saliency.StaticSaliencySpectralResidual and StaticSaliencyFineGrained | Planned | Compare ranked foreground prominence on controlled images and emit a selected mask. |
| S4-54 | Motion saliency maps | saliency.MotionSaliencyBinWangApr2014 and bioinspired transient-area segmentation | Planned | Verify warmup, reset and localization on moving parts and transient changes in a short clip. |
| S4-55 | Motion history and time surfaces | motempl.updateMotionHistory, calcMotionGradient, segmentMotion | Planned | Recover motion age and dominant direction with explicit frame timestamps and reset behavior. |
| S4-56 | Additional adaptive foreground models | bgsegm MOG, GMG and CNT | Planned | Compare warmup and foreground response using the existing empty-conveyor fixture. |
| S4-57 | Sparse-to-dense flow interpolation | optflow.calcOpticalFlowSparseToDense | Planned | Verify displacement, sparse sampling controls and behavior at motion boundaries. |
| S4-58 | PCAFlow and SimpleFlow | optflow.createOptFlow_PCAFlow and createOptFlow_SimpleFlow | Planned | Compare each estimator on a known displacement and preserve vector units. |
| S4-59 | Edge-aware interpolation of image matches | ximgproc.EdgeAwareInterpolator | Planned | Turn sparse matches into a dense field and verify preservation of known motion edges. |
| S4-60 | Image matting from a trimap | alphamat.infoFlow or equivalent | Planned | The pinned Python wheel has no direct binding. Qualify another backend or tested equivalent and recover a soft alpha channel with known opaque, transparent and mixed regions. |
| S4-61 | Retina-inspired contrast and motion preprocessing | bioinspired.Retina | Planned | Verify image enhancement and motion output on a short exposure-change sequence with reset. |
| S4-62 | Block-matching 3D denoising of images | xphoto.bm3dDenoising | Planned | Measure denoising and edge retention on seeded noise. Verify that the target build enables the algorithm. |
| S4-63 | Additional HDR tone mapping | xphoto.createTonemapDurand | Planned | Compare local contrast against baseline tone mapping. Verify that the target build enables the algorithm. |
| S4-64 | Frequency-selective detail smoothing | xphoto.oilPainting | Planned | Verify supported encodings and degree of texture suppression on a segmented surface fixture. |
| S4-65 | Percentile white balance | xphoto.createSimpleWB | Planned | Verify independent channel stretch and clipping behavior with known channel extrema. |
| S4-66 | Learned white balance | xphoto.createLearningBasedWB | Planned | Verify color correction on independent illuminant fixtures and record the model revision. |
| S4-67 | Native text component detection | text.detectRegions and supported extremal-region grouping | Planned | Extract candidate regions from contrast-varied printed text and verify grouping against marked words. |
| S4-68 | OCR through the contrib Tesseract adapter | text.OCRTesseract | Planned | Qualify a Tesseract-enabled backend and pinned language data. Verify recognition, confidence and source boxes on labeled crops. The pinned wheel reported Tesseract unavailable and returned empty text on a known label. |
| S4-69 | Color-independent global image normalization | ximgproc.PeiLinNormalization | Planned | Return the normalization transform and verify alignment of equivalent shape images. |
| S4-70 | Fuzzy image smoothing and inpainting | ft.filter and ft.inpaint | Planned | Verify local smoothing or fill quality on an image fixture, with bounded controls. |

### Additional upstream image and video families

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-71 | Hierarchical feature-selection segmentation | hfs.HfsSegment | Planned | Compare boundaries and region IDs against a known segmented image with CPU execution. |
| S4-72 | Mapper-based multiscale image registration | reg MapperGrad variants, MapperPyramid and Map | Planned | Recover a declared shift, Euclidean, similarity, affine or projective mapping from paired images. Verify seeded refinement, source-frame maps and inverse round trips on valid coverage. |
| S4-73 | Low-light enhancement | intensity_transform.BIMEF and BIMEF2 | Planned | Recover known shadow contrast with automatic and explicit exposure settings. Quantify noise and clipping against the baseline. |
| S4-74 | Boosted binary image descriptors | xfeatures2d.BoostDesc | Planned | Compare matching under illumination or viewpoint changes on a known registered pair. |
| S4-75 | Contrib surface texture descriptors | xfeatures2d VGG or other supported local descriptors | Planned | Verify texture correspondence on pinned local descriptor assets and a known view pair. |
| S4-76 | Background sequences for controlled tracking validation | bgsegm.createSyntheticSequenceGenerator | Planned | Emit synthetic moving-object frames with the native foreground reference masks and reproducible ticket order. |
| S4-77 | Contrib optical flow interpolation with robust correspondence | ximgproc.RICInterpolator | Planned | Interpolate a known sparse field and validate deformation boundaries and unsupported regions. |
| S4-78 | Variance-adaptive edge-preserving smoothing | ximgproc.edgePreservingFilter | Planned | Reduce noise while preserving the position and strength of a measured image edge. |
| S4-79 | Edge Drawing segments and primitives | ximgproc.createEdgeDrawing | Planned | Verify ordered edge segments and fitted line or ellipse outputs against known geometry. |
| S4-80 | Tree-based image regions | xfeatures2d.TBMR | Planned | Verify keypoint location and scale through a known image transform and compare repeatability with MSER. |
| S4-81 | Classical multi-frame video super-resolution | superres BTV-L1 or equivalent reconstruction | Planned | Qualify a native build or equivalent because the pinned wheel has no direct superres bindings. Verify temporal window, alignment, moving-detail recovery and artifacts against per-frame interpolation. |

### Contour spectra and sparse-mask operations

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-82 | Fourier contour descriptors and alignment | ximgproc.contourSampling, fourierDescriptor and createContourFitting | Planned | Verify periodic contour sampling, transform recovery, and shape matching under known rotation and scale. |
| S4-83 | Run-length encoded binary morphology | ximgproc run-length mask encoding and morphology | Planned | Verify erosion and dilation parity with dense masks and measure memory on sparse large tiles. Check Python support or qualify an equivalent implementation. |

### Additional sparse motion

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-84 | Sparse RLOF correspondence | optflow.calcOpticalFlowSparseRLOF | Planned | Track wired image points with configurable rejection and compare displacement, status and error on known motion. |

### Color templates, texture and spatial verification

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-85 | Joint color template correlation | ximgproc.colorMatchTemplate | Planned | Use a color template with equal-luminance distractors. Verify score peaks, source coordinates and typed color handling, and compare localization with the existing template matcher. |
| S4-86 | Multiscale local-contrast edge maps | ximgproc.BrightEdges or a tested equivalent | Planned | Measure edge localization and continuity on low-contrast parts over an uneven background. Verify range and threshold controls and preserve the selected region. The pinned wheel lacks a direct binding. |
| S4-87 | Angular projection profiles for inspection | ximgproc.RadonTransform | Planned | Verify angle and distance axes on synthetic stripes and tilted edges, retain projection precision for scoring, and map the strongest directions back to the inspected region. |
| S4-88 | Smooth texture while retaining structural edges | ximgproc.bilateralTextureFilter | Planned | Use a known textured surface with a measured boundary. Verify texture attenuation, edge displacement, selected-region writes and parameter limits. |
| S4-89 | Scan-based superpixel segmentation | ximgproc.createScanSegment | Planned | Measure target region count, boundary recall and runtime on a representative image. Verify integer labels, stable image coordinates and label-mask output. |
| S4-90 | Spatial texture covariance from image patches | ximgproc.covarianceEstimation | Planned | Verify window dimensions, sample support and covariance against a direct patch reference. Compare normal and defective texture while bounding output memory. |
| S4-91 | Distribution-based image texture signatures | xfeatures2d.PCTSignatures and PCTSignaturesSQFD | Planned | Generate seeded signatures and compare known-good and defective patches. Verify ranking, repeatability and persistence of the sampling and clustering settings. |
| S4-92 | Spatial verification for visual-word matches | xfeatures2d.matchLOGOS with a pinned visual vocabulary | Planned | Match repeated parts against a trained vocabulary. Verify that spatial support suppresses look-alike outliers, and preserve image positions and correspondence IDs. |

### Text, appearance, image retrieval and object proposals

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-93 | Detect text by stroke width | text.detectTextSWT | Planned | Detect marked text regions on light and dark backgrounds with controlled stroke widths. Verify component and grouped-box outputs, polarity, selected-region coordinates, and rejection of non-text strokes. |
| S4-94 | Recognize text with native HMM or beam search | text.OCRHMMDecoder and text.OCRBeamSearchDecoder with compatible classifiers | Planned | Pin classifiers, alphabet and transition data. Verify ambiguous characters and out-of-vocabulary text on labeled crops, and preserve per-read confidence, source boxes and method provenance. |
| S4-95 | Use image appearance to distinguish similar contours | ShapeContextDistanceExtractor.setImages and appearance weighting | Planned | Rank otherwise similar outlines that have different markings or texture. Verify contour and image coordinate alignment, configured appearance and geometry weights, and ranking on known deformation. |
| S4-96 | Search a saved image hash catalog | Versioned perceptual hashes and method-specific ranking | Partial | cv.image_hash and inspect.compare_image_hashes compute and compare binary hashes. Build and reload a reference catalog with stable image IDs. Verify near-duplicate ranking, score direction, settings compatibility and rejection thresholds for each supported hash family. |
| S4-97 | Propose image regions using objectness scores | saliency.ObjectnessBING and pinned training assets | Planned | Return ranked source-grid boxes with their native scores. Verify proposal recall, score order and clipping on cluttered images with known parts, and diagnose missing or incompatible model assets. |

### Local contrast, line registration and restoration support

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-98 | Preserve local contrast in unevenly lit images | bioinspired.RetinaFastToneMapping or Retina.applyFastToneMapping | Planned | Enhance mixed bright and shadowed regions while preserving fine inspection features. Compare local contrast, clipping and false defect response against the original image. |
| S4-99 | Register low-texture views from line correspondences | line_descriptor matches and a robust line-based image transform fit | Planned | Recover a source-to-reference transform from annotated lines with partial segments and false matches. Verify transferred geometry on independent marks and reject parallel or otherwise degenerate configurations. |
| S4-100 | Recover masked image samples with explicit restoration support | ft.FT02D_iteration and its output support mask | Planned | Restore holes on a known image and report which samples each iteration can support. Verify convergence or bounded stop, unchanged measured samples, and unsupported regions when valid neighbors are insufficient. |

### Reference spaces and color profile verification

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-101 | Fit camera color correction against measured reference spaces | ccm.ColorCorrectionModel custom Lab or wide-gamut RGB references | Partial | calibration.fit_ccm accepts sRGB and linear sRGB references. Qualify measured Lab and wide-gamut RGB with explicit numeric encoding, illuminant or white point, patch ordering, clipping and held-out correction error. |
| S4-102 | Verify a color correction profile on independent captures | ColorCorrectionModel.correctImage, detected chart patches and per-patch Delta E | Partial | calibration.detect_color_chart, calibration.fit_ccm and calibration.apply_ccm provide chart samples and correction. Test a saved profile on independently captured patches, compare corrected and uncorrected error, and report per-patch exclusions and pass/fail thresholds tied to the profile revision. |

## Module scope boundaries

The [scope map](opencv-scope-map.md) details the owning module families.
The following groups remain outside the percentage calculation. This makes it
possible to review omissions without counting overloads, infrastructure and
platform variants as separate image features.

| Group | Where it belongs |
| --- | --- |
| Core scalar and matrix utilities, generic ML, generic graph algorithms | Internal primitives or user tools. Image-specific transforms, image features, and camera-derived geometry are tracked in stages 1–3. |
| HighGUI, interactive visualization and OpenCV hardware abstraction | The application UI and backend roadmap own display, input handling, and device abstraction. |
| G-API, CUDA/OpenCL, platform codecs and alternate accelerators | Backend, portability and performance qualification. The primary image capability is counted once. |
| Domain-specific or device-only extra modules | Keep specialized device SDKs, bio/face-specific application packages and C++-only infrastructure outside the baseline. Add an image workflow with a defined fixture before changing the scope. |
| Non-OpenCV image libraries | Count explicit workflow adapters such as LightGlue, Annoy and ZXing in their owning stages. Separate scikit-image tools stay in the toolbox and are outside this OpenCV roadmap. |

Stage 5 job packages and the parallel editor, acquisition and production milestones
have their own acceptance criteria in the [roadmap](roadmap.md).
