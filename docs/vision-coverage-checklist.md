# Image and video coverage checklist

This is the implementation scope for stages 1–4 of the [roadmap](roadmap.md).
The [coverage backlog](opencv-coverage.md) keeps broader candidates and background.
The native API column names the main functions or building blocks for each job.
Each row represents one independently verifiable capability, even when several
native calls or graph blocks implement it.

<!-- coverage-summary:start -->

| Stage | Complete | Partial | Planned | Total | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. Core and baseline 2D | 61 | 6 | 2 | 69 | 88.4% |
| 2. ML and DNN | 7 | 4 | 12 | 23 | 30.4% |
| 3. 3D from camera imagery | 21 | 8 | 4 | 33 | 63.6% |
| 4. Selected contrib | 11 | 0 | 15 | 26 | 42.3% |
| **Overall** | 100 | 18 | 33 | 151 | 66.2% |

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
  depth or surfaces reconstructed from camera imagery. General matrix
  operations, generic table ML, GUI/runtime infrastructure, PLC and robot
  transport, hardware qualification, and application-specific job packages
  are outside this denominator.
- Assign overlaps to one stage. Stage 2 owns learned vision and image-feature
  models, stage 3 owns camera geometry and depth, and stage 4 owns the remaining
  selected extra modules. For build-dependent rows, confirm the target Python
  binding and verify the supported fallback or backend before marking complete.
- The [raw Python binding inventory](opencv-bindings.tsv) helps discover APIs
  and aliases. This checklist defines the denominator for progress reporting.

Run `python scripts/vision_coverage.py` to calculate progress. Use `--update`
after editing statuses to refresh the table, and `--check` to check the saved
table. The [block parameter reference](block-parameters.md) lists current ports
and configuration. For planned or partial rows, the last column gives the
fixture or result needed to close the item.

## Stage 1 — OpenCV core and baseline 2D

[Stage tracker](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) ·
[Sequence and measurement follow-up](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S1-01 | Tiled inspection mosaic | Registered tile windows, warpPerspective, remap, seam and coverage masks | Partial | cv.stitch_scan_sequence handles up to eight views in a bounded canvas. Render a scan beyond those limits as bounded tiles, verify overlap and source transforms, and compare against the existing renderer on a shared fixture. |
| S1-02 | Continuous centered stabilization | calcOpticalFlowPyrLK, estimateAffinePartial2D, warpAffine, bounded look-ahead | Partial | sequence.stabilize_batch smooths up to 64 frames. Stream a longer clip with bounded memory and latency, preserve frame timestamps and coverage, and verify reset and end-of-stream flush. |
| S1-03 | Local still-image replay and save | imread, imwrite | Complete | input.file, output.image. Read typed image files, replay a list or folder with filename tickets, and save the result. |
| S1-04 | Image encode/decode in memory | imencode, imdecode | Complete | cv.encode_image, cv.decode_image. PNG/JPEG buffer round trips with compression settings and spatial provenance. |
| S1-05 | Multipage TIFF replay and save | imreadmulti, imwritemulti | Complete | input.tiff_file, output.tiff_file. Ordered, ticketed page ranges and lossless batch output. |
| S1-06 | JPEG/PNG metadata and orientation read | imreadWithMetadata, EXIF orientation transforms | Complete | input.file. Report metadata and exercise the saved-image orientation fixture. |
| S1-07 | Image metadata write | imwriteWithMetadata, imencodeWithMetadata | Planned | Round-trip supported metadata types on JPEG/PNG. Verify output orientation semantics and reject unsupported chunks explicitly. |
| S1-08 | Raw sensor color decoding | demosaicing, cvtColor | Complete | cv.demosaic_bayer, input.raw_yuv, cv.decode_yuv. Bayer phase at a crop and packed YUV layouts. |
| S1-09 | Camera image capture | VideoCapture.open, read, release | Complete | input.camera. Graph camera frame contract with backend and format settings. Physical backend qualification is tracked under acquisition. |
| S1-10 | Video file replay and recording | VideoCapture, VideoWriter | Complete | input.video_file, output.video. Frame bounds, stride, stop/loop, frame timestamps, and a local round trip. |
| S1-11 | Timestamp-paced file video | VideoCapture timestamps and replay clock | Partial | input.video_file exposes media timestamps. Verify paced playback with a fake clock, variable gaps, stride, and prompt Stop/Resume. |
| S1-12 | Network video input | VideoCapture with a supported network backend | Planned | Replay a local test stream and verify reconnect, bounded read failure, and clean stop. |
| S1-13 | Color conversion and pixel depth | cvtColor, convertTo, convertScaleAbs | Complete | cv.gray, cv.convert, cv.scale_absolute. Grayscale, color-space/depth conversion, and signed magnitude display. |
| S1-14 | Channel split, merge and mixing | split, merge, mixChannels, extractChannel, insertChannel, transform | Complete | cv.split_channels, cv.merge_alpha_channels, cv.route_channels, cv.channel_transform. Preserve channel order and applicable pixel depth. |
| S1-15 | Image arithmetic and comparison | add, subtract, multiply, divide, absdiff, compare, inRange | Complete | cv.image_arithmetic, cv.absdiff, cv.compare_pixels, cv.in_range. Test mixed depths, zero divisors, masks, and integer labels. |
| S1-16 | Image masking, bitwise operations and blending | bitwise_and/or/xor/not, addWeighted, blendLinear | Complete | cv.apply_mask, cv.bitwise, cv.blend, cv.blend_weight_maps. Selected processing and per-pixel weights. |
| S1-17 | Intensity transforms and normalization | normalize, LUT, log, exp, pow | Complete | cv.normalize, cv.intensity_lut, cv.intensity_transform, cv.gamma. Masked transforms and channel-wise 8/16-bit LUTs. |
| S1-18 | Image statistics and nonfinite samples | meanStdDev, minMaxLoc, norm, findNonZero, checkRange-style validation | Complete | cv.image_statistics, cv.image_norm, cv.find_nonzero_pixels, inspect.image_validity, cv.replace_nonfinite. Report extrema, selected values, invalid pixels and replacements. |
| S1-19 | Histogram, comparison and backprojection | calcHist, compareHist, calcBackProject | Complete | inspect.histogram, inspect.compare_histograms, inspect.histogram_backproject. Per-channel summaries and selected intensity distributions. |
| S1-20 | Local and directional statistics | integral, integral2, tilted integral, sampled reductions | Complete | cv.local_statistics, cv.tilted_statistics, cv.intensity_profile, cv.path_intensity_profile. Window and path statistics with sample counts. |
| S1-21 | Contrast equalization and scalar palettes | equalizeHist, createCLAHE, applyColorMap | Complete | cv.equalize, cv.clahe, cv.colorize_scalar. Global/local contrast and scalar previews with validity masks. |
| S1-22 | Common smoothing | GaussianBlur, boxFilter, medianBlur, bilateralFilter | Complete | cv.gaussian_blur, cv.box_filter, cv.median_blur, cv.bilateral. Native kernels, depth and border controls where supported. |
| S1-23 | Custom convolution and sharpening | filter2D, sepFilter2D, getGaborKernel, unsharp blending | Complete | cv.filter2d, cv.separable_filter, cv.unsharp, cv.gabor_bank. Editable kernels and directional responses. |
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
| S1-34 | Line and circle detection | HoughLines/P/PointSet, HoughCircles, createLineSegmentDetector | Complete | inspect.hough_polar_lines, inspect.hough_lines, inspect.hough_point_lines, inspect.hough_circles, inspect.line_segments. Geometry, strengths where native, and overlays. |
| S1-35 | Calipers and oriented edge checks | Sobel or sampled derivatives, remap | Complete | inspect.oriented_edges, inspect.bead_calipers, cv.unwrap_path_strip, inspect.strip_width. Edge direction, paired widths and curved-path width. |
| S1-36 | Image resize, crops, padding and orientation | resize, getRectSubPix, copyMakeBorder, flip, rotate, transpose | Complete | cv.resize, cv.crop, cv.crop_subpixel, cv.copy_border, cv.rotate90, cv.flip, cv.transpose_image. Coordinate round trips and interpolation controls. |
| S1-37 | Image composition | hconcat, vconcat, repeat, masked copy | Complete | cv.join_images, cv.tile_image, cv.paste_patch. Bound composition size and retain panel or placement coordinates. |
| S1-38 | Affine and projective warps | getAffineTransform, getPerspectiveTransform, warpAffine, warpPerspective, perspectiveTransform | Complete | cv.control_point_transform, cv.warp_affine, cv.warp_perspective, cv.rectify_quadrilateral, cv.transform_points. Defined transform direction and output grid. |
| S1-39 | Remapping and polar grids | remap, convertMaps, warpPolar | Complete | cv.remap, cv.pack_remap_map, cv.remap_packed, cv.warp_polar. Reusable floating/fixed maps and inverse polar mapping. |
| S1-40 | Gaussian image pyramids | pyrUp, pyrDown | Complete | cv.pyramid_up, cv.pyramid_down, cv.image_pyramid, cv.pyramid_level. Select a level and track source coordinates. |
| S1-41 | Frequency-domain image processing | dft, idft, dct, idct, mulSpectrums, createHanningWindow | Complete | cv.dft, cv.idft, cv.dct, cv.idct, cv.combine_fourier_spectra, inspect.fourier_correlation. Round trips, band masks and displacement. |
| S1-42 | Corner and feature detection | goodFeaturesToTrack, cornerHarris, cornerMinEigenVal, cornerSubPix, FAST | Complete | inspect.corners, inspect.corner_response, inspect.refine_corners, inspect.fast. Response maps, refined points and selections. |
| S1-43 | Binary and float local descriptors | ORB, SIFT, AKAZE, KAZE, BRISK | Complete | inspect.orb, inspect.sift, inspect.akaze, inspect.kaze, inspect.brisk. Descriptor data linked to pixel coordinates. |
| S1-44 | Descriptor matching and review | BFMatcher, FlannBasedMatcher, knnMatch, radiusMatch | Complete | inspect.match_sift, inspect.match_orb, inspect.match_flann, inspect.match_descriptor_radius, inspect.review_matches. Ratio, cross-check, radius and inlier review. |
| S1-45 | Template matching with multiple instances | matchTemplate, local-peak selection, subpixel refinement | Complete | inspect.match_template, inspect.template_instances. Multiple candidates with scores, locations and configurable suppression. |
| S1-46 | Trainable shape and texture matching | Contour geometry, ORB/SIFT descriptors, homography | Complete | pattern.train_shape, pattern.match_shape, pattern.train_texture, pattern.match_texture. Reusable models and image poses. |
| S1-47 | Generalized Hough shape search | createGeneralizedHoughBallard, createGeneralizedHoughGuil | Complete | inspect.hough_template, inspect.guil_template. Fixed or rotation/scale-aware search with votes. |
| S1-48 | Registered reference alignment | phaseCorrelate, findTransformECC, estimateAffine2D, estimateAffinePartial2D, findHomography | Complete | inspect.phase_correlation, inspect.phase_correlation_iterative, inspect.align_ecc, inspect.affine, inspect.homography, inspect.translation. Robust alignment with residual or quality checks. |
| S1-49 | Bounded scan stitching | Local registration, warpPerspective, seam and exposure blending | Complete | cv.stitch_image_pair, cv.stitch_scan_sequence. Overlapping pairs and ordered short scans with masks, drift correction, and exposure compensation. |
| S1-50 | Recorded and streaming motion baseline | calcOpticalFlowPyrLK, estimateAffinePartial2D, warpAffine | Complete | sequence.stabilize_camera, sequence.stabilize_batch. Causal and finite centered correction with coverage and validation. |
| S1-51 | Sparse and dense optical flow | calcOpticalFlowPyrLK, calcOpticalFlowFarneback, DISOpticalFlow | Complete | sequence.track_points, sequence.lk_grid_flow, sequence.farneback, sequence.dis. Flow and consistency checks. |
| S1-52 | Baseline background subtraction | createBackgroundSubtractorMOG2, createBackgroundSubtractorKNN, temporal median | Complete | sequence.mog2, sequence.knn_background, sequence.temporal_median. Learning and foreground masks on replay. |
| S1-53 | Seeded object and conveyor tracking | TrackerCSRT, TrackerKCF, image-space association | Complete | sequence.track_csrt, sequence.track_kcf, sequence.track_conveyor. Seeded boxes and multi-part image/encoder tracking. |
| S1-54 | Track across missed image detections | KalmanFilter and gated assignment | Partial | sequence.track_conveyor predicts with encoder motion. Add a reusable image-track predictor with confidence, bounded missed frames and crossing-part tests. |
| S1-55 | Reference-based image error | absdiff, norm, PSNR | Complete | inspect.image_quality. Selected MAE, RMSE, maximum absolute error and PSNR on aligned frames. |
| S1-56 | Illumination and color inspection | cvtColor, channel gains, flat-field/shading correction, Lab difference | Complete | calibration.color_balance, cv.flat_field, cv.adaptive_background, inspect.measure_lab, inspect.compare_lab. Selected Lab statistics and tolerance. |
| S1-57 | Inpainting and denoising | inpaint, fastNlMeansDenoising/Colored/Multi, denoise_TVL1 | Complete | cv.inpaint, cv.non_local_means, cv.temporal_nlm, cv.denoise_tvl1. Known degraded images and masked restoration. |
| S1-58 | Deblurring with a known blur kernel | DFT-based Wiener inverse | Complete | cv.wiener_deblur. Gaussian, defocus, line-motion and editable point-spread functions. |
| S1-59 | HDR exposure processing | AlignMTB, CalibrateDebevec/Robertson, MergeDebevec/Robertson/Mertens, Tonemap | Complete | cv.align_exposures, cv.calibrate_debevec, cv.merge_robertson, cv.fuse_exposures, cv.tonemap_hdr. Exposure alignment, radiance and display output. |
| S1-60 | Native QR and linear barcode decoding | QRCodeDetector, barcode.BarcodeDetector | Complete | inspect.read_qr, inspect.read_barcode. Decoded strings and image-space outlines. |
| S1-61 | Visible support for 2D pick candidates | Projected template support, masked comparisons and contour coverage | Partial | inspect.select_pick ranks score, angle and clearance. Verify a visible-support score on clear, touching, and partly covered parts. |
| S1-62 | Image and pose-relative selections | Rectangle, circle and polygon rasterization, Boolean masks | Complete | region.shape, mask.shape, region.combine, region.pose, region.fitted_shape. Apply editable and measured selections to image operations. |
| S1-63 | Object collection filtering and sorting | Contour/component measurements, stable ranking and source overlays | Complete | collection.filter, collection.sort, collection.clip_lines, collection.draw. Retain IDs through size, orientation, shape and position filters. |
| S1-64 | Stable intensity regions | MSER.detectRegions | Complete | inspect.mser. Stable grayscale and color regions as image coordinates and filterable bounds. |
| S1-65 | Planar neighborhoods and cell masks | Subdiv2D Delaunay and Voronoi | Complete | inspect.delaunay_triangulation, inspect.voronoi_cells. Recover fixture neighbor geometry and select a cell mask. |
| S1-66 | Mean-shift color smoothing | pyrMeanShiftFiltering | Complete | cv.mean_shift_filter. Flatten local color variation while retaining boundaries for segmentation. |
| S1-67 | Multi-reference color classification | Lab statistics and CIE76/CIEDE2000 distances | Complete | inspect.classify_lab. Rank configured reference colors and reject out-of-tolerance samples. |
| S1-68 | Compare reference edge maps | distanceTransform and symmetric edge tolerances | Complete | inspect.compare_edge_maps. Report missing and added edges after allowing bounded position error. |
| S1-69 | Map selections through nonlinear lens correction | remap and lens map validity | Partial | region.transform handles affine/projective mapping and calibration.undistort_lens_points maps points. Carry an entire source selection into the corrected image and verify a masked measurement through a nonlinear warp. |

## Stage 2 — ML and DNN for images and video

[Stage tracker](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) ·
[Current ML workflows](ml.md)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S2-01 | Single-image ONNX inference | dnn.readNetFromONNX, blobFromImage, Net.setInput, Net.forward | Complete | ml.dnn. Local model load, crop/resize, channel order, normalization and a reproducible CPU inference fixture. |
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
| S2-13 | Text regions and CTC recognition | Probability-map contours, perspective crop, CTC decode | Complete | ml.text_regions, ml.recognize_text. Recognize image text with rectified regions and score results. |
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

## Stage 3 — 3D from camera imagery

[Camera and robot calibration](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2) ·
[3D expansion](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6) ·
[Current 3D workflows](vision-3d.md)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S3-01 | Calibration target detection and capture checks | findChessboardCorners/SB, cornerSubPix, findCirclesGrid, ArUco/ChArUco detectors | Complete | calibration.grid, calibration.chessboard_quality, calibration.charuco, inspect.aruco_markers. Identified target observations and capture-quality checks. |
| S3-02 | Pinhole calibration and held-out validation | calibrateCameraExtended, projectPoints | Complete | calibration.lens, calibration.validate_lens. Intrinsics, distortion and per-view residuals with independent target checks. |
| S3-03 | Lens remapping and aligned image points | initUndistortRectifyMap, undistortPoints, remap | Complete | calibration.lens_maps, calibration.undistort_lens_points, calibration.undistort. Image/map size, valid coverage and points on the corrected grid. |
| S3-04 | Fit fisheye intrinsics from target images | fisheye.calibrate, fisheye.projectPoints | Partial | 3d.camera and 3d.rectify_camera consume existing fisheye intrinsics. Fit a profile from observed targets and check held-out reprojection and valid coverage. |
| S3-05 | Fit omnidirectional intrinsics | omnidir.calibrate, omnidir.projectPoints | Partial | 3d.camera and 3d.rectify_camera consume an existing model. Estimate intrinsics and xi and verify held-out edge observations. |
| S3-06 | Planar registration and physical workplane | findHomography, perspectiveTransform, warpPerspective | Complete | calibration.fit_planar, calibration.validate_planar, calibration.rectify_workplane. Independent marker checks and a uniform physical image grid. |
| S3-07 | Physical dimensions from calibrated 2D images | Planar point mapping and scaled shape fitting | Complete | calibration.inspect_planar_distance, inspect.workplane_rectangle, inspect.workplane_ellipse, inspect.strip_width. Ticketed dimensions and tolerance reasons with source-aligned graphics. |
| S3-08 | Versioned camera and rig profiles | Versioned intrinsics, map and extrinsic records tied to an image grid | Partial | input.calibration, output.calibration, input.lens_maps and output.lens_maps cover profiles and maps. Add explicit revision selection and reject mismatched capture or grid metadata. |
| S3-09 | Project camera points and rays | projectPoints, undistortPoints and lens-family variants | Complete | 3d.project, 3d.undistort_points, 3d.rectify_camera. Pixel projection and ray/rectification fixtures for declared lens families. |
| S3-10 | Estimate object pose and fiducial pose | solvePnP, solvePnPRansac, solvePnPRefineLM/VVS, Rodrigues | Complete | 3d.pnp, 3d.aruco_pose. Known poses, inlier and reprojection checks, and camera-relative frames. |
| S3-11 | Relative pose and epipolar geometry | findFundamentalMat, findEssentialMat, recoverPose, computeCorrespondEpilines | Complete | 3d.fundamental, 3d.essential_pose, 3d.check_epipolar. Reject outlier pairs and report per-view residuals. |
| S3-12 | Sparse triangulation | triangulatePoints | Complete | 3d.triangulate. Recover 3D positions with cheirality and reprojection bounds from camera correspondences. |
| S3-13 | Stereo rig calibration and rectification | stereoCalibrate, stereoRectify, initUndistortRectifyMap | Complete | 3d.stereo_calibrate, 3d.stereo_rectify. Matched pinhole target observations and a rectified pair with projection and Q matrices. |
| S3-14 | Wide-angle stereo fitting and rectification | fisheye.stereoCalibrate/stereoRectify, omnidir.stereoCalibrate and rectification maps | Planned | Fit and rectify a wide-angle rig and check held-out vertical disparity and depth scale. |
| S3-15 | Stereo disparity | StereoBM, StereoSGBM | Complete | 3d.disparity. Known pixel shift, invalid mask and fixed-point scaling. |
| S3-16 | Stereo confidence and filtering | ximgproc.createRightMatcher, createDisparityWLSFilter, left–right consistency | Complete | 3d.disparity. Verify confidence on known correspondences and occlusion. |
| S3-17 | Disparity-to-depth reprojection | reprojectImageTo3D | Complete | 3d.reproject. Known baseline and depth plane, retained pixel positions and masks. |
| S3-18 | RGB-D depth and color registration | Depth backprojection, rgbd.registerDepth, intrinsic and pose transforms | Complete | input.depth_file, image.rgbd, 3d.register_depth, 3d.depth_cloud. Metric depth and color with explicit camera grid. |
| S3-19 | Depth cleanup and edge-aware smoothing | Invalid-depth masks, hole fill, joint bilateral or guided filtering | Partial | 3d.fill_depth_holes fills small gaps. Validate a depth-aware smoother at a foreground edge with invalid samples and confidence. |
| S3-20 | Organized normals and cloud sampling | rgbd.RgbdNormals, voxel sampling and valid pixel selection | Complete | 3d.depth_normals, 3d.cloud_filter, 3d.cloud_voxel. Camera-derived points retain useful correspondence through selection. |
| S3-21 | Camera-derived cloud interchange and preview | PLY/PCD, pixel provenance and depth projection rendering | Complete | input.cloud, output.cloud, 3d.cloud_render. Round-trip camera-derived clouds and display a bounded sample. |
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
| S3-33 | Sparse multi-view reconstruction | Feature tracks, recoverPose, triangulatePoints, bundle adjustment | Partial | 3d.essential_pose and 3d.triangulate handle a pair. Verify tracks across three or more views and measure reprojection and scale behavior after refinement. |

## Stage 4 — Selected contrib functions for images and video

[OpenCV extra modules](https://docs.opencv.org/5.0/) ·
[Contrib source](https://github.com/opencv/opencv_contrib)

| ID | Capability | Functions or implementation | State | Shipped evidence or completion check |
| --- | --- | --- | --- | --- |
| S4-01 | Guided and domain-transform filtering | ximgproc.guidedFilter, dtFilter | Complete | cv.guided_filter, cv.domain_transform_filter. Edge-preserving smoothing with a guide image and selected writes. |
| S4-02 | Locally adaptive threshold formulas | ximgproc.niBlackThreshold | Complete | cv.local_threshold. Compare Niblack, Sauvola, Wolf and NICK on uneven illumination. |
| S4-03 | Superpixel segmentation | ximgproc.createSuperpixelSLIC, createSuperpixelLSC, createSuperpixelSEEDS | Complete | cv.slic_superpixels, cv.lsc_superpixels, cv.seeds_superpixels. Integer labels, boundary masks and measured boundary recall. |
| S4-04 | Graph segmentation and region proposals | ximgproc.segmentation.createGraphSegmentation, createSelectiveSearchSegmentation | Complete | cv.graph_segments, inspect.selective_search. Label planes and ranked candidate rectangles. |
| S4-05 | Native ridge response | ximgproc.RidgeDetectionFilter_create | Complete | cv.ridge_response. Dark and bright narrow structures on a known fixture. |
| S4-06 | Binary thinning | ximgproc.thinning | Complete | inspect.thin_mask. Zhang–Suen and Guo–Hall centerline masks. |
| S4-07 | Structural similarity and no-reference quality | quality.QualitySSIM_compute, QualityBRISQUE_create | Complete | inspect.image_quality, inspect.brisque_quality. Map-based comparison and model-backed sharp/blurred/compressed fixtures. |
| S4-08 | Gradient-based image quality | quality.QualityGMSD_compute | Planned | Compare spatial degradation of aligned images, emit the native map and verify score ordering with blur and noise. |
| S4-09 | Binary perceptual hashes | img_hash.AverageHash/PHash/BlockMeanHash/MarrHildrethHash | Complete | cv.image_hash, inspect.compare_image_hashes. Exact and near-duplicate comparison with declared hash type. |
| S4-10 | Color and radial perceptual hashes | img_hash.ColorMomentHash, RadialVarianceHash | Planned | Preserve native descriptor type and compare rotated/resampled and changed fixtures with metric-specific bounds. |
| S4-11 | Color chart detection and correction | mcc chart detection, ccm.ColorCorrectionModel | Complete | calibration.detect_color_chart, calibration.fit_ccm, calibration.apply_ccm. Fit and apply correction on declared chart samples with residuals. |
| S4-12 | White-balance algorithms | xphoto.createGrayworldWB, createSimpleWB, createLearningBasedWB | Planned | Verify neutral patch recovery, guard saturation, and pin any learned model used. |
| S4-13 | Fast global edge-preserving smoothing | ximgproc.createFastGlobalSmootherFilter | Planned | Remove small intensity variations while preserving measured edges on a selected image. |
| S4-14 | Joint bilateral filtering | ximgproc.jointBilateralFilter | Planned | Verify image detail transfer using an aligned guide and typed image output. |
| S4-15 | Rolling guidance filtering | ximgproc.rollingGuidanceFilter | Planned | Compare removal of small texture against preservation of larger part boundaries. |
| S4-16 | Extended inpainting for defect cleanup | xphoto.inpaint | Planned | Fill a masked defect using the supported algorithm and verify known background recovery. |
| S4-17 | DCT denoising | xphoto.dctDenoising | Planned | Quantify noise reduction and edge retention on a seeded color or grayscale fixture. |
| S4-18 | Fast line detection | ximgproc.createFastLineDetector | Planned | Verify segment positions and orientation on straight edges and retain collection metadata. |
| S4-19 | Compact binary descriptors | xfeatures2d.BriefDescriptorExtractor, FREAK | Planned | Compute descriptors at wired image keypoints and check Hamming matching across a registered image pair. |
| S4-20 | Dense local descriptors | xfeatures2d.DAISY | Planned | Sample image descriptors on a controlled grid and verify matching under known displacement. |
| S4-21 | Additional background models | bgsegm.createBackgroundSubtractorGSOC, createBackgroundSubtractorLSBP | Complete | sequence.gsoc_background, sequence.lsbp_background. Warm up on an empty scene and replay foreground motion. |
| S4-22 | Dual TV-L1 flow | optflow.createOptFlow_DualTVL1 | Planned | Report flow and confidence on a known moving patch and compare with the existing flow baseline. |
| S4-23 | DeepFlow | optflow.createOptFlow_DeepFlow | Planned | Verify displacement and occlusion behavior on a short recorded or synthetic sequence. |
| S4-24 | Dense RLOF | optflow.createOptFlow_DenseRLOF | Planned | Verify displacement, support mode and forward/backward validity on a translated texture. |
| S4-25 | Small or low-contrast QR decoding | wechat_qrcode.WeChatQRCode | Planned | Compare with the native QR path on scaled and low-contrast labels. Pin any detector and super-resolution assets. |
| S4-26 | HOG image descriptors | HOGDescriptor.compute | Complete | inspect.hog. Describe a configured image window and visualize cell orientations. |

Stage 5 job packages and the parallel editor, acquisition and production milestones
have their own acceptance criteria in the [roadmap](roadmap.md).
