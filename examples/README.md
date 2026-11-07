# Examples

Browse examples by workflow below. Each example has its own folder with a runnable
project, reusable ToolBlock, or Python plugin and a README with setup and what to
inspect. Start with [Starter inspection](getting_started/starter_inspection/README.md).
Run commands from the repository root. Shared fixtures are in [assets](assets/README.md).

## Getting Started

| Example | Demonstrates |
| --- | --- |
| [Lena image inspection](getting_started/lena/README.md) | Use a nested preprocessing ToolBlock, histogram and edge views, and template matching on Lena. |
| [Quick-start demo](getting_started/demo/README.md) | Run the editor's default two-part inspection and inspect its contour output. |
| [Starter inspection](getting_started/starter_inspection/README.md) | Crop a synthetic scene, filter noise, threshold and clean the mask, and measure two parts. |

## Inputs

| Example | Demonstrates |
| --- | --- |
| [Bayer demosaicing](inputs/bayer_demosaicing/README.md) | Decode a synthetic 16-bit camera mosaic with bilinear and edge-aware interpolation. |
| [EXIF orientation](inputs/exif_orientation/README.md) | Read a JPEG's metadata and compare the original and oriented pixel grids. |
| [Image-set workspace](inputs/image_set_workspace/README.md) | Inspect a sequence of three bundled images in a grouped workspace with filename tickets. |
| [Raw YUV decoding](inputs/raw_yuv/README.md) | Decode the same NV12 frame from raw bytes and a packed matrix. |
| [TIFF page sequence](inputs/tiff_page_sequence/README.md) | Replay a three-page 16-bit TIFF as separately ticketed images. |

## Image Processing

| Example | Demonstrates |
| --- | --- |
| [Adaptive background correction](image_processing/adaptive_background/README.md) | Flatten a lighting gradient so a fixed threshold finds a dark defect. |
| [Blend with weight maps](image_processing/blend_weight_maps/README.md) | Feather a boundary between a sharp image and its blurred copy. |
| [Channels and image pyramids](image_processing/channels_pyramids/README.md) | Split and reassemble BGR channels, then carry payloads and coordinate transforms through a four-level image pyramid. |
| [Check image samples](image_processing/image_validity/README.md) | Detect a missing depth patch and use the valid-pixel mask to measure the corresponding intensity pixels. |
| [Color normalization](image_processing/normalize_color/README.md) | Compare shared and independent channel scaling inside a region while preserving outside pixels. |
| [Colorize scalar image](image_processing/colorize_scalar/README.md) | Show how far each foreground pixel lies from the edge of a synthetic part. |
| [Derivative-fed Canny](image_processing/canny_derivatives/README.md) | Inspect both Sobel derivatives and compare the resulting edges with ordinary Canny. |
| [Discrete cosine transform](image_processing/dct_round_trip/README.md) | Inspect DCT coefficients and verify the reconstructed image against the source. |
| [Domain transform filter](image_processing/domain_transform_filter/README.md) | Compare three edge-preserving smoothing modes on the same noisy image. |
| [Find nonzero pixels](image_processing/nonzero_pixels/README.md) | Extract a bounded set of coordinates and full bounds from a texture mask. |
| [Flat-field illumination correction](image_processing/flat_field/README.md) | Remove a lighting gradient using bright and dark references while retaining a synthetic defect. |
| [Fourier gain mask](image_processing/fourier_gain/README.md) | Apply a custom notch at a specific frequency pair and reconstruct the image. |
| [Fourier spectrum filtering](image_processing/fourier_filter/README.md) | Reject a periodic stripe pattern with a band-stop filter and inverse DFT. |
| [Fourier spectrum product](image_processing/spectrum_product/README.md) | Combine two DFT spectra and recover a known wrapped shift from the reconstructed lag map. |
| [Gabor orientation bank](image_processing/gabor_orientation_bank/README.md) | Find dominant stripe directions and reject a flat patch. |
| [Gabor texture direction](image_processing/gabor_texture/README.md) | Compare two oriented texture responses on vertical and horizontal stripes. |
| [Guided edge-preserving filter](image_processing/guided_filter/README.md) | Smooth a noisy image while preserving its main intensity edge. |
| [Hann window](image_processing/hann_window/README.md) | Compare Fourier leakage before and after tapering an off-bin sinusoid. |
| [Image arithmetic](image_processing/image_arithmetic/README.md) | Subtract a blurred background and apply scalar gain to the signed residual. |
| [Intensity lookup table](image_processing/intensity_lut/README.md) | Apply distinct intensity curves to an RGB ramp. |
| [Intensity transform](image_processing/intensity_transform/README.md) | Clip a ramp, apply a natural logarithm, and recover the clipped values with exponentiation. |
| [Join images](image_processing/join_images/README.md) | Compare a full inspection image with a filtered crop, separated by a gap. |
| [Labeled distance transform](image_processing/labeled_distance/README.md) | Partition an image into nearest-seed zones and emit a mask for one zone. |
| [Lena features and contrast](image_processing/lena_features/README.md) | Enhance local contrast on Lena and inspect corners, Canny edges, and Hough line segments. |
| [Linear channel transform](image_processing/channel_transform/README.md) | Mix BGR channels and derive a grayscale difference image inside a region. |
| [Local window statistics](image_processing/local_statistics/README.md) | Distinguish a flat patch from a checker texture using local standard deviation. |
| [Masks from signed labels](image_processing/label_masks/README.md) | Select watershed boundaries and interiors directly from the label image. |
| [Motion deblur](image_processing/motion_deblur/README.md) | Restore a frame blurred by an eight-pixel horizontal motion path. |
| [MSER region detection](image_processing/mser_regions/README.md) | Compare stable extremal regions detected with both polarities and bright-only detection. |
| [Paste image patch](image_processing/paste_patch/README.md) | Copy a masked crop onto the inspection image, clipping at the canvas edge. |
| [Pixel comparison](image_processing/pixel_comparison/README.md) | Compare a background difference with a scalar to build a texture mask. |
| [Polar field rotation](image_processing/polar_fields/README.md) | Rotate a Sobel gradient field by 90 degrees while preserving its magnitude. |
| [Polar image warp](image_processing/polar_image_warp/README.md) | Unwrap a circular edge with a simulated defect and rewrap it into the source frame. |
| [Pyramid mean-shift filtering](image_processing/mean_shift_filter/README.md) | Flatten checkerboard color texture and separate the two color classes. |
| [Replace nonfinite samples](image_processing/nonfinite_replacement/README.md) | Fill a known synthetic depth patch with a configured value, track the changed pixels, and verify the result. |
| [Route image channels](image_processing/route_channels/README.md) | Build a false-color image by routing two source channels and a zero-filled channel. |
| [Scale absolute to 8-bit](image_processing/scale_absolute/README.md) | Convert signed Sobel responses to magnitudes and threshold both edge polarities. |
| [Seeded random images and noise](image_processing/seeded_noise/README.md) | Compare two independently scheduled noise branches for exact agreement. |
| [Skeletonization and spur pruning](image_processing/skeleton_pruning/README.md) | Extract a binary centerline and visualize the short branches removed from it. |
| [Stitch an ordered inspection scan](image_processing/stitch_scan_sequence/README.md) | Automatically register a three-tile scan and measure only covered pixels. |
| [Close the loop on an inspection scan](image_processing/stitch_scan_loop/README.md) | Compare adjacent registration with a bounded correction when the scan returns to its first view. |
| [Correct exposure in an inspection scan](image_processing/stitch_scan_exposure/README.md) | Compare raw and exposure-matched 16-bit mosaics with feathered seams. |
| [Stitch overlapping inspection views](image_processing/stitch_image_pair/README.md) | Reconstruct a scan from two crops and verify covered pixels against the original. |
| [Temporal non-local means](image_processing/temporal_nlm/README.md) | Denoise the middle frame of a short sequence and preserve its ticket. |
| [Tile image](image_processing/tile_image/README.md) | Repeat a reference in a 2 × 3 grid and find all six copies with template matching. |
| [Tilted window statistics](image_processing/tilted_statistics/README.md) | Measure two diagonal patches using batched 45-degree windows. |
| [Transpose image](image_processing/transpose_image/README.md) | Reflect a crop across its anti-diagonal and map points back to the original image. |
| [TV-L1 denoising](image_processing/tvl1_denoising/README.md) | Fuse three aligned views to suppress impulse noise. |
| [Watershed separation](image_processing/watershed/README.md) | Separate touching parts using distance-peak seeds or points placed on the image. |
| [Wiener deblur](image_processing/wiener_deblur/README.md) | Recover detail after a known Gaussian blur and measure the change in RMSE. |

## Pattern Matching and Registration

| Example | Demonstrates |
| --- | --- |
| [25 template instances](pattern_matching/template_instances/README.md) | Find a repeated fixed-scale template in a 25-part synthetic scene and emit a pose for each match. |
| [AKAZE registration](pattern_matching/akaze_registration/README.md) | Match binary AKAZE descriptors and recover a known image translation. |
| [BRISK registration](pattern_matching/brisk_registration/README.md) | Detect and match BRISK keypoints before estimating a known image shift. |
| [Descriptor radius matching](pattern_matching/descriptor_radius/README.md) | Keep multiple BRISK candidates within a Hamming distance limit and estimate the image shift. |
| [FLANN feature registration](pattern_matching/flann_registration/README.md) | Match SIFT descriptors through a KD-tree and recover a known translation. |
| [Fourier correlation](pattern_matching/fourier_correlation/README.md) | Recover a known signed shift and inspect the lag map. |
| [Generalized Hough matching](pattern_matching/generalized_hough/README.md) | Locate two copies of a template shape, then restrict the search results to a region. |
| [Iterative phase correlation](pattern_matching/iterative_phase_correlation/README.md) | Estimate a fractional-pixel shift and apply the inverse warp. |
| [KAZE registration](pattern_matching/kaze_registration/README.md) | Match extended KAZE descriptors with FLANN and recover a known translation. |
| [Lena image registration](pattern_matching/lena_registration/README.md) | Rotate Lena by a known angle, match ORB descriptors, estimate the homography, and inspect the aligned difference. |
| [Matched-feature translation](pattern_matching/translation_registration/README.md) | Fit a translation from SIFT matches and reproduce a known shifted view. |
| [Rotated and scaled Hough matching](pattern_matching/generalized_hough_guil/README.md) | Find a template with a different rotation and scale, and inspect its oriented outline. |
| [SIFT registration](pattern_matching/sift_registration/README.md) | Match SIFT descriptors and recover a known image translation. |
| [Single template match](pattern_matching/template_matching/README.md) | Crop a synthetic reference patch and locate it in the full image. |
| [Subpixel patch extraction](pattern_matching/subpixel_crop/README.md) | Crop around a refined template center and map patch coordinates back to the scene. |
| [Subpixel template locations](pattern_matching/subpixel_template/README.md) | Compare integer and refined template matches against two known fractional placements. |
| [Trained patterns and oriented edges](pattern_matching/trained_patterns/README.md) | Train geometric and ORB references, match them in a synthetic scene, and measure oriented edges. |

## Inspection and Measurement

| Example | Demonstrates |
| --- | --- |
| [Absolute image error](inspection/absolute_image_error/README.md) | Compare mean, RMS, and maximum absolute error on two known deviations. |
| [Bead width calipers](inspection/bead_calipers/README.md) | Measure paired edges and bead width with a synthetic image. |
| [Contour defect inspection](inspection/contour_defects/README.md) | Replay passing and failing parts with shallow and deep edge notches. |
| [Contour profile tolerance](inspection/contour_profile/README.md) | Compare aligned outlines for missing and excess material. |
| [Conveyor pick selection](inspection/conveyor_pick/README.md) | Find 25 fixed-scale parts and select a candidate using clearance and downstream position. |
| [Data Matrix reading](inspection/datamatrix_reading/README.md) | Read the bundled synthetic Data Matrix and verify the decoded text. |
| [Earth Mover's Distance](inspection/earth_movers_distance/README.md) | Measure a known shift between histograms with different sample counts. |
| [Histogram thresholds](inspection/histogram_thresholds/README.md) | Visualize percentiles, mean, and median, and wire measured cutoffs into three threshold masks. |
| [Image summary statistics](inspection/image_statistics/README.md) | Measure a selected patch and mark the extrema in source pixel coordinates. |
| [Intensity profile](inspection/intensity_profile/README.md) | Plot the mean and locate the maximum in each selected column. |
| [Lab color classification](inspection/lab_color_classification/README.md) | Classify a measured image patch against a named Lab palette. |
| [Lab color matching](inspection/lab_color_match/README.md) | Measure separate reference and sample regions and compare their Lab colors. |
| [Measure image norms](inspection/image_norm/README.md) | Measure absolute and reference-relative norms inside a patch and apply an inclusive limit. |
| [Overlay result graphics](inspection/overlay_result_graphics/README.md) | Combine selected region and contour graphics on the final color image. |
| [Path intensity profile](inspection/path_intensity_profile/README.md) | Sample all color channels along a bent path and preserve excluded samples. |
| [Pose-relative inspection](inspection/pose_inspection/README.md) | Train a geometric pattern and use its detected pose to position a circular inspection region. |
| [Validation image set](inspection/validation_images/README.md) | Validate a synthetic inspection against bundled pass, fail, and unknown image cases. |

## Geometry and Regions

| Example | Demonstrates |
| --- | --- |
| [Clip line segments](geometry/clip_line_segments/README.md) | Measure the portion of a long detected edge inside a rectangular inspection window. |
| [Combined inspection regions](geometry/combined_regions/README.md) | Subtract the center of a circle and union the resulting ring with a second part region. |
| [Contour approximation](geometry/contour_approximation/README.md) | Simplify the dense outline of a chamfered part into six polygon vertices. |
| [Contour hierarchy](geometry/contour_hierarchy/README.md) | Follow parent and child links through nested outlines, holes, and islands. |
| [Convex contour approximation](geometry/convex_contour_approximation/README.md) | Compare a four-sided footprint with a six-sided approximation under an area budget. |
| [Convex polygon overlap](geometry/convex_overlap/README.md) | Compare a detected part hull with an editable polygon and display the intersection. |
| [Delaunay triangulation](geometry/delaunay_triangulation/README.md) | Connect five detected dot centers and overlay their four triangles and eight edges. |
| [Editable inspection regions](geometry/regions/README.md) | Use circular, rectangular, and polygon regions to restrict measurements and image processing. |
| [Enclosing convex polygon](geometry/enclosing_polygon/README.md) | Fit and compare a triangle and quadrilateral around a detected contour. |
| [Fitted ellipse inspection band](geometry/fitted_shape_region/README.md) | Fit a rotated ellipse, subtract a smaller fitted region from a larger one, and measure the selected edge band. |
| [Hough circle collections](geometry/hough_circle_collections/README.md) | Rank circles by native detection strength and draw only those within the desired radius range. |
| [Hough lines from sparse points](geometry/point_set_lines/README.md) | Find two gapped lines amid outliers, then sort and draw their supported spans. |
| [Hough polar lines](geometry/hough_polar_lines/README.md) | Compare the strongest lines under binary and intensity-weighted edge voting. |
| [Minimum spanning tree](geometry/minimum_spanning_tree/README.md) | Find and overlay the shortest network connecting five detected centers using Delaunay edges. |
| [Object collections](geometry/object_collections/README.md) | Detect blobs, contours, and lines, then filter, sort, and render the collections. |
| [Regions through image transforms](geometry/transformed_regions/README.md) | Carry a region through crop, resize, and rotation, then measure the transformed selection. |
| [Rotated rectangle overlap](geometry/rotated_rectangle_overlap/README.md) | Fit a part footprint and measure overlap against a reference. |
| [Voronoi cells](geometry/voronoi_cells/README.md) | Divide an image around five detected centers and measure intensities inside the selected cell. |

## Calibration

| Example | Demonstrates |
| --- | --- |
| [ChArUco board](calibration/charuco_board/README.md) | Identify corners on a cropped target and recover the board pose. |
| [ChArUco calibration validation](calibration/validate_charuco_calibration/README.md) | Check the fitted profile on a separate cropped board and identify the worst corner. |
| [ChArUco lens calibration](calibration/charuco_lens_calibration/README.md) | Batch eight target views and fit a lens profile from their visible corners. |
| [Checkerboard capture quality](calibration/chessboard_quality/README.md) | Measure blur and contrast across a sharp checkerboard and a defocused copy. |
| [Circle grid calibration](calibration/circle_grid_calibration/README.md) | Detect a staggered dot target and map its centers to physical coordinates. |
| [Color chart calibration](calibration/color_chart_calibration/README.md) | Detect a bundled synthetic color chart, fit a correction model, and apply it to the image. |
| [Hand–eye calibration](calibration/hand_eye_calibration/README.md) | Fit a robot-mounted camera from paired poses and compose its pose in the robot base frame. |
| [Lens correction maps](calibration/lens_correction_maps/README.md) | Load a lens profile, build reusable coordinate maps, and straighten a synthetic distorted grid. |
| [Lens fit report](calibration/lens_fit_report/README.md) | Fit six circle-grid views, identify the largest reprojection error, and inspect parameter uncertainty. |
| [Lens map displacement](calibration/lens_map_displacement/README.md) | Measure correction strength and source coverage, and overlay directions on a heatmap. |
| [Lens map files](calibration/lens_map_files/README.md) | Reload a saved map pair and straighten a synthetic grid using the restored maps. |
| [Packed remap coordinates](calibration/packed_remap/README.md) | Straighten a grid with floating and fixed-point maps, then compare pixel differences. |
| [Validate lens calibration](calibration/validate_lens_calibration/README.md) | Check a saved profile on a separate circle-grid view and overlay the reprojection residuals. |

## 3D

| Example | Demonstrates |
| --- | --- |
| [3D pose and triangulation](3d/pose_3d/README.md) | Project synthetic 3D points into two cameras, recover a pose with PnP, and triangulate points. |
| [Depth hole filling](3d/depth_hole_filling/README.md) | Fill small enclosed depth gaps and mark estimated samples before point-cloud reconstruction. |
| [RGB-D point cloud](3d/rgbd_cloud/README.md) | Project synthetic depth into 3D, filter and voxel-sample the cloud, and estimate normals. |
| [Stereo depth and point cloud](3d/stereo_cloud/README.md) | Rectify synthetic stereo images, estimate disparity, and reproject it to a measured 3D cloud. |
| [Stereo WLS confidence](3d/stereo_confidence/README.md) | Gate disparity with a confidence image before reconstructing a partly occluded stereo pair. |
| [Structured-light Gray code](3d/structured_light/README.md) | Generate synthetic Gray-code pattern images and decode projector correspondences. |
| [TSDF depth fusion](3d/tsdf_fusion/README.md) | Fuse posed synthetic depth frames into a volume and render the extracted point cloud. |

## ML and DNN

| Example | Demonstrates |
| --- | --- |
| [Annoy nearest neighbors](ml/annoy_search/README.md) | Turn synthetic image colors into an ONNX embedding and rank three labeled reference vectors with an Annoy index. |
| [DNN color embeddings](ml/dnn_embeddings/README.md) | Use a bundled ONNX model that averages RGB channels, then expose classification and vector outputs. |
| [K-means clustering](ml/kmeans/README.md) | Cluster image pixels and numeric samples, then inspect the color map, quantized image, and cluster plot. |
| [LightGlue feature matching](ml/lightglue_match/README.md) | Match a synthetic textured reference into a scene using SIFT features and LightGlue. |
| [Principal components](ml/principal_components/README.md) | Fit PCA, project numeric rows, and compare reduced and full reconstruction. |

## Numerical Tools

| Example | Demonstrates |
| --- | --- |
| [Eigendecomposition](numerical/matrix_eigen/README.md) | Project a measurement onto covariance axes and check the spectrum of a nonsymmetric mixing matrix. |
| [Feature covariance and outliers](numerical/feature_covariance/README.md) | Score correlated measurements and identify rows outside a distance limit. |
| [Find nearest rows](numerical/nearest_rows/README.md) | Compare scaled measurements with masked candidates and mutual nearest-neighbor matching. |
| [Gram matrix](numerical/gram_matrix/README.md) | Compute feature covariance and reuse the learned offset for a new batch. |
| [Invert matrix](numerical/matrix_inverse/README.md) | Recover signals from a square calibration matrix and compare a rank-deficient pseudoinverse. |
| [Matrix structure and regularization](numerical/matrix_structure/README.md) | Mirror a triangle, add a scaled identity, and inspect the diagonal and trace before inversion. |
| [Multiply matrices](numerical/matrix_multiply/README.md) | Use fitted calibration coefficients to predict new readings and apply offsets. |
| [Singular value decomposition](numerical/matrix_svd/README.md) | Keep the dominant measurement mode, inspect its retained energy, and apply the reconstructed matrix. |
| [Solve linear program](numerical/linear_program/README.md) | Allocate shared capacity or meet demand at minimum cost, with constraint slack and solver status. |
| [Solve linear system](numerical/linear_system/README.md) | Fit two calibration curves with a shared coefficient matrix and compare the residuals. |
| [Solve polynomial roots](numerical/polynomial_roots/README.md) | Fit a measurement curve, solve for its zero crossings, and compare with a quartic that has complex roots. |
| [Sort numeric table](numerical/numeric_table_sort/README.md) | Rank complete part records by score and compare that with independent column sorting. |

## Control Flow and Sequences

| Example | Demonstrates |
| --- | --- |
| [Camera motion stabilization](flow_control/camera_stabilization/README.md) | Replay known camera jitter and compare raw and corrected error against an anchor frame. |
| [Conveyor tracking](flow_control/conveyor_tracking/README.md) | Find repeated parts in a synthetic moving sequence and track them using encoder displacement. |
| [Coordinated camera inputs](flow_control/input_coordination/README.md) | Join two synthetic camera images to a simulated PLC part counter by ticket. |
| [Native background subtraction](flow_control/native_background/README.md) | Compare OpenCV MOG2 and KNN foreground masks on a synthetic sequence. |
| [Native box tracking](flow_control/native_box_tracking/README.md) | Compare CSRT and KCF tracking of a box in a synthetic image sequence. |
| [Parallel inspection of 25 parts](flow_control/parallel_parts/README.md) | Find 25 synthetic parts and run a reusable inspection around each part pose. |
| [Queued image inspection](flow_control/queued_input/README.md) | Place a FIFO image queue ahead of the 25-part parallel inspection. |
| [Sequence memory](flow_control/sequence_memory/README.md) | Run background subtraction, dense optical flow, point tracking, and reference-image learning on a synthetic sequence. |
| [Ticket results and final judgment](flow_control/ticket_results/README.md) | Track block results for a synthetic acquisition and publish a final judgment before a pass-only branch. |
| [Ticketed control flow](flow_control/control_flow/README.md) | Group three synthetic acquisitions and exercise split, join, routing, and FIFO reordering with tickets. |
| [Ticketed recipes](flow_control/recipes/README.md) | Select recipe settings and trained models, then coordinate shape and texture results for a final judgment. |

## Outputs and Codecs

| Example | Demonstrates |
| --- | --- |
| [In-memory image codecs](outputs/in_memory_image_codecs/README.md) | Encode and decode 16-bit RGBA through a typed PNG packet and verify zero error. |
| [Multipage TIFF output](outputs/multipage_tiff_output/README.md) | Batch three pages and write a new lossless TIFF. |

## User Tools and Plugins

| Example | Demonstrates |
| --- | --- |
| [Custom Python invert block](user_tools/custom_invert/README.md) | Edit an inversion band in a plugin popup and preview it with numeric and cross-field validation. |
| [Inspect shapes ToolBlock](user_tools/inspect_shapes/README.md) | Reuse threshold and contour inspection with a nested preprocessing graph. |
| [Registered defect inspection ToolBlock](user_tools/registered_defect_inspection/README.md) | Align a sample to a reference with ECC, difference the images, and find defect blobs. |
