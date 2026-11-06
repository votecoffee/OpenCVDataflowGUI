# Examples

Each folder contains a runnable project, reusable ToolBlock, or Python plugin and
a README with setup and what to inspect. Start with [Starter inspection](starter_inspection/README.md).
Run commands from the repository root. Shared fixtures are in [assets](assets/README.md).

| Example | Demonstrates |
| --- | --- |
| [25 template instances](template_instances/README.md) | Find a repeated fixed-scale template in a 25-part synthetic scene and emit a pose for each match. |
| [3D pose and triangulation](pose_3d/README.md) | Project synthetic 3D points into two cameras, recover a pose with PnP, and triangulate points. |
| [Annoy nearest neighbors](annoy_search/README.md) | Turn synthetic image colors into an ONNX embedding and rank three labeled reference vectors with an Annoy index. |
| [Bead width calipers](bead_calipers/README.md) | Measure paired edges and bead width with a synthetic image. |
| [Channels and image pyramids](channels_pyramids/README.md) | Split and reassemble BGR channels, then carry payloads and coordinate transforms through a four-level image pyramid. |
| [ChArUco board](charuco_board/README.md) | Identify corners on a cropped target and recover the board pose. |
| [ChArUco lens calibration](charuco_lens_calibration/README.md) | Batch eight target views and fit a lens profile from their visible corners. |
| [ChArUco calibration validation](validate_charuco_calibration/README.md) | Check the fitted profile on a separate cropped board and identify the worst corner. |
| [Hand–eye calibration](hand_eye_calibration/README.md) | Fit a robot-mounted camera from paired poses and compose its pose in the robot base frame. |
| [Circle grid calibration](circle_grid_calibration/README.md) | Detect a staggered dot target and map its centers to physical coordinates. |
| [Checkerboard capture quality](chessboard_quality/README.md) | Measure blur and contrast across a sharp checkerboard and a defocused copy. |
| [Lens correction maps](lens_correction_maps/README.md) | Load a lens profile, build reusable coordinate maps, and straighten a synthetic distorted grid. |
| [Packed remap coordinates](packed_remap/README.md) | Straighten a grid with floating and fixed-point maps, then compare pixel differences. |
| [Lens fit report](lens_fit_report/README.md) | Fit six circle-grid views, identify the largest reprojection error, and inspect parameter uncertainty. |
| [Validate lens calibration](validate_lens_calibration/README.md) | Check a saved profile on a separate circle-grid view and overlay the reprojection residuals. |
| [Lens map displacement](lens_map_displacement/README.md) | Measure correction strength and source coverage, and overlay directions on a heatmap. |
| [Lens map files](lens_map_files/README.md) | Reload a saved map pair and straighten a synthetic grid using the restored maps. |
| [Color chart calibration](color_chart_calibration/README.md) | Detect a bundled synthetic color chart, fit a correction model, and apply it to the image. |
| [Combined inspection regions](combined_regions/README.md) | Subtract the center of a circle and union the resulting ring with a second part region. |
| [Conveyor pick selection](conveyor_pick/README.md) | Find 25 fixed-scale parts and select a candidate using clearance and downstream position. |
| [Conveyor tracking](conveyor_tracking/README.md) | Find repeated parts in a synthetic moving sequence and track them using encoder displacement. |
| [Coordinated camera inputs](input_coordination/README.md) | Join two synthetic camera images to a simulated PLC part counter by ticket. |
| [Custom Python invert block](custom_invert/README.md) | Run a custom block that inverts a band of the image and demonstrates numeric and cross-field validation. |
| [Principal components](principal_components/README.md) | Fit PCA, project numeric rows, and compare reduced and full reconstruction. |
| [Feature covariance and outliers](feature_covariance/README.md) | Score correlated measurements and identify rows outside a distance limit. |
| [Solve linear system](linear_system/README.md) | Fit two calibration curves with a shared coefficient matrix and compare the residuals. |
| [Solve linear program](linear_program/README.md) | Allocate shared capacity or meet demand at minimum cost, with constraint slack and solver status. |
| [Solve polynomial roots](polynomial_roots/README.md) | Fit a measurement curve, solve for its zero crossings, and compare with a quartic that has complex roots. |
| [Multiply matrices](matrix_multiply/README.md) | Use fitted calibration coefficients to predict new readings and apply offsets. |
| [Invert matrix](matrix_inverse/README.md) | Recover signals from a square calibration matrix and compare a rank-deficient pseudoinverse. |
| [Singular value decomposition](matrix_svd/README.md) | Keep the dominant measurement mode, inspect its retained energy, and apply the reconstructed matrix. |
| [Eigendecomposition](matrix_eigen/README.md) | Project a measurement onto covariance axes and check the spectrum of a nonsymmetric mixing matrix. |
| [Check image samples](image_validity/README.md) | Detect a missing depth patch and use the valid-pixel mask to measure the corresponding intensity pixels. |
| [Replace nonfinite samples](nonfinite_replacement/README.md) | Fill a known synthetic depth patch with a configured value, track the changed pixels, and verify the result. |
| [Join images](join_images/README.md) | Compare a full inspection image with a filtered crop, separated by a gap. |
| [Paste image patch](paste_patch/README.md) | Copy a masked crop onto the inspection image, clipping at the canvas edge. |
| [Subpixel patch extraction](subpixel_crop/README.md) | Crop around a refined template center and map patch coordinates back to the scene. |
| [Polar image warp](polar_image_warp/README.md) | Unwrap a circular edge with a simulated defect and rewrap it into the source frame. |
| [Tile image](tile_image/README.md) | Repeat a reference in a 2 × 3 grid and find all six copies with template matching. |
| [Transpose image](transpose_image/README.md) | Reflect a crop across its anti-diagonal and map points back to the original image. |
| [Scale absolute to 8-bit](scale_absolute/README.md) | Convert signed Sobel responses to magnitudes and threshold both edge polarities. |
| [Gram matrix](gram_matrix/README.md) | Compute feature covariance and reuse the learned offset for a new batch. |
| [Find nearest rows](nearest_rows/README.md) | Compare scaled measurements with masked candidates and mutual nearest-neighbor matching. |
| [Sort numeric table](numeric_table_sort/README.md) | Rank complete part records by score and compare that with independent column sorting. |
| [K-means clustering](kmeans/README.md) | Cluster image pixels and numeric samples, then inspect the color map, quantized image, and cluster plot. |
| [Histogram thresholds](histogram_thresholds/README.md) | Visualize percentiles, mean, and median, and wire measured cutoffs into three threshold masks. |
| [Flat-field illumination correction](flat_field/README.md) | Remove a lighting gradient using bright and dark references while retaining a synthetic defect. |
| [Tilted window statistics](tilted_statistics/README.md) | Measure two diagonal patches using batched 45-degree windows. |
| [Earth Mover's Distance](earth_movers_distance/README.md) | Measure a known shift between histograms with different sample counts. |
| [Absolute image error](absolute_image_error/README.md) | Compare mean, RMS, and maximum absolute error on two known deviations. |
| [Rotated rectangle overlap](rotated_rectangle_overlap/README.md) | Fit a part footprint and measure overlap against a reference. |
| [Convex polygon overlap](convex_overlap/README.md) | Compare a detected part hull with an editable polygon and display the intersection. |
| [Enclosing convex polygon](enclosing_polygon/README.md) | Fit and compare a triangle and quadrilateral around a detected contour. |
| [Delaunay triangulation](delaunay_triangulation/README.md) | Connect five detected dot centers and overlay their four triangles and eight edges. |
| [Hough lines from sparse points](point_set_lines/README.md) | Find two gapped lines amid outliers, then sort and draw their supported spans. |
| [Hough polar lines](hough_polar_lines/README.md) | Compare the strongest lines under binary and intensity-weighted edge voting. |
| [Hough circle collections](hough_circle_collections/README.md) | Rank circles by native detection strength and draw only those within the desired radius range. |
| [Voronoi cells](voronoi_cells/README.md) | Divide an image around five detected centers and measure intensities inside the selected cell. |
| [Minimum spanning tree](minimum_spanning_tree/README.md) | Find and overlay the shortest network connecting five detected centers using Delaunay edges. |
| [Generalized Hough matching](generalized_hough/README.md) | Locate two copies of a template shape, then restrict the search results to a region. |
| [Rotated and scaled Hough matching](generalized_hough_guil/README.md) | Find a template with a different rotation and scale, and inspect its oriented outline. |
| [Image summary statistics](image_statistics/README.md) | Measure a selected patch and mark the extrema in source pixel coordinates. |
| [Path intensity profile](path_intensity_profile/README.md) | Sample all color channels along a bent path and preserve excluded samples. |
| [Intensity profile](intensity_profile/README.md) | Plot the mean and locate the maximum in each selected column. |
| [Find nonzero pixels](nonzero_pixels/README.md) | Extract a bounded set of coordinates and full bounds from a texture mask. |
| [Pixel comparison](pixel_comparison/README.md) | Compare a background difference with a scalar to build a texture mask. |
| [Motion deblur](motion_deblur/README.md) | Restore a frame blurred by an eight-pixel horizontal motion path. |
| [Hann window](hann_window/README.md) | Compare Fourier leakage before and after tapering an off-bin sinusoid. |
| [Adaptive background correction](adaptive_background/README.md) | Flatten a lighting gradient so a fixed threshold finds a dark defect. |
| [Wiener deblur](wiener_deblur/README.md) | Recover detail after a known Gaussian blur and measure the change in RMSE. |
| [TV-L1 denoising](tvl1_denoising/README.md) | Fuse three aligned views to suppress impulse noise. |
| [Temporal non-local means](temporal_nlm/README.md) | Denoise the middle frame of a short sequence and preserve its ticket. |
| [Pyramid mean-shift filtering](mean_shift_filter/README.md) | Flatten checkerboard color texture and separate the two color classes. |
| [Gabor orientation bank](gabor_orientation_bank/README.md) | Find dominant stripe directions and reject a flat patch. |
| [Gabor texture direction](gabor_texture/README.md) | Compare two oriented texture responses on vertical and horizontal stripes. |
| [Derivative-fed Canny](canny_derivatives/README.md) | Inspect both Sobel derivatives and compare the resulting edges with ordinary Canny. |
| [Polar field rotation](polar_fields/README.md) | Rotate a Sobel gradient field by 90 degrees while preserving its magnitude. |
| [Image arithmetic](image_arithmetic/README.md) | Subtract a blurred background and apply scalar gain to the signed residual. |
| [Intensity transform](intensity_transform/README.md) | Clip a ramp, apply a natural logarithm, and recover the clipped values with exponentiation. |
| [Local window statistics](local_statistics/README.md) | Distinguish a flat patch from a checker texture using local standard deviation. |
| [Intensity lookup table](intensity_lut/README.md) | Apply distinct intensity curves to an RGB ramp. |
| [Color normalization](normalize_color/README.md) | Compare shared and independent channel scaling inside a region while preserving outside pixels. |
| [Linear channel transform](channel_transform/README.md) | Mix BGR channels and derive a grayscale difference image inside a region. |
| [Route image channels](route_channels/README.md) | Build a false-color image by routing two source channels and a zero-filled channel. |
| [Blend with weight maps](blend_weight_maps/README.md) | Feather a boundary between a sharp image and its blurred copy. |
| [Colorize scalar image](colorize_scalar/README.md) | Show how far each foreground pixel lies from the edge of a synthetic part. |
| [Labeled distance transform](labeled_distance/README.md) | Partition an image into nearest-seed zones and emit a mask for one zone. |
| [Masks from signed labels](label_masks/README.md) | Select watershed boundaries and interiors directly from the label image. |
| [Fourier correlation](fourier_correlation/README.md) | Recover a known signed shift and inspect the lag map. |
| [Iterative phase correlation](iterative_phase_correlation/README.md) | Estimate a fractional-pixel shift and apply the inverse warp. |
| [Matched-feature translation](translation_registration/README.md) | Fit a translation from SIFT matches and reproduce a known shifted view. |
| [Fourier gain mask](fourier_gain/README.md) | Apply a custom notch at a specific frequency pair and reconstruct the image. |
| [Fourier spectrum filtering](fourier_filter/README.md) | Reject a periodic stripe pattern with a band-stop filter and inverse DFT. |
| [Discrete cosine transform](dct_round_trip/README.md) | Inspect DCT coefficients and verify the reconstructed image against the source. |
| [EXIF orientation](exif_orientation/README.md) | Read a JPEG's metadata and compare the original and oriented pixel grids. |
| [Raw YUV decoding](raw_yuv/README.md) | Decode the same NV12 frame from raw bytes and a packed matrix. |
| [Multipage TIFF output](multipage_tiff_output/README.md) | Batch three pages and write a new lossless TIFF. |
| [TIFF page sequence](tiff_page_sequence/README.md) | Replay a three-page 16-bit TIFF as separately ticketed images. |
| [In-memory image codecs](in_memory_image_codecs/README.md) | Encode and decode 16-bit RGBA through a typed PNG packet and verify zero error. |
| [Bayer demosaicing](bayer_demosaicing/README.md) | Decode a synthetic 16-bit camera mosaic with bilinear and edge-aware interpolation. |
| [Domain transform filter](domain_transform_filter/README.md) | Compare three edge-preserving smoothing modes on the same noisy image. |
| [Guided edge-preserving filter](guided_filter/README.md) | Smooth a noisy image while preserving its main intensity edge. |
| [Subpixel template locations](subpixel_template/README.md) | Compare integer and refined template matches against two known fractional placements. |
| [Contour profile tolerance](contour_profile/README.md) | Compare aligned outlines for missing and excess material. |
| [Contour defect inspection](contour_defects/README.md) | Replay passing and failing parts with shallow and deep edge notches. |
| [Contour approximation](contour_approximation/README.md) | Simplify the dense outline of a chamfered part into six polygon vertices. |
| [Contour hierarchy](contour_hierarchy/README.md) | Follow parent and child links through nested outlines, holes, and islands. |
| [Lab color classification](lab_color_classification/README.md) | Classify a measured image patch against a named Lab palette. |
| [Stereo WLS confidence](stereo_confidence/README.md) | Gate disparity with a confidence image before reconstructing a partly occluded stereo pair. |
| [Depth hole filling](depth_hole_filling/README.md) | Fill small enclosed depth gaps and mark estimated samples before point-cloud reconstruction. |
| [SIFT registration](sift_registration/README.md) | Match SIFT descriptors and recover a known image translation. |
| [AKAZE registration](akaze_registration/README.md) | Match binary AKAZE descriptors and recover a known image translation. |
| [FLANN feature registration](flann_registration/README.md) | Match SIFT descriptors through a KD-tree and recover a known translation. |
| [KAZE registration](kaze_registration/README.md) | Match extended KAZE descriptors with FLANN and recover a known translation. |
| [Descriptor radius matching](descriptor_radius/README.md) | Keep multiple BRISK candidates within a Hamming distance limit and estimate the image shift. |
| [BRISK registration](brisk_registration/README.md) | Detect and match BRISK keypoints before estimating a known image shift. |
| [MSER region detection](mser_regions/README.md) | Compare stable extremal regions detected with both polarities and bright-only detection. |
| [Skeletonization and spur pruning](skeleton_pruning/README.md) | Extract a binary centerline and visualize the short branches removed from it. |
| [Watershed separation](watershed/README.md) | Separate touching parts using distance-peak seeds or points placed on the image. |
| [Data Matrix reading](datamatrix_reading/README.md) | Read the bundled synthetic Data Matrix and verify the decoded text. |
| [DNN color embeddings](dnn_embeddings/README.md) | Use a bundled ONNX model that averages RGB channels, then expose classification and vector outputs. |
| [Editable inspection regions](regions/README.md) | Use circular, rectangular, and polygon regions to restrict measurements and image processing. |
| [Image-set workspace](image_set_workspace/README.md) | Inspect a sequence of three bundled images in a grouped workspace with filename tickets. |
| [Inspect shapes ToolBlock](inspect_shapes/README.md) | Reuse threshold and contour inspection with a nested preprocessing graph. |
| [Lab color matching](lab_color_match/README.md) | Measure separate reference and sample regions and compare their Lab colors. |
| [Lena features and contrast](lena_features/README.md) | Enhance local contrast on Lena and inspect corners, Canny edges, and Hough line segments. |
| [Lena image inspection](lena/README.md) | Use a nested preprocessing ToolBlock, histogram and edge views, and template matching on Lena. |
| [Lena image registration](lena_registration/README.md) | Rotate Lena by a known angle, match ORB descriptors, estimate the homography, and inspect the aligned difference. |
| [LightGlue feature matching](lightglue_match/README.md) | Match a synthetic textured reference into a scene using SIFT features and LightGlue. |
| [Native background subtraction](native_background/README.md) | Compare OpenCV MOG2 and KNN foreground masks on a synthetic sequence. |
| [Native box tracking](native_box_tracking/README.md) | Compare CSRT and KCF tracking of a box in a synthetic image sequence. |
| [Object collections](object_collections/README.md) | Detect blobs, contours, and lines, then filter, sort, and render the collections. |
| [Overlay result graphics](overlay_result_graphics/README.md) | Combine selected region and contour graphics on the final color image. |
| [Parallel inspection of 25 parts](parallel_parts/README.md) | Find 25 synthetic parts and run a reusable inspection around each part pose. |
| [Pose-relative inspection](pose_inspection/README.md) | Train a geometric pattern and use its detected pose to position a circular inspection region. |
| [Quick-start demo](demo/README.md) | Run the editor's default two-part inspection and inspect its contour output. |
| [Queued image inspection](queued_input/README.md) | Place a FIFO image queue ahead of the 25-part parallel inspection. |
| [Regions through image transforms](transformed_regions/README.md) | Carry a region through crop, resize, and rotation, then measure the transformed selection. |
| [Registered defect inspection ToolBlock](registered_defect_inspection/README.md) | Align a sample to a reference with ECC, difference the images, and find defect blobs. |
| [RGB-D point cloud](rgbd_cloud/README.md) | Project synthetic depth into 3D, filter and voxel-sample the cloud, and estimate normals. |
| [Sequence memory](sequence_memory/README.md) | Run background subtraction, dense optical flow, point tracking, and reference-image learning on a synthetic sequence. |
| [Single template match](template_matching/README.md) | Crop a synthetic reference patch and locate it in the full image. |
| [Starter inspection](starter_inspection/README.md) | Crop a synthetic scene, filter noise, threshold and clean the mask, and measure two parts. |
| [Stereo depth and point cloud](stereo_cloud/README.md) | Rectify synthetic stereo images, estimate disparity, and reproject it to a measured 3D cloud. |
| [Structured-light Gray code](structured_light/README.md) | Generate synthetic Gray-code pattern images and decode projector correspondences. |
| [Ticket results and final judgment](ticket_results/README.md) | Track block results for a synthetic acquisition and publish a final judgment before a pass-only branch. |
| [Ticketed control flow](control_flow/README.md) | Group three synthetic acquisitions and exercise split, join, routing, and FIFO reordering with tickets. |
| [Ticketed recipes](recipes/README.md) | Select recipe settings and trained models, then coordinate shape and texture results for a final judgment. |
| [Trained patterns and oriented edges](trained_patterns/README.md) | Train geometric and ORB references, match them in a synthetic scene, and measure oriented edges. |
| [TSDF depth fusion](tsdf_fusion/README.md) | Fuse posed synthetic depth frames into a volume and render the extracted point cloud. |
| [Validation image set](validation_images/README.md) | Validate a synthetic inspection against bundled pass, fail, and unknown image cases. |
