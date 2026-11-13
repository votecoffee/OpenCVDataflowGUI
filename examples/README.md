# Examples

Each example has its own folder and README. Start with
[Starter inspection](demos/starter_inspection/README.md). Commands run from
the repository root. Shared fixtures are in [assets](assets/README.md).
Examples with third-party sources include a **Comment** on the canvas.
See [image sources and licenses](assets/photograph-sources.md) for the bundled photographs.

## Demos

| Example | Demonstrates |
| --- | --- |
| [Image-set workspace](demos/image_set_workspace/README.md) | Inspect a sequence of three bundled images in a grouped workspace with filename tickets. |
| [Lena features and contrast](demos/lena_features/README.md) | Enhance local contrast on Lena and inspect corners, Canny edges, and Hough line segments. |
| [Lena image inspection](demos/lena/README.md) | Use a nested preprocessing ToolBlock, histogram and edge views, and template matching on Lena. |
| [Quick-start demo](demos/demo/README.md) | Run the editor's default two-part inspection and inspect its contour output. |
| [Starter inspection](demos/starter_inspection/README.md) | Crop a synthetic scene, filter noise, threshold and clean the mask, and measure two parts. |
| [Validation image set](demos/validation_images/README.md) | Validate a synthetic inspection against bundled pass, fail, and unknown image cases. |

## Applications

See the [applications guide](applications/README.md) for the application collection.

| Example | Demonstrates |
| --- | --- |
| [Bead width calipers](applications/bead_calipers/README.md) | Measure paired edges and bead width with a synthetic image. |
| [Contour defect inspection](applications/contour_defects/README.md) | Replay passing and failing parts with shallow and deep edge notches. |
| [Contour profile tolerance](applications/contour_profile/README.md) | Compare aligned outlines for missing and excess material. |
| [Conveyor pick selection](applications/conveyor_pick/README.md) | Find 25 fixed-scale parts and select a candidate using clearance and downstream position. |
| [Conveyor tracking](applications/conveyor_tracking/README.md) | Find repeated parts in a synthetic moving sequence and track them using encoder displacement. |
| [Data Matrix reading](applications/datamatrix_reading/README.md) | Read the bundled synthetic Data Matrix and verify the decoded text. |
| [Lab color classification](applications/lab_color_classification/README.md) | Classify a measured image patch against a named Lab palette. |
| [Lab color matching](applications/lab_color_match/README.md) | Measure separate reference and sample regions and compare their Lab colors. |
| [PCB outline](applications/pcb_outline/README.md) | Check board presence and overlay its outline on two inspection photographs. |
| [Pose-relative inspection](applications/pose_inspection/README.md) | Train a geometric pattern and use its detected pose to position a circular inspection region. |
| [Shipping label barcode reading](applications/shipping_label_barcodes/README.md) | Replay four Dynamsoft shipping label photos for Code 128 and read MaxiCode in an isolated crop. |

## Simple Plugins

| Example | Demonstrates |
| --- | --- |
| [SimpleSegmentation](<Simple Plugins/pixel_segmentation/README.md>) | Train foreground masks for brick mortar, a cup rim, PCB laminate, and inventory in real totes. |
| [SimpleClassify](<Simple Plugins/image_classification/README.md>) | Compare two classifier profiles on labeled VegQual produce crops. |
| [SimplePatternMatch](<Simple Plugins/pattern_matching/asymmetric_brackets/README.md>) | Locate synthetic brackets with direct and wired recipes, and inspect the pose and edge overlays. |
| [SimplePatternMatch on real PCB photographs](<Simple Plugins/pattern_matching/visa_pcb2/README.md>) | Train on one VisA PCB and transfer its mounting-hole datum to a separate photograph. |

See the [Simple Plugins guide](<Simple Plugins/README.md>) for setup and project links.

## Industrial integration

| Example | Demonstrates |
| --- | --- |
| [FANUC hand–eye calibration](<industrial integration/fanuc_hand_eye_calibration/README.md>) | Convert synthetic FANUC XYZ/WPR captures and fit the camera mount from ticket-paired poses. |
| [FANUC registers and frames](<industrial integration/fanuc_registers_and_frames/README.md>) | Wire R and PR reads, UFRAME and UTOOL offsets, flange-to-base composition, and reviewed register changes. |
| [ControlLogix trigger and writeback](<industrial integration/controllogix_trigger_and_writeback/README.md>) | Trigger a ticketed inspection, write the part count, and verify readback. |

The two hardware templates open with controller access disabled.
See the [industrial integration examples](<industrial integration/README.md>) for setup and replay details.

## OpenCV tools

| Example | Demonstrates |
| --- | --- |
| [25 template instances](<opencv tools/template_instances/README.md>) | Find a repeated fixed-scale template in a 25-part synthetic scene and emit a pose for each match. |
| [3D pose and triangulation](<opencv tools/pose_3d/README.md>) | Project synthetic 3D points into two cameras, recover a pose with PnP, and triangulate points. |
| [Affine-invariant features](<opencv tools/affine_features/README.md>) | Match an original tilted inspection fixture using SIFT/ORB affine views, retain source-space support geometry, and verify against a known transform. |
| [AKAZE registration](<opencv tools/akaze_registration/README.md>) | Match binary AKAZE descriptors and recover a known image translation. |
| [Bayer demosaicing](<opencv tools/bayer_demosaicing/README.md>) | Decode a synthetic 16-bit camera mosaic with bilinear and edge-aware interpolation. |
| [Brick edges](<opencv tools/brick_edges/README.md>) | Reduce surface texture with Gaussian blur and find mortar edges with Canny. |
| [BRISK registration](<opencv tools/brisk_registration/README.md>) | Detect and match BRISK keypoints before estimating a known image shift. |
| [Channels and image pyramids](<opencv tools/channels_pyramids/README.md>) | Split and reassemble BGR channels, then carry payloads and coordinate transforms through a four-level image pyramid. |
| [ChArUco board](<opencv tools/charuco_board/README.md>) | Identify corners on a cropped target and recover the board pose. |
| [ChArUco lens calibration](<opencv tools/charuco_lens_calibration/README.md>) | Batch eight target views and fit a lens profile from their visible corners. |
| [Circle grid calibration](<opencv tools/circle_grid_calibration/README.md>) | Detect a staggered dot target and map its centers to physical coordinates. |
| [Color chart calibration](<opencv tools/color_chart_calibration/README.md>) | Detect a bundled synthetic color chart, fit a correction model, and apply it to the image. |
| [Color normalization](<opencv tools/normalize_color/README.md>) | Compare shared and independent channel scaling inside a region while preserving outside pixels. |
| [Colorize scalar image](<opencv tools/colorize_scalar/README.md>) | Show how far each foreground pixel lies from the edge of a synthetic part. |
| [Contour approximation](<opencv tools/contour_approximation/README.md>) | Simplify the dense outline of a chamfered part into six polygon vertices. |
| [Circle and arc fitting](<opencv tools/circle_arc_fitting/README.md>) | Compare robust and ordinary circle fits with outliers, build a circle-relative region, and reject a five-degree arc. |
| [Contour hierarchy](<opencv tools/contour_hierarchy/README.md>) | Follow parent and child links through nested outlines, holes, and islands. |
| [Contour rasterization](<opencv tools/contour_rasterization/README.md>) | Select a part with its holes and islands, compare explicit solid filling, and verify exact source-grid masks. |
| [Convex contour approximation](<opencv tools/convex_contour_approximation/README.md>) | Compare a four-sided footprint with a six-sided approximation under an area budget. |
| [Delaunay triangulation](<opencv tools/delaunay_triangulation/README.md>) | Connect five detected dot centers and overlay their four triangles and eight edges. |
| [Derivative-fed Canny](<opencv tools/canny_derivatives/README.md>) | Inspect both Sobel derivatives and compare the resulting edges with ordinary Canny. |
| [Descriptor radius matching](<opencv tools/descriptor_radius/README.md>) | Keep multiple BRISK candidates within a Hamming distance limit and estimate the image shift. |
| [DIS optical flow](<opencv tools/dis_optical_flow/README.md>) | Measure motion of a textured patch with a known displacement. |
| [Discrete cosine transform](<opencv tools/dct_round_trip/README.md>) | Inspect DCT coefficients and verify the reconstructed image against the source. |
| [DNN color embeddings](<opencv tools/dnn_embeddings/README.md>) | Use a bundled ONNX model that averages RGB channels, then expose classification and vector outputs. |
| [Domain transform filter](<opencv tools/domain_transform_filter/README.md>) | Compare three edge-preserving smoothing modes on the same noisy image. |
| [Earth Mover's Distance](<opencv tools/earth_movers_distance/README.md>) | Measure a known shift between histograms with different sample counts. |
| [Eigendecomposition](<opencv tools/matrix_eigen/README.md>) | Project a measurement onto covariance axes and check the spectrum of a nonsymmetric mixing matrix. |
| [Enclosing convex polygon](<opencv tools/enclosing_polygon/README.md>) | Fit and compare a triangle and quadrilateral around a detected contour. |
| [EXIF orientation](<opencv tools/exif_orientation/README.md>) | Read a JPEG's metadata and compare the original and oriented pixel grids. |
| [Feature covariance and outliers](<opencv tools/feature_covariance/README.md>) | Score correlated measurements and identify rows outside a distance limit. |
| [Find nonzero pixels](<opencv tools/nonzero_pixels/README.md>) | Extract a bounded set of coordinates and full bounds from a texture mask. |
| [FLANN feature registration](<opencv tools/flann_registration/README.md>) | Match SIFT descriptors through a KD-tree and recover a known translation. |
| [Binary mask boundary and area inspection](<opencv tools/mask_boundary/README.md>) | Compare full mask boundaries and missing/extra areas using tab/notch pass/fail fixtures. |
| [Fourier component reconstruction](<opencv tools/fourier_components/README.md>) | Compare full, phase-only and magnitude-only color reconstruction, verify independent impulse truth, and inspect typed spectral diagnostics. |
| [Focus stacking](<opencv tools/focus_stack/README.md>) | Fuse complementary focal captures, inspect level/coverage maps and verify sharp truth after optional ECC registration. |
| [Log-polar rotation/scale registration](<opencv tools/logpolar_registration/README.md>) | Recover an independently known similarity, reject a symmetric pose and test an explicit orientation prior. |
| [Video codec round trip](<opencv tools/video_codec_roundtrip/README.md>) | Record/replay original moving color targets and qualify six lossless/lossy codecs for exact pixels or bounded error, count, order and rate. |
| [Fourier correlation](<opencv tools/fourier_correlation/README.md>) | Recover a known signed shift and inspect the lag map. |
| [Aligned image comparison](<opencv tools/aligned_image_comparison/README.md>) | Compare an intact part and a missing-bore fixture with a reference, per-channel correlation and normalized residuals. |
| [Laplacian pyramids](<opencv tools/laplacian_pyramid/README.md>) | Decompose an odd-sized 16-bit color image into signed bands, verify exact reconstruction, and remove the finest detail. |
| [Floating-point image codecs](<opencv tools/float_image_codecs/README.md>) | Verify exact float32/64 TIFF storage and bounded Radiance HDR loss while preserving raw sample units and channel order. |
| [High-bit-depth codecs](<opencv tools/high_bit_depth_codecs/README.md>) | Verify exact 16-bit PNG and JPEG 2000 color/alpha or grayscale round trips, with explicit codec controls. |
| [Packed RAW10/RAW12 sensors](<opencv tools/packed_raw_sensor/README.md>) | Decode original MIPI byte fixtures with headers and padded rows, verify ADC counts, and demosaic an odd-origin crop using phase metadata. |
| [Fourier spectrum product](<opencv tools/spectrum_product/README.md>) | Combine two DFT spectra and recover a known wrapped shift from the reconstructed lag map. |
| [Gabor texture direction](<opencv tools/gabor_texture/README.md>) | Compare two oriented texture responses on vertical and horizontal stripes. |
| [Generalized Hough matching](<opencv tools/generalized_hough/README.md>) | Locate two copies of a template shape, then restrict the search results to a region. |
| [Gram matrix](<opencv tools/gram_matrix/README.md>) | Compute feature covariance and reuse the learned offset for a new batch. |
| [Guided edge-preserving filter](<opencv tools/guided_filter/README.md>) | Smooth a noisy image while preserving its main intensity edge. |
| [Hand–eye calibration](<opencv tools/hand_eye_calibration/README.md>) | Fit a robot-mounted camera from paired poses and compose its pose in the robot base frame. |
| [Hann window](<opencv tools/hann_window/README.md>) | Compare Fourier leakage before and after tapering an off-bin sinusoid. |
| [Hough circle collections](<opencv tools/hough_circle_collections/README.md>) | Rank circles by native detection strength and draw only those within the desired radius range. |
| [Hough lines from sparse points](<opencv tools/point_set_lines/README.md>) | Find two gapped lines amid outliers, then sort and draw their supported spans. |
| [Hough polar lines](<opencv tools/hough_polar_lines/README.md>) | Compare the strongest lines under binary and intensity-weighted edge voting. |
| [Image arithmetic](<opencv tools/image_arithmetic/README.md>) | Subtract a blurred background and apply scalar gain to the signed residual. |
| [Image summary statistics](<opencv tools/image_statistics/README.md>) | Measure a selected patch and mark the extrema in source pixel coordinates. |
| [In-memory image codecs](<opencv tools/in_memory_image_codecs/README.md>) | Encode and decode 16-bit RGBA through a typed PNG packet and verify zero error. |
| [Intensity lookup table](<opencv tools/intensity_lut/README.md>) | Apply distinct intensity curves to an RGB ramp. |
| [Intensity transform](<opencv tools/intensity_transform/README.md>) | Clip a ramp, apply a natural logarithm, and recover the clipped values with exponentiation. |
| [Invert matrix](<opencv tools/matrix_inverse/README.md>) | Recover signals from a square calibration matrix and compare a rank-deficient pseudoinverse. |
| [Iterative phase correlation](<opencv tools/iterative_phase_correlation/README.md>) | Estimate a fractional-pixel shift and apply the inverse warp. |
| [K-means clustering](<opencv tools/kmeans/README.md>) | Cluster image pixels and numeric samples, then inspect the color map, quantized image, and cluster plot. |
| [KAZE registration](<opencv tools/kaze_registration/README.md>) | Match extended KAZE descriptors with FLANN and recover a known translation. |
| [Labeled distance transform](<opencv tools/labeled_distance/README.md>) | Partition an image into nearest-seed zones and emit a mask for one zone. |
| [Lena image registration](<opencv tools/lena_registration/README.md>) | Rotate Lena by a known angle, match ORB descriptors, estimate the homography, and inspect the aligned difference. |
| [Lens correction maps](<opencv tools/lens_correction_maps/README.md>) | Load a lens profile, build reusable coordinate maps, and straighten a synthetic distorted grid. |
| [Linear channel transform](<opencv tools/channel_transform/README.md>) | Mix BGR channels and derive a grayscale difference image inside a region. |
| [Local window statistics](<opencv tools/local_statistics/README.md>) | Distinguish a flat patch from a checker texture using local standard deviation. |
| [Intensity-weighted center and orientation](<opencv tools/weighted_moments/README.md>) | Compare a 16-bit asymmetric brightness feature with analytic geometric/weighted centers, axes and covariance, then position an inspection region from its pose. |
| [Local structure tensor and orientation](<opencv tools/structure_tensor/README.md>) | Measure eigenvalues, confidence and unsigned normal/tangent directions on a 16-bit fixture; verify known axes and ambiguous corner/flat/weak probes. |
| [Motion-compensated frame interpolation](<opencv tools/frame_interpolation/README.md>) | Generate a 16-bit intermediate frame, check independent image truth and coverage, and retain explicit source timing and lineage. |
| [Matched-feature translation](<opencv tools/translation_registration/README.md>) | Fit a translation from SIFT matches and reproduce a known shifted view. |
| [Matrix structure and regularization](<opencv tools/matrix_structure/README.md>) | Mirror a triangle, add a scaled identity, and inspect the diagonal and trace before inversion. |
| [Measure image norms](<opencv tools/image_norm/README.md>) | Measure absolute and reference-relative norms inside a patch and apply an inclusive limit. |
| [MSER region detection](<opencv tools/mser_regions/README.md>) | Compare stable extremal regions detected with both polarities and bright-only detection. |
| [Multipage TIFF output](<opencv tools/multipage_tiff_output/README.md>) | Batch three pages and write a new lossless TIFF. |
| [Multiply matrices](<opencv tools/matrix_multiply/README.md>) | Use fitted calibration coefficients to predict new readings and apply offsets. |
| [Native background subtraction](<opencv tools/native_background/README.md>) | Compare OpenCV MOG2 and KNN foreground masks on a synthetic sequence. |
| [Native box tracking](<opencv tools/native_box_tracking/README.md>) | Compare CSRT and KCF tracking of a box in a synthetic image sequence. |
| [Packed remap coordinates](<opencv tools/packed_remap/README.md>) | Straighten a grid with floating and fixed-point maps, then compare pixel differences. |
| [PCB motion](<opencv tools/pcb_motion/README.md>) | Replay two PCB photographs through DIS to estimate apparent motion. |
| [Pixel comparison](<opencv tools/pixel_comparison/README.md>) | Compare a background difference with a scalar to build a texture mask. |
| [Polar field rotation](<opencv tools/polar_fields/README.md>) | Rotate a Sobel gradient field by 90 degrees while preserving its magnitude. |
| [Polar image warp](<opencv tools/polar_image_warp/README.md>) | Unwrap a circular edge with a simulated defect and rewrap it into the source frame. |
| [Principal components](<opencv tools/principal_components/README.md>) | Fit PCA, project numeric rows, and compare reduced and full reconstruction. |
| [Pyramid mean-shift filtering](<opencv tools/mean_shift_filter/README.md>) | Flatten checkerboard color texture and separate the two color classes. |
| [Raw YUV decoding](<opencv tools/raw_yuv/README.md>) | Decode the same NV12 frame from raw bytes and a packed matrix. |
| [Rotated and scaled Hough matching](<opencv tools/generalized_hough_guil/README.md>) | Find a template with a different rotation and scale, and inspect its oriented outline. |
| [Rotated rectangle overlap](<opencv tools/rotated_rectangle_overlap/README.md>) | Fit a part footprint and measure overlap against a reference. |
| [Scale absolute to 8-bit](<opencv tools/scale_absolute/README.md>) | Convert signed Sobel responses to magnitudes and threshold both edge polarities. |
| [Source-over alpha composition](<opencv tools/alpha_composition/README.md>) | Compare straight, premultiplied and external-map alpha on original 16-bit layers against an exact-rational reference, including transparent backgrounds. |
| [SIFT registration](<opencv tools/sift_registration/README.md>) | Match SIFT descriptors and recover a known image translation. |
| [Single template match](<opencv tools/template_matching/README.md>) | Crop a synthetic reference patch and locate it in the full image. |
| [Singular value decomposition](<opencv tools/matrix_svd/README.md>) | Keep the dominant measurement mode, inspect its retained energy, and apply the reconstructed matrix. |
| [Solve linear program](<opencv tools/linear_program/README.md>) | Allocate shared capacity or meet demand at minimum cost, with constraint slack and solver status. |
| [Solve linear system](<opencv tools/linear_system/README.md>) | Fit two calibration curves with a shared coefficient matrix and compare the residuals. |
| [Solve polynomial roots](<opencv tools/polynomial_roots/README.md>) | Fit a measurement curve, solve for its zero crossings, and compare with a quartic that has complex roots. |
| [Stereo depth and point cloud](<opencv tools/stereo_cloud/README.md>) | Rectify synthetic stereo images, estimate disparity, and reproject it to a measured 3D cloud. |
| [Stereo WLS confidence](<opencv tools/stereo_confidence/README.md>) | Gate disparity with a confidence image before reconstructing a partly occluded stereo pair. |
| [Structured-light Gray code](<opencv tools/structured_light/README.md>) | Generate synthetic Gray-code pattern images and decode projector correspondences. |
| [Subpixel patch extraction](<opencv tools/subpixel_crop/README.md>) | Crop around a refined template center and map patch coordinates back to the scene. |
| [Temporal non-local means](<opencv tools/temporal_nlm/README.md>) | Denoise the middle frame of a short sequence and preserve its ticket. |
| [TIFF page sequence](<opencv tools/tiff_page_sequence/README.md>) | Replay a three-page 16-bit TIFF as separately ticketed images. |
| [Transpose image](<opencv tools/transpose_image/README.md>) | Reflect a crop across its anti-diagonal and map points back to the original image. |
| [TV-L1 denoising](<opencv tools/tvl1_denoising/README.md>) | Fuse three aligned views to suppress impulse noise. |
| [Voronoi cells](<opencv tools/voronoi_cells/README.md>) | Divide an image around five detected centers and measure intensities inside the selected cell. |
| [Watershed separation](<opencv tools/watershed/README.md>) | Separate touching parts using distance-peak seeds or points placed on the image. |
| [Unicode inspection labels](<opencv tools/unicode_annotations/README.md>) | Compose Latin, Greek and Cyrillic labels over an original 16-bit fixture with selectable drawing layers, preserved tickets and a bundled licensed font. |

## Custom tools

| Example | Demonstrates |
| --- | --- |
| [Absolute image error](<custom tools/absolute_image_error/README.md>) | Compare mean, RMS, and maximum absolute error on two known deviations. |
| [Adaptive background correction](<custom tools/adaptive_background/README.md>) | Flatten a lighting gradient so a fixed threshold finds a dark defect. |
| [Annoy nearest neighbors](<custom tools/annoy_search/README.md>) | Turn synthetic image colors into an ONNX embedding and rank three labeled reference vectors with an Annoy index. |
| [Blend with weight maps](<custom tools/blend_weight_maps/README.md>) | Feather a boundary between a sharp image and its blurred copy. |
| [Camera motion stabilization](<custom tools/camera_stabilization/README.md>) | Replay known camera jitter and compare raw and corrected error against an anchor frame. |
| [ChArUco calibration validation](<custom tools/validate_charuco_calibration/README.md>) | Check the fitted profile on a separate cropped board and identify the worst corner. |
| [Check image samples](<custom tools/image_validity/README.md>) | Detect a missing depth patch and use the valid-pixel mask to measure the corresponding intensity pixels. |
| [Check stabilization on a second plane](<custom tools/stabilization_validation/README.md>) | Detect accumulated drift in a separate inspection region while the background fit still passes. |
| [Check stabilization on recorded depth variation](<custom tools/recorded_stabilization_validation/README.md>) | Verify drift rejection and recovery using depth-separated patches in a recorded camera sequence. |
| [Checkerboard capture quality](<custom tools/chessboard_quality/README.md>) | Measure blur and contrast across a sharp checkerboard and a defocused copy. |
| [Choose a content-aware seam](<custom tools/stitch_seam/README.md>) | Compare averaging with graph-cut selection on 16-bit views and try automatic seam scaling for larger overlaps. |
| [Clip line segments](<custom tools/clip_line_segments/README.md>) | Measure the portion of a long detected edge inside a rectangular inspection window. |
| [Close the loop on an inspection scan](<custom tools/stitch_scan_loop/README.md>) | Compare adjacent registration with a bounded correction when the scan returns to its first view. |
| [Coffee histograms](<custom tools/coffee_histograms/README.md>) | Compare hue, saturation, and value with a grayscale histogram and percentile markers. |
| [Combined inspection regions](<custom tools/combined_regions/README.md>) | Subtract the center of a circle and union the resulting ring with a second part region. |
| [Convex polygon overlap](<custom tools/convex_overlap/README.md>) | Compare a detected part hull with an editable polygon and display the intersection. |
| [Continuous centered stabilization](<custom tools/continuous_camera_stabilization/README.md>) | Smooth a 144-frame camera path with bounded look-ahead and automatic tail flush while retaining frame tickets. |
| [Correct exposure in an inspection scan](<custom tools/stitch_scan_exposure/README.md>) | Compare raw and exposure-matched 16-bit mosaics with feathered seams. |
| [Depth hole filling](<custom tools/depth_hole_filling/README.md>) | Fill small enclosed depth gaps and mark estimated samples before point-cloud reconstruction. |
| [Editable inspection regions](<custom tools/regions/README.md>) | Use circular, rectangular, and polygon regions to restrict measurements and image processing. |
| [Find nearest rows](<custom tools/nearest_rows/README.md>) | Compare scaled measurements with masked candidates and mutual nearest-neighbor matching. |
| [Fitted ellipse inspection band](<custom tools/fitted_shape_region/README.md>) | Fit a rotated ellipse, subtract a smaller fitted region from a larger one, and measure the selected edge band. |
| [Flat-field illumination correction](<custom tools/flat_field/README.md>) | Remove a lighting gradient using bright and dark references while retaining a synthetic defect. |
| [Build flat-field references](<custom tools/flat_field_reference/README.md>) | Estimate bright/dark maps from repeated captures, compare mean/median handling and verify a held-out color image. |
| [Fourier gain mask](<custom tools/fourier_gain/README.md>) | Apply a custom notch at a specific frequency pair and reconstruct the image. |
| [Fourier spectrum filtering](<custom tools/fourier_filter/README.md>) | Reject a periodic stripe pattern with a band-stop filter and inverse DFT. |
| [Gabor orientation bank](<custom tools/gabor_orientation_bank/README.md>) | Find dominant stripe directions and reject a flat patch. |
| [Histogram thresholds](<custom tools/histogram_thresholds/README.md>) | Visualize percentiles, mean, and median, and wire measured cutoffs into three threshold masks. |
| [Intensity profile](<custom tools/intensity_profile/README.md>) | Plot the mean and locate the maximum in each selected column. |
| [Join images](<custom tools/join_images/README.md>) | Compare a full inspection image with a filtered crop, separated by a gap. |
| [Lens fit report](<custom tools/lens_fit_report/README.md>) | Fit six circle-grid views, identify the largest reprojection error, and inspect parameter uncertainty. |
| [Lens map displacement](<custom tools/lens_map_displacement/README.md>) | Measure correction strength and source coverage, and overlay directions on a heatmap. |
| [Lens map files](<custom tools/lens_map_files/README.md>) | Reload a saved map pair and straighten a synthetic grid using the restored maps. |
| [LightGlue feature matching](<custom tools/lightglue_match/README.md>) | Match a synthetic textured reference into a scene using SIFT features and LightGlue. |
| [Line-pair metrology](<custom tools/line_pair_metrology/README.md>) | Detect cropped edges, align fitted lines to the original image, and judge corner angles or parallel gaps. |
| [Slanted-edge sharpness](<custom tools/slanted_edge_sharpness/README.md>) | Compare original sharp/blurred edges using ESF/SFR, MTF50 and an inclusive inspection limit. |
| [Masks from signed labels](<custom tools/label_masks/README.md>) | Select watershed boundaries and interiors directly from the label image. |
| [Minimum spanning tree](<custom tools/minimum_spanning_tree/README.md>) | Find and overlay the shortest network connecting five detected centers using Delaunay edges. |
| [Motion deblur](<custom tools/motion_deblur/README.md>) | Restore a frame blurred by an eight-pixel horizontal motion path. |
| [Object collections](<custom tools/object_collections/README.md>) | Detect blobs, contours, and lines, then filter, sort, and render the collections. |
| [Overlay result graphics](<custom tools/overlay_result_graphics/README.md>) | Combine selected region and contour graphics on the final color image. |
| [Paste image patch](<custom tools/paste_patch/README.md>) | Copy a masked crop onto the inspection image, clipping at the canvas edge. |
| [Path intensity profile](<custom tools/path_intensity_profile/README.md>) | Sample all color channels along a bent path and preserve excluded samples. |
| [Recorded camera stabilization](<custom tools/recorded_camera_stabilization/README.md>) | Replay real camera motion and measure raw and corrected error over the same coverage mask. |
| [Regions through image transforms](<custom tools/transformed_regions/README.md>) | Carry a region through crop, resize, and rotation, then measure the transformed selection. |
| [Replace nonfinite samples](<custom tools/nonfinite_replacement/README.md>) | Fill a known synthetic depth patch with a configured value, track the changed pixels, and verify the result. |
| [RGB-D point cloud](<custom tools/rgbd_cloud/README.md>) | Project synthetic depth into 3D, filter and voxel-sample the cloud, and estimate normals. |
| [Route image channels](<custom tools/route_channels/README.md>) | Build a false-color image by routing two source channels and a zero-filled channel. |
| [Seeded random images and noise](<custom tools/seeded_noise/README.md>) | Compare two independently scheduled noise branches for exact agreement. |
| [Sequence memory](<custom tools/sequence_memory/README.md>) | Run background subtraction, dense optical flow, point tracking, and reference-image learning on a synthetic sequence. |
| [Debounce foreground components](<custom tools/foreground_debounce/README.md>) | Confirm a moving part, reject intermittent glints/shadows, and require reconfirmation after a missed acquisition using exact expected masks. |
| [Controlled image distortion fixtures](<custom tools/distortion_fixture/README.md>) | Generate reproducible shear, perspective and lens-like warps, inspect analytic donor probes and reuse typed maps against independent 16-bit RGB references. |
| [Signal/background contrast](<custom tools/region_contrast/README.md>) | Compare known 16-bit RGB patches, verify channel CNR and expected strong/weak outcomes, and inspect an intentional overlap error. |
| [Reference-guided cast shadows](<custom tools/reference_shadows/README.md>) | Restore a clear/shadow pair with exact 16-bit RGB pixel and defect-contrast truth; inspect a deliberately rejected nonuniform model. |
| [Skeletonization and spur pruning](<custom tools/skeleton_pruning/README.md>) | Extract a binary centerline and visualize the short branches removed from it. |
| [Smooth a recorded camera path](<custom tools/offline_camera_stabilization/README.md>) | Apply centered look-ahead to a recorded batch and replay corrected frames with their source tickets. |
| [Sort numeric table](<custom tools/numeric_table_sort/README.md>) | Rank complete part records by score and compare that with independent column sorting. |
| [Stitch an ordered inspection scan](<custom tools/stitch_scan_sequence/README.md>) | Automatically register a three-tile scan and measure only covered pixels. |
| [Stitch overlapping inspection views](<custom tools/stitch_image_pair/README.md>) | Reconstruct a scan from two crops and verify covered pixels against the original. |
| [Subpixel template locations](<custom tools/subpixel_template/README.md>) | Compare integer and refined template matches against two known fractional placements. |
| [Tile image](<custom tools/tile_image/README.md>) | Repeat a reference in a 2 × 3 grid and find all six copies with template matching. |
| [Tiled inspection mosaic](<custom tools/tiled_scan_mosaic/README.md>) | Register twelve views and render bounded windows beyond dense-canvas limits, preserving source coordinates. |
| [Tilted window statistics](<custom tools/tilted_statistics/README.md>) | Measure two diagonal patches using batched 45-degree windows. |
| [Trained patterns and oriented edges](<custom tools/trained_patterns/README.md>) | Train geometric and ORB references, match them in a synthetic scene, and measure oriented edges. |
| [TSDF depth fusion](<custom tools/tsdf_fusion/README.md>) | Fuse posed synthetic depth frames into a volume and render the extracted point cloud. |
| [Validate lens calibration](<custom tools/validate_lens_calibration/README.md>) | Check a saved profile on a separate circle-grid view and overlay the reprojection residuals. |
| [Wiener deblur](<custom tools/wiener_deblur/README.md>) | Recover detail after a known Gaussian blur and measure the change in RMSE. |

## Control flow tools

Examples of pipeline control, including split, join, routing, queues, and decisions.

| Example | Demonstrates |
| --- | --- |
| [Coordinated camera inputs](<control flow tools/input_coordination/README.md>) | Join two synthetic camera images to a simulated PLC part counter by ticket. |
| [Parallel inspection of 25 parts](<control flow tools/parallel_parts/README.md>) | Find 25 synthetic parts and run a reusable inspection around each part pose. |
| [Queued image inspection](<control flow tools/queued_input/README.md>) | Place a FIFO image queue ahead of the 25-part parallel inspection. |
| [Ticket results and final judgment](<control flow tools/ticket_results/README.md>) | Track block results for a synthetic acquisition and publish a final judgment before a pass-only branch. |
| [Ticketed control flow](<control flow tools/control_flow/README.md>) | Group three synthetic acquisitions and exercise split, join, routing, and FIFO reordering with tickets. |
| [Ticketed recipes](<control flow tools/recipes/README.md>) | Select recipe settings and trained models, then coordinate shape and texture results for a final judgment. |

## User tools and plugins

| Example | Demonstrates |
| --- | --- |
| [Custom Python invert block](user_tools/custom_invert/README.md) | Edit an inversion band in a plugin popup and preview it with numeric and cross-field validation. |
| [Inspect shapes ToolBlock](user_tools/inspect_shapes/README.md) | Reuse threshold and contour inspection with a nested preprocessing graph. |
| [Registered defect inspection ToolBlock](user_tools/registered_defect_inspection/README.md) | Align a sample to a reference with ECC, difference the images, and find defect blobs. |
