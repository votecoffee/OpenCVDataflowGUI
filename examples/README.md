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
| [Color chart calibration](color_chart_calibration/README.md) | Detect a bundled synthetic color chart, fit a correction model, and apply it to the image. |
| [Combined inspection regions](combined_regions/README.md) | Subtract the center of a circle and union the resulting ring with a second part region. |
| [Conveyor pick selection](conveyor_pick/README.md) | Find 25 fixed-scale parts and select a candidate using clearance and downstream position. |
| [Conveyor tracking](conveyor_tracking/README.md) | Find repeated parts in a synthetic moving sequence and track them using encoder displacement. |
| [Coordinated camera inputs](input_coordination/README.md) | Join two synthetic camera images to a simulated PLC part counter by ticket. |
| [Custom Python invert block](custom_invert/README.md) | Run a custom block that inverts a band of the image and demonstrates numeric and cross-field validation. |
| [K-means clustering](kmeans/README.md) | Cluster image pixels and numeric samples, then inspect the color map, quantized image, and cluster plot. |
| [Histogram thresholds](histogram_thresholds/README.md) | Visualize percentiles, mean, and median, and wire measured cutoffs into three threshold masks. |
| [Flat-field illumination correction](flat_field/README.md) | Remove a lighting gradient using bright and dark references while retaining a synthetic defect. |
| [Guided edge-preserving filter](guided_filter/README.md) | Smooth a noisy image while preserving its main intensity edge. |
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
