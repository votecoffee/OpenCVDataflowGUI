# OpenCV tools

These examples demonstrate OpenCV image/video processing, features, calibration,
restoration, ML and 3D operations. Short pipelines around an OpenCV module also
belong here: stitching, stabilization, Fourier filtering and channel routing
are library workflows even when several blocks cooperate.

Browse the [complete OpenCV example index](../README.md#opencv-tools). Each
folder explains the relevant project, controls, expected result and any optional
dependency or asset preparation. Lena examples retain their explicit download
step and the local image remains excluded from Git.

For a first introduction to running and editing graphs, use the
[Demos guided tour](../demos/README.md). [Custom tools](<../custom tools/README.md>)
contains project-defined inspection algorithms and other-library integrations.

[Planar fiducial targets](fiducial_targets/README.md) generates printable
ArUco markers, ArUco grids and ChArUco boards and checks digital detection.

[Planar pose hypotheses](planar_pose_hypotheses/README.md) shows how close
reprojection scores can leave a plane orientation ambiguous.

[Epipolar match refinement](epipolar_refinement/README.md) scores noisy
pairs in two camera grids and checks corrected positions on a known rig.

[Uncalibrated stereo rectification](uncalibrated_stereo/README.md) uses
image matches to align scanlines and emit per-view and shared support masks.

[Stereo calibration quality](stereo_calibration_quality/README.md) checks
per-camera fit residuals and cross-camera prediction on held-out targets.

[Multi-camera rig calibration](multicamera_rig/README.md) fits a three-view
chain and validates it on disjoint captures.

[Stereo disparity search](stereo_disparity_search/README.md) derives a
smaller search from calibrated depth limits and verifies valid support.

[Disparity speckle removal](disparity_speckles/README.md) shows how to remove
small islands from fixed-point and signed float disparity.

[Stereo disparity evaluation](disparity_reference/README.md) compares an
estimate with known truth across the whole field and a selected region.

[Binary Census stereo](binary_stereo/README.md) compares Census/Hamming
matching and native SGBM on an exposure-shifted pair.

[Stereo exposure and color](stereo_color/README.md) matches a calibrated pair
from Census correspondences and compares native StereoBM on held-out pixels.

[Recorded RGB-D](recorded_rgbd/README.md) pairs staggered color and depth
frames, aligns depth and confidence, and builds a colored point cloud.

[Fisheye camera calibration](fisheye_calibration/README.md) shows a fit from
multiple grid images followed by held-out reprojection and rectification.
[Omnidirectional calibration](omnidir_calibration/README.md) fits the Mei camera
model and checks a separate target near the sensor edge.

[Calibration revisions](calibration_revisions/README.md) demonstrates named
rig versions, archived maps and a rejected capture mismatch.

[Wide-angle stereo](wide_angle_stereo/README.md) uses both lens families
with held-out perspective rectification and metric reconstruction.

[Depth-backed region reprojection](depth_region_projection/README.md) transfers
a physical patch across different lens models and checks the projected support.

[Confidence-weighted depth smoothing](depth_smoothing/README.md) preserves a
sharp step and rejects unreliable depth samples before cloud reconstruction.

[Registered depth fusion](registered_depth_fusion/README.md) blends aligned
range images with confidence and retains a narrow foreground.

[Multi-baseline stereo depth](multi_baseline_depth/README.md) registers
three Q-derived depths and compares consensus and nearest-surface fusion.

[Camera depth evaluation](depth_reference/README.md) checks mixed-unit depth
error, validity and coverage over a known step.

[Calibrated depth silhouette](depth_silhouette/README.md) measures visible
length, width and thickness for two raised parts on a tilted datum.

[Depth component measurement](depth_components/README.md) labels touching
parts at different depths and filters the reconstructed mask by metric width.

[Support plane and foreground separation](depth_plane/README.md) extracts
raised parts above a tilted surface into a foreground-only point cloud.

[Calibrated cloud visibility](cloud_visibility/README.md) renders depth and
color and checks visible surfaces against observed camera depth.

[Local cloud feature registration](cloud_feature_registration/README.md)
registers noisy partial scans and measures correspondence error and overlap.

[Phase-shift fringe replay](phase_shift/README.md) recovers absolute subpixel
projector coordinates and marks low-signal and inconsistent captures.

[Camera/projector reconstruction](camera_projector/README.md) fits separate
pixel grids and verifies metric depth on a held-out stepped scene.

[Robot-world and hand–eye calibration](robot_world/README.md) fits both
rigid transforms for fixed and robot-mounted cameras and checks held-out images.

[Sparse multi-view reconstruction](sparse_multiview/README.md) packs four
ChArUco views into a sparse cloud and compares fixed and refined camera poses.

[Metric scene-scale recovery](scene_scale/README.md) aligns a unitless cloud
and camera poses to a metric survey frame and checks eight holdout points.

[All examples](../README.md)

For ordered in-memory image groups, try [multipage TIFF buffers](multipage_memory/README.md).
For random pages and a bounded LRU cache, try [TIFF page access](tiff_page_access/README.md).
For temporal noise and channel/source covariance, try [pixel statistics](temporal_pixel_statistics/README.md).
For native QR finder backends, original-image corners and rectified modules, try [QR comparisons](qr_backends/README.md).
For ordered multipart payload recovery and deliberate rejection cases, try [structured-append QR](qr_structured_append/README.md).
For detector repeatability and matcher threshold curves, try [image feature evaluation](feature_evaluation/README.md).
For calibrated geometry, reordered/missing views and fixed canvas coordinates, try [mosaic layout reuse](mosaic_layout_reuse/README.md).
For independent registered frames, common-valid crops and serial inspections, try [registered-view replay](registered_view_replay/README.md).
For supported frequency gain/phase and exact blur/shift references, try [relative frequency response](frequency_response/README.md).
For saved corrections, content/timing checks and no-fit replay, try [motion trajectories](motion_trajectory/README.md).
For sharper registered neighbors, change gates and preserved target support, try [neighbor restoration](neighbor_restoration/README.md).
For validated row-dependent motion, video correction and current-part preservation, try [rolling-shutter correction](rolling_shutter/README.md).
For saved clean scenes, first-image foreground masks and guarded updates, try [background reference restoration](background_reference/README.md).
For reader/encoder checks and page counts, try [codec preflight](codec_probe/README.md).
For stored displacement and verified image pairs, try [flow file round trips](flow_file_roundtrip/README.md).
For seeded local tracking and explicit loss/reacquisition, try [MIL box tracking](mil_box_tracking/README.md).
For bounded native stream decoding and grouped inspections, try [video buffers](video_buffer/README.md).
For exact binary neighborhoods, try [hit-or-miss patterns](hit_or_miss/README.md).
For sparse-mask extraction and allocation limits, try [run-linked contours](run_contours/README.md).
For oriented part identities and bounded missed detections, try [image tracking](image_tracking/README.md).
For scheduled full-frame detection with local image refinement, try [detector refresh](detector_refresh/README.md).
For actual variable or unreliable clocks and independent time comparisons, try [video timing controls](video_timing/README.md).
For independent frame sampling with requested/actual time and bracket policies, try [video at time](video_time/README.md).
For live reception, bounded reconnects and sequence resets, try [network video](network_video/README.md).
For recorded media timing, stride and loop/reset inspection, try [paced video replay](paced_video/README.md).
For whole selections through lens correction, exact holes and border validity, try [selection remap](selection_remap/README.md).
For candidate appearance support before clearance/downstream selection, try [visible support](visible_support/README.md).
For motion-aware HDR, measured support and native exposure alignment, try [HDR motion](hdr_motion/README.md).
For unordered rotating views, fitted cameras, partial components and mapped measurements, try [rotating panoramas](rotating_panorama/README.md).
For explicit native solvers, fixed/refined intrinsics and reserved validation, try [panorama refinement](panorama_refinement/README.md).
For stabilized borders, neighboring samples and unchanged measured coverage, try [review border fills](stabilization_fill/README.md).
For requested capture settings, actual backend replies and strict/report-only policies, try [capture negotiation](capture_negotiation/README.md).
For wrapped QR labels, independent surface/module references and declared decoder misses, try [curved QR](curved_qr/README.md).
For detector head formats, source boxes, class-aware NMS and a pinned real-photo model, try [DNN detection](dnn_detection/README.md).
For paired images, numeric auxiliaries, four named inputs and explicit ticket/reference roles, try [multi-input inference](multi_input_inference/README.md).

For native image batches with dynamic dimensions, explicit tails and per-image tickets, try [batch inference](batch_inference/README.md).

For model/settings revisions, portable bundles, receipt verbosity and target fallback, try [model contracts](model_contracts/README.md).

For oriented boxes, rotated IoU, affine poses and local inspection regions, try [rotated detection](rotated_detection/README.md).

For per-pixel score/ID heads, source-grid masks, confidence and area checks, try [semantic segmentation](semantic_segmentation/README.md).

For separate object masks, preserved overlap, ROI/prototype heads and per-instance inspection, try [instance segmentation](instance_segmentation/README.md).

For ordered keypoints, heatmap fitting, partial results and local inspection poses, try [landmarks](landmarks/README.md).

For saved palettes, HOG vocabularies and unknown-sample inspection, try [reusable image clusters](reusable_clusters/README.md).

For native HOG training, held-out labels and unknown-image policies, try [K-nearest classification](knn_classification/README.md).

For native forest weights, feature importance and held-out class votes, try [random forest classification](rtrees_classification/README.md).

For labeled image sets, grouped splits, confusion matrices and explicit wrong/rejected/error controls, try [classifier evaluation](classifier_evaluation/README.md).

For temperature fitting, held-out reliability and an abstaining native ONNX classifier, try [confidence calibration](confidence_calibration/README.md).

For crop training, bounded false-positive mining and held-out full scenes, try [HOG detector training](hog_detector_training/README.md).

For class-aware vote grouping and stable detections, try [spatial votes](spatial_vote_grouping/README.md).

For known-good descriptor training and a sealed margin cutoff, try [one-class SVM inspection](one_class_svm/README.md).

For source-box labels, IoU matches, precision/recall/AP and duplicate/miss/error controls, try [detector evaluation](detector_evaluation/README.md).

For explicit Classic/New/Auto loading, frozen snapshots and native rejection controls, try [ONNX engine dispatch](dnn_engines/README.md).

For a pinned pretrained SSD model, separate output heads, source-image graphics and a hand-marked real photograph, try [SSD detection](ssd_detection/README.md).

For pinned raw/decoded YOLO layouts, portrait/landscape letterboxing, class-aware suppression and duplicate validation, try [YOLO detection](yolo_detection/README.md).

For parallel crop detectors, shared-ticket source mapping and duplicate suppression, try [tile detection fusion](detection_tile_fusion/README.md).

For optional appearance association and measured identity switches/misses/fragments, try [detection tracking](detection_tracking/README.md).

For point/box requests, inspectable native masks and exact source-grid shape checks, try [prompted segmentation](prompted_segmentation/README.md).
