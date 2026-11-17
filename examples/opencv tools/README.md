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
