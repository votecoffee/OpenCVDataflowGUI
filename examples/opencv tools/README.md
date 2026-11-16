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
For reader/encoder checks and page counts, try [codec preflight](codec_probe/README.md).
For stored displacement and verified image pairs, try [flow file round trips](flow_file_roundtrip/README.md).
For seeded local tracking and explicit loss/reacquisition, try [MIL box tracking](mil_box_tracking/README.md).
For bounded native stream decoding and grouped inspections, try [video buffers](video_buffer/README.md).
For exact binary neighborhoods, try [hit-or-miss patterns](hit_or_miss/README.md).
For sparse-mask extraction and allocation limits, try [run-linked contours](run_contours/README.md).
