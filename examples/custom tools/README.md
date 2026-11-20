# Custom tools

This category demonstrates project-defined algorithms, inspection/metrology
tools and integrations with libraries other than OpenCV or scikit-image.
Examples include flat-field/reference-shadow models, contrast and slanted-edge
measurements, trained shape patterns, region construction/transforms, controlled
distortion fixtures, LightGlue and Annoy. Check each README for dependencies.

The [complete custom-tool index](../README.md#custom-tools) lists these examples.
Native OpenCV operations and module workflows are under
[OpenCV tools](<../opencv tools/README.md>), while scikit-image examples are under
[skimage tools](<../skimage tools/README.md>). Guided editor/runtime tours are
under [Demos](../demos/README.md). Python plugin authoring examples remain in
[user_tools](../README.md#user-tools-and-plugins), and Simple Plugins keep their
[dedicated collection](<../Simple Plugins/README.md>).

[All examples](../README.md)

For external descriptor tensors, learned matching and explicit detector limits,
try [local features](local_features/README.md). The [SuperPoint v6 projects](superpoint_v6/README.md)
show trained extraction and planar matching with a known shift.

For real learned extraction, known transforms and separate PCB photographs,
try [DISK and LightGlue](disk_matching/README.md).

For ALIKED presets, native subpixel/fallback controls and matching local weights,
try [ALIKED and LightGlue](aliked_matching/README.md).

For image measurements on a shared pixel or physical grid, try [profile resampling](profile_resampling/README.md).
For conditional motion-kernel estimation, separate validation and gated restoration, try [motion blur estimation](motion_blur_estimation/README.md).

[Point-cloud sampling](cloud_sampling/README.md) compares coverage at an
equal point budget and preserves the source cloud's row IDs.

[Point-cloud octree queries](cloud_octree_query/README.md) check nearest and
radius results against known coordinates in a colored PLY.

[Time-of-flight depth cleanup](tof_depth_cleanup/README.md) flags low-confidence,
mixed-edge and isolated depth samples while preserving the raw exposure and true
depth step.
