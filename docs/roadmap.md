# Roadmap

## Implemented prototype

- Node canvas, typed connections, configurable port visibility, inspector and previews.
- Single and continuous execution, worker-thread scheduling, save/load, trusted plugins.
- Acquisition, basic processing, contour inspection, image and structured result sinks.
- Reusable nested ToolBlocks, public-terminal mapping, editor navigation and JSON definitions.
- Toolbox tree grouped by block type and subtype; monochrome and per-channel histograms.
- Grid-derived planar mapping, lens calibration/undistortion, white/color balance and profile files.
- Typed depth planes, depth-file input, RGB-D pairing and sampling with physical units.
- Base OpenCV conversion, geometry, filters, segmentation, morphology, image math and gradients.
- Component measurements, template matching, portable sample jobs and a headless repeat/report runner.
- Additional contrast/channel/mask tools, corners, Hough lines/circles and distance transform.
- Affine/perspective geometry, pyramids/filtering, FAST/ORB descriptors, matching and image registration.
- Explicit enum dropdowns and expanded algorithm controls, plus a generated parameter reference.
- Separable/box filtering, corner response/refinement, phase correlation, affine estimation, remap and polar fields.
- Circle/rectangle/polygon regions, shared mask selection, selected processing and result filtering.
- Interactive region drawing, resize/move handles, polygon vertex editing and captured-frame previews.
- Nested region union/intersection/difference/XOR/inversion and selected-area/coverage measurements.
- Affine/projective region transforms, image-geometry direction/size checks and transformed inspection jobs.
- Optional port payloads, image/root coordinate provenance, bundled Gaussian pyramids and per-input level selection.
- Three/four-channel split/merge, channel extraction/replacement and explicit pyramid up/down tools.
- Blob/contour/Hough/LSD collections, geometric and Hu descriptors, range/region filtering, stable sorting and ranked overlays.
- Trainable geometric and ORB/SIFT patterns, persisted/importable models, pose outputs and wired trainers.
- Oriented edge bands with thickness, continuity, strength, intensity, contrast and coverage filtering.
- Image-plane poses, feature-derived frames, local pattern/pose offsets and pose-relative inspection regions.
- Interactive pose-local region and mask editors, captured frame pairs and optional-frame Shape mask blocks.

## Point clouds and 3D vision

[Milestone 1](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/1): point-cloud
types, PLY/PCD files, viewing, calibrated RGB-D projection and filtering.
Acceptance scope is tracked in [issue 1](https://github.com/votecoffee/OpenCVDataflowGUI/issues/1).

## Camera frames and robot integration

[Milestone 2](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2): managed
intrinsic/extrinsic models, identifiable grid origins, offline hand-eye calibration,
frame transforms and robot pose/measurement adapters. See
[issue 2](https://github.com/votecoffee/OpenCVDataflowGUI/issues/2) and
[issue 3](https://github.com/votecoffee/OpenCVDataflowGUI/issues/3).

## Vision development

- Calibrated region overlays, physical measurements and nonlinear region remapping for lens distortion.
- More OpenCV tools: multiple template matches, additional descriptors/robust matching and advanced calibration.
- Dynamic parameter binding, richer types, promoted parameters and versioned ToolBlock libraries.
- Compile compatible connected segments into shared native G-API computations.
- Validate OpenCV 5 compatibility, including G-API's move to contrib, before expanding the supported dependency range.
- Consider OpenCV.js/WebAssembly for a future browser viewer/editor; retain the native Python/Qt runtime today.
- Camera enumeration, trigger controls, bounded acquisition and recorded test sets.

## Application runtime

- Separate operator view, recipes, runtime-only packages and headless CLI execution.
- Result routing, inspection history, configurable retention and rotating image sinks.
- Database adapters beyond local SQLite, schema mapping and integration interfaces.
- Plugin packaging/version compatibility, project migrations, undo/redo and recovery.

These are future scope, not claims of existing capability or production readiness.
