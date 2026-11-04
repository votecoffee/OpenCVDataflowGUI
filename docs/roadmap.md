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

- ROI editing and overlays with calibrated coordinates and measurement tools.
- More OpenCV tools: multi-match/rotated matching, feature detection and advanced calibration.
- Dynamic parameter binding, richer types, promoted parameters and versioned ToolBlock libraries.
- Compile compatible connected segments into shared native G-API computations.
- Camera enumeration, trigger controls, bounded acquisition and recorded test sets.

## Application runtime

- Separate operator view, recipes, runtime-only packages and headless CLI execution.
- Result routing, inspection history, configurable retention and rotating image sinks.
- Database adapters beyond local SQLite, schema mapping and integration interfaces.
- Plugin packaging/version compatibility, project migrations, undo/redo and recovery.

These are future scope, not claims of existing capability or production readiness.
