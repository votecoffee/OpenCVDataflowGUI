# Roadmap

Use the vision tool priorities to choose the next block or workflow. Editor
reliability, acquisition and production runtime have their own milestones and can
advance alongside the tool work. GitHub issues track acceptance criteria and
completion evidence.

## Vision tool priorities

| Order | Milestone | Next work | Exit criteria |
| --- | --- | --- | --- |
| 1 | [OpenCV core and baseline 2D](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) | Start with OpenCV's literal `core` array operations, including spectral transforms, arithmetic and reductions. Continue with commonly used baseline image processing, geometry, video and image I/O. Sequence and calibrated measurement follow-up is tracked in [#14](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14). | Common 2D workflows run headlessly with examples and validated parameter, pixel-depth, selection and coordinate contracts. Classify the remaining public bindings by workflow, alias, low-level API or build availability. |
| 2 | [ML and DNN](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) | Add multi-input and batched ONNX execution, model-specific detection and mask decoding, more LightGlue adapters, and reusable classical ML train/predict blocks. | Versioned model and preprocessing fixtures reproduce CPU results. Measure accelerator support on each backend used. |
| 3 | 3D and calibrated geometry | Finish calibration revision management and offline hand-eye work in [milestone 2](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2). Extend stereo, point-cloud interchange, mesh output and bounded reconstruction in [milestone 6](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6). | Held-out targets meet declared geometric tolerances. Recorded sequences demonstrate frame, unit and memory bounds. Live robot pose exchange follows offline geometry and acquisition timestamp validation. |
| 4 | Contrib expansion | Select extra-module tools by workflow value, starting with segmentation, texture, restoration and specialized feature or motion methods from the [coverage backlog](opencv-coverage.md). | Each selected tool is available on the pinned contrib build and ships with a CPU fixture, example and any capability limits. |
| 5 | Application-specific ToolBlocks | Package reusable tray orientation, conveyor pick selection, depth-assisted bin picking, surface and bead inspection, OCR/code grading, color checks and robot-frame measurements. | Each package has documented ports and calibration assumptions, known-good and known-bad validation sets, and repeatable ticket verdicts and reasons. Hardware-dependent acceptance is tracked separately. |

Stage 1 is in progress. Cosine and Fourier transforms, inverse reconstruction,
radial Fourier filtering, custom spectrum masks, Fourier correlation,
per-channel intensity LUTs, local and tilted window statistics,
unary intensity transforms, binary image arithmetic, pixel comparison, and
nonzero pixel coordinates are available alongside intensity profiles and
selected image summary statistics. Feature covariance fitting and Mahalanobis
scoring cover joint variation and numeric outlier checks. Principal component
fitting, projection, and reconstruction are available for numeric tables.
Specialized array operations remain open. The
[coverage backlog](opencv-coverage.md) tracks the missing workflows and the
[binding inventory](opencv-bindings.tsv) maps static references to the pinned
build. Coverage is measured by runnable graph use cases. Aliases, alternate
overloads and low-level infrastructure may be covered through an existing block
or have no graph-facing use case.

## Implemented prototype

- Node canvas, typed connections, configurable port visibility, inspector and previews.
- Single and continuous execution, worker-thread scheduling, save/load, trusted plugins.
- Acquisition, basic processing, contour inspection, image and structured result sinks.
- Reusable nested ToolBlocks, public-terminal mapping, editor navigation and JSON definitions.
- Toolbox tree grouped by block type and subtype; monochrome and per-channel histograms.
- Grid-derived planar mapping, lens calibration/undistortion, white/color balance and profile files.
- Typed depth planes, depth-file input, RGB-D pairing and sampling with physical units.
- OpenCV 5 camera/pose geometry, stereo, RGB-D registration and organized normals,
  typed point clouds, PLY/PCD files, CPU viewer, ICP/PPF, ArUco and Gray-code tools.
- Two-frame RGB-D odometry and bounded posed TSDF fusion using OpenCV 5 ptcloud.
  See [3D workflows](vision-3d.md) for tested scope and build limits.
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
- OpenCV 5 contrib runtime, retained native G-API nodes and normalized line-detector vector outputs.
- Acquisition tickets, bounded join/batch/FIFO buffers, serial group expansion, case routing and explicit branch concurrency.
- Parallel per-pose ToolBlock inspections, aggregate outcomes, scoped part tickets and N-pass/N-failure stopping.
- Bounded input-image FIFO queues, serialized frame leases, optional background acquisition and Stop/Resume retention.
- Ordinary Grayscale/Gaussian blur/Canny defaults and examples; legacy blocks under Advanced > G-API.
- Per-connection outcomes, per-input failure gates, bounded shared ticket history,
  explicit final judgments and a live searchable ticket-results window.
- Stop/Resume retains the last completed result, including results held for throttled UI updates.
- Explicit ticket/deadline, latest, trigger-latest and first-arrival input coordination,
  freshness guards and a synthetic shared part-counter example.
- Versioned job recipe catalog, scalar/model outputs, temporary parameter/model bindings,
  queued-frame revision pinning and GUI recipe/training-asset management.
- Grouped workspaces, saved viewport, minimap, typed colors, finite image sets and
  verdict filmstrips.
- Per-block diagnostics, ticket timing, concurrency views, running counts,
  failure reasons, scored validation sets and Debug/Balanced/Lean run profiles.
- ONNX inference, classification/embeddings, SIFT LightGlue and Annoy index search.

The point-cloud baseline shipped in `f15cf75`. It includes typed clouds, retained
pixel correspondence, calibrated projection, filtering, PLY/PCD interchange and
the viewer. [Completed milestone](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/1),
[acceptance issue](https://github.com/votecoffee/OpenCVDataflowGUI/issues/1).

## Application milestones

| Milestone | Work | Exit criteria and dependencies |
| --- | --- | --- |
| [Reliable acquisition and replay](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/3) | Independent camera workers, bounded queues, timestamp-window matching, backend enumeration and triggers, multi-source recordings and validation bundles. [#5](https://github.com/votecoffee/OpenCVDataflowGUI/issues/5), [#6](https://github.com/votecoffee/OpenCVDataflowGUI/issues/6) | A stalled source cannot stall another source or coordination deadlines. Replay reproduces ticket outcomes and failure reasons. Physical synchronization needs separate backend and trigger evidence. |
| [Editor and reusable jobs](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/4) | Undo/redo, autosave recovery, schema migrations, promoted parameters, dynamic bindings and versioned ToolBlock/plugin contracts. [#7](https://github.com/votecoffee/OpenCVDataflowGUI/issues/7), [#8](https://github.com/votecoffee/OpenCVDataflowGUI/issues/8). | Existing and nested jobs round-trip with their layout and interfaces. Interrupted saves can be recovered. Start alongside acquisition. |
| [Production runtime and validation](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/5) | Operator view, deployable jobs, recipe and asset revisions, durable ticket history, result routing and retention. [#9](https://github.com/votecoffee/OpenCVDataflowGUI/issues/9), [#10](https://github.com/votecoffee/OpenCVDataflowGUI/issues/10), [#11](https://github.com/votecoffee/OpenCVDataflowGUI/issues/11) | Operator and headless runs agree on the same package. An unattended replay pilot survives restart and sink failures, preserves decision provenance, and reports latency, throughput and memory. Depends on replay and compatible job files. |
| [Measured execution and browser feasibility](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/8) | Opt-in compiled ToolBlocks and a scoped OpenCV.js/WebAssembly viewer study. [#4](https://github.com/votecoffee/OpenCVDataflowGUI/issues/4), [#16](https://github.com/votecoffee/OpenCVDataflowGUI/issues/16) | Compilation preserves output semantics and selected diagnostics and shows a repeatable end-to-end benefit after amortization. The browser study ends in a go/no-go decision based on module coverage, transport, latency and memory. [Benchmark criteria](opencv-backend.md#compiled-toolblocks-future). |

Start with editor undo/recovery ([#7](https://github.com/votecoffee/OpenCVDataflowGUI/issues/7))
and the shared acquisition/replay contract ([#5](https://github.com/votecoffee/OpenCVDataflowGUI/issues/5)
and [#6](https://github.com/votecoffee/OpenCVDataflowGUI/issues/6)). Current
graph save/load, per-ticket diagnostics, local replay and the headless runner
provide the baseline for these milestones. Operator packaging and durable
ticket history follow the compatible project and replay contracts.
