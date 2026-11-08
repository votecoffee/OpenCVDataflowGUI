# Roadmap

Use the vision tool priorities to choose the next block or workflow. Editor
reliability, acquisition and production runtime have their own milestones and can
advance alongside the tool work. GitHub issues track acceptance criteria and
completion evidence.

Prioritize reusable inspection workflows, especially image acquisition,
processing, registration, and measurement. Add built-in blocks when their
inputs, parameters, and results fit a graph and recur across jobs. Numeric and
control-flow tools qualify when they support those workflows. Use Python
plugins for specialized operations with job-specific interfaces.

## Short-term focus

Work through older [LearnOpenCV tutorials](https://github.com/spmallick/learnopencv)
that teach reusable OpenCV workflows. Build and run a graph for each selected
topic. Use a known result or a before-and-after measurement to verify it. Add a
reusable block when the workflow exposes a gap. Defer framework-specific
training and workflows that depend primarily on tools outside OpenCV.

Use original or clearly licensed fixtures and credit each tutorial. Keep study
projects under the ignored `examples/private/learnopencv` collection while
upstream reuse permission is unclear. Publish independently authored tool
implementations and tests.

The local study index at `examples/private/learnopencv/README.md` now covers
centroid measurement, convex hulls, Hu shape matching, Hough line and circle
detection, masked inpainting, QR label rectification, temporal median background
estimation, checkerboard lens calibration, ArUco fixture rectification, camera
projection, lens distortion, epipolar correspondence review, and metric stereo
depth, optical flow through occlusion, contour retrieval, masked Otsu, and
tolerant edge-map comparison. A linear filtering study compares smoothing,
sharpening, and the polarity of correlation and convolution at slot edges.
The studies check known geometry, decisions, decoded text, or reconstruction
error against original synthetic references. An adaptive foreground study
compares GSoC and LSBP after learning an empty conveyor fixture.

## Vision tool priorities

| Order | Milestone | Next work | Exit criteria |
| --- | --- | --- | --- |
| 1 | [OpenCV core and baseline 2D](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) | Close common gaps in image processing, geometry, replay, and measurement. Track larger inspection mosaics and long-clip stabilization in the coverage backlog. Sequence and calibrated measurement follow-up is tracked in [#14](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14). | Common inspection workflows run headlessly with examples and validated parameter, pixel-depth, selection and coordinate contracts. Use the binding inventory to evaluate candidate APIs. |
| 2 | [ML and DNN](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) | Add multi-input and batched ONNX execution, model-specific detection and mask decoding, more LightGlue adapters, and reusable classical ML train/predict blocks. | Versioned model and preprocessing fixtures reproduce CPU results. Measure accelerator support on each backend used. |
| 3 | 3D and calibrated geometry | Finish calibration revision management and offline hand-eye work in [milestone 2](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2). Extend stereo, point-cloud interchange, mesh output and bounded reconstruction in [milestone 6](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6). | Held-out targets meet declared geometric tolerances. Recorded sequences demonstrate frame, unit and memory bounds. Live robot pose exchange follows offline geometry and acquisition timestamp validation. |
| 4 | Contrib expansion | Select extra-module tools by workflow value, starting with segmentation, texture, restoration and specialized feature or motion methods from the [coverage backlog](opencv-coverage.md). | Each selected tool is available on the pinned contrib build and ships with a CPU fixture, example and any capability limits. |
| 5 | Application-specific ToolBlocks | Package reusable tray orientation, conveyor pick selection, depth-assisted bin picking, surface and bead inspection, OCR/code grading, color checks and robot-frame measurements. | Each package has documented ports and calibration assumptions, known-good and known-bad validation sets, and repeatable ticket verdicts and reasons. Hardware-dependent acceptance is tracked separately. |

Stage 1 is in progress. Cosine and Fourier transforms, inverse reconstruction,
radial Fourier filtering, custom spectrum masks, complex spectrum products, Fourier correlation,
per-channel intensity LUTs, local and tilted window statistics,
unary intensity transforms, binary image arithmetic, pixel comparison, and
nonzero pixel coordinates are available alongside intensity profiles and
selected image summary statistics. Feature covariance fitting and Mahalanobis
scoring cover joint variation and numeric outlier checks. Principal component
fitting, projection, and reconstruction are available for numeric tables.
Singular value decomposition exposes the spectrum, vector bases, and low-rank
approximations of numeric tables. Real eigendecomposition exposes square-matrix
eigenpairs and checks their residuals.
Image validity checks locate nonfinite and out-of-range samples and emit masks
for downstream measurement. NaN and infinity can be replaced by channel with
configured values inside a mask or region.
Image joins build horizontal or vertical comparison views with per-panel
coordinates and source frames. Image tiling builds repeatable inspection
fixtures from a single reference.
Masked patch placement copies pixels into a canvas and reports the affected
area and coordinate mapping.
Subpixel cropping samples around a fractional match or pose origin with
a patch-to-source transform and a border coverage mask.
Point-set Hough detection reports line votes and support spans from sparse
image coordinates. Raster Hough detection reports the strongest polar lines
and support spans from binary or intensity-weighted edge maps.
Line collection clipping measures visible spans inside rectangular inspection
windows and retains source IDs for downstream sorting and drawing.
Hough circle detection emits fitted radii and native strengths for collection
filtering, ranking, and filled disk masks.
Fitted ellipse and enclosing-circle selections follow measured geometry for
interior and perimeter inspections.
Fixed-side convex contour approximation reports the polygon and area added
over the source hull, including when a configured budget stops reduction early.
Polar image warping straightens circular inspection features and can rewrap
an intact polar grid to its original Cartesian frame.
Iterative phase correlation estimates fractional-pixel motion between two
images and supplies an inverse translation for alignment.
Matched-feature registration estimates translation with outlier rejection,
per-pair residuals and configurable inlier acceptance.
Fixed-point coordinate maps reduce map storage for reusable remapping and
carry the lens correction frame through the packed path.
Checkerboard capture checks measure edge width and exposure before
forwarding accepted calibration images.
Scaled absolute conversion maps signed responses to an 8-bit magnitude image
for display and thresholding.
Channel routing assembles selected channels from one or two images without
changing their sample values.
Weight-map blending combines aligned images with a separate contribution at
each pixel.
Scalar colorization turns distance, response and depth maps into display
images with reusable palette and range controls.
Image transposition swaps axes without interpolation and tracks the
coordinate transform through crops and pyramids.
Labeled distance transforms divide an image into zones by its nearest seeds.
Pixel comparisons and ranges turn signed segmentation labels into masks.
Image norms measure signal magnitude and absolute or relative reference differences
over selected pixels. Seeded synthetic images and additive noise provide
repeatable fixtures across serial and parallel branches. Matrix structure tools
construct scaled identities and custom diagonals, complete symmetric matrices,
and report diagonals and traces.
[Stitch overlapping views](<../examples/custom tools/stitch_image_pair/README.md>)
builds a registered two-view mosaic and emits coverage and source mappings.
[Choose a content-aware seam](<../examples/custom tools/stitch_seam/README.md>)
demonstrates graph-cut selection of individual source samples in an overlap,
with a bounded seam work grid for larger overlaps.
[Stitch an ordered inspection scan](<../examples/custom tools/stitch_scan_sequence/README.md>)
automatically registers adjacent views and checks the fit and coverage of each join.
[Close the loop on an inspection scan](<../examples/custom tools/stitch_scan_loop/README.md>)
demonstrates bounded drift correction when the last view overlaps the first.
[Correct exposure in an inspection scan](<../examples/custom tools/stitch_scan_exposure/README.md>)
demonstrates overlap-based exposure correction and feathering on 16-bit tiles.
DIS optical flow adds a dense motion estimator with native preset and patch
controls, masked outputs, and an offline PCB replay.
[Camera motion stabilization](<../examples/custom tools/camera_stabilization/README.md>)
corrects translation or similarity jitter with bounded sequence memory,
coverage, and a measured synthetic replay.
[Recorded camera stabilization](<../examples/custom tools/recorded_camera_stabilization/README.md>)
measures before and after over the same covered pixels and verifies
recovery after a coverage rejection.
[Check stabilization on a second plane](<../examples/custom tools/stabilization_validation/README.md>)
tracks an independent area from the anchor and rejects accumulated drift.
[Check stabilization on recorded depth variation](<../examples/custom tools/recorded_stabilization_validation/README.md>)
verifies rejection and recovery on depth-separated patches of a recorded scene.
[Smooth a recorded camera path](<../examples/custom tools/offline_camera_stabilization/README.md>)
uses centered look-ahead within finite segments and preserves frame tickets.
The [coverage backlog](opencv-coverage.md) tracks the remaining inspection
workflows. Specialized array formats and additional matrix utilities are
deferred until a concrete workflow needs them.

## Structured-light example candidates

- **Accurate Structured Light Scanning Simulation.** Use one calibrated physical
  capture to replay a pattern sequence, or compare reconstructed depth and a
  point cloud against synthetic ground truth and the source mesh. The project
  provides raw and processed calibration, physical HDR captures, synthetic
  HDR/LDR images, and a Python/C++ pipeline with a Docker/Jupyter demo.
  The authors list CC BY 4.0 for the datasets, with separate licenses for seven
  textured or colored models. [Project, datasets, and license](https://geometryprocessing.github.io/scanner-sim/)
- **3DLF-Scan v1.** Use a small set of turntable views to demonstrate sparse-to-dense
  depth comparison, point-cloud registration, or pose-aware multi-view fusion.
  The collection pairs light-field and structured-light captures with metric
  depth, point clouds, camera poses, and calibration. Mendeley lists CC BY 4.0
  for version 1 (DOI `10.17632/ngvgpsvd8b.1`).
  [Dataset record](https://data.mendeley.com/datasets/ngvgpsvd8b/1)
- **Structured-light APIs and toolkits.** Use OpenCV's Gray-code and sinusoidal
  pattern references to guide generation and decoding examples. Use SLStudio
  as a reference for synchronized camera/projector capture, and SLTK for
  calibration and point-cloud processing. [OpenCV Structured Light API](https://docs.opencv.org/4.13.0/d1/d90/group__structured__light.html) ·
  [SLStudio](https://github.com/jakobwilm/slstudio) · [SLTK](https://github.com/jhdewitt/sltk)

Build on the existing Gray-code workflow. When choosing a fixture, record the
source revision, license, and any changes, and credit the source in an on-canvas
Comment. Confirm redistribution terms for the selected files before bundling a
subset. [Current 3D workflows](vision-3d.md#motion-fusion-and-structured-light)

## Implemented prototype

- Node canvas, typed connections, configurable port visibility, inspector and previews.
- Single and continuous execution, worker-thread scheduling, save/load, trusted plugins.
- Plugin-owned inspector dialogs with validated drafts and frame-aware image snapshots.
- Opt-in modeless plugin dialogs for capturing outputs across graph runs.
- Plugin output subscriptions for preview and capture tools.
- Plugin input captures that follow graph wires and preserve processed-image provenance.
- Reusable image and pyramid-level picker for plugin dialogs.
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
- Ordinary Grayscale/Gaussian blur/Canny defaults and examples; legacy blocks under OpenCV > Advanced > G-API.
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
