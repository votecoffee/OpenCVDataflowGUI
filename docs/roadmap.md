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

Application-facing [Demos](../examples/demos/README.md), applications and
industrial integration examples should prefer openly licensed real captures.
Keep source/creator/license/hash/change records and on-canvas attribution.
Use controlled synthetic inputs for focused tool examples and exact regression
fixtures. Image-dependent robot or conveyor claims need the corresponding
pose, calibration or encoder evidence. See the [example data policy](../examples/README.md#image-policy).

## Validation cadence

Use targeted tests and a runnable example for each implementation increment.
Run a broader regression check after roughly 20 increments, or earlier when a
specific concern involves shared execution, persistence or plugin contracts.
The last broad checkpoint is `43afa2e` (S1-139): **2,859 passed and 13
skipped** in the offscreen suite. It completes the 20-increment cycle following
`0c8c6e9`. Skips cover 11 unavailable optional-plugin checks and two native
OpenEXR round trips; those paths remain unqualified on this environment.
The current cycle has **19 targeted increments of about 20** completed:
the previously recorded eleven, followed by the expanded block editor,
masked photo correction (S1-88), masked color editing (S1-89),
contrast-preserving decolorization (S1-90), dense-flow warping (S1-95),
color-distribution region tracking (S1-108), variational flow refinement (S1-109),
and image metadata round trips (S1-07).
The editor increment passed 16 focused Qt
checks. Example reorganizations and frozen third-party reviews do not add
vision capability completions.
Update the count as implementation continues and reset it after the next
broader check.

## Short-term focus

The requested [scikit-image inspection expansion](scikit-image.md) adds the
44-property standard 2D regionprops catalog, custom collection metric filters
and sorts, Multi-Otsu, seven global and five local threshold methods. Two original
public examples cover measurements and 16-bit thresholding. These are separate
from the OpenCV coverage denominator; existing OpenCV counts do not change.
Validation: 90 focused tests passed, including native-property/threshold parity,
known geometry and material-band truth, collection interoperability, XML presets
and contained errors. Both public examples also passed installed CLI runs in
Debug and Lean profiles. No broad regression run was needed for this increment.

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
compares GSoC and LSBP after learning an empty conveyor fixture. A resize
study checks letterbox padding, source-coordinate round trips, and the
effect of interpolation on categorical masks. A rotation and translation
study compares clipped and fitted canvases and follows a region through the
combined warp. An annotation study draws named geometry and a measurement-driven
label over a resized inspection image and compares two layer selections.
A GrabCut study uses editable seed masks to isolate a label from a similar
background patch and checks both segmentation branches against a known outline.
The follow-up finds the label corners and rectifies the original image
against a known destination size and a held-out landmark. A two-view
study compares automatic edge-based sizing with a fixed inspection grid.
The BRISQUE study compares sharp, blurred, and compressed captures of the
same surface with the OpenCV sample model. A frame hashing study
compares an exact repeat and a resampled repeat with a changed fixture.
An HDR study compares Debevec and Robertson against a synthetic radiance
reference and an assumed-linear baseline. A focus study compares regional
and whole-frame sharpness when a part blurs against a sharp fixture.
A local threshold study scores four neighborhood formulas and Gaussian
adaptive thresholding against a known mark mask under uneven illumination.
Superpixel studies compare boundary recall across SLICO region sizes and
across SLICO, LSC, and SEEDS at similar output region counts. A label
measurement study checks the area and centroid of segmented coating regions.
The follow-up converts the retained label back to its exact mask and counts
the two disconnected marks. A texture study compares local binary pattern
histograms for an unchanged surface under different brightness and a worn surface. A
co-occurrence study detects a rotated brushed pattern with the same intensity histogram.
A ridge study checks dark and bright crack-like marks on a synthetic surface. A
multiscale follow-up measures how a single-scale filter misses wider marks. A
mask clearance study distinguishes missing material from a narrow section
along a nominal bead centerline. A path continuity study distinguishes short,
scattered defects from a long gap, including across a closed path's start. A
path strip study straightens a curved bead and measures foreground width
across the remapped strip. A follow-up checks width tolerance, missing material,
and clipped edges with distinct ticket reasons. A source review study projects
the strip-width annotations back onto the original curved bead. A defect
report groups adjacent failed columns by type and estimates their length.
A recipe study applies per-kind count and length limits to the complete report.
A calibrated follow-up rectifies a perspective view and applies millimeter limits.
A held-out fiducial check gates the calibrated inspection if the camera shifts.
A fit study uses disjoint ArUco IDs for estimating and verifying a planar mapping.
A part study checks physical rectangle dimensions and orientation after rectification.
A spacing study measures two detected centers on the original camera image.
A bore study checks ellipse diameters, axis difference and fit error in millimeters.
A planar mosaic study solves four point pairs and reconstructs an independent
16-bit inspection plane exactly, with a separate no-overlap failure project.
A camera-jitter study checks a twelve-frame centered path smoother against
known translations and an independent landmark, preserves coverage masks and
replays every original filename ticket. Both add attributed local examples using
existing tools; they do not increase completed function coverage.
A HOG/SVM study now trains, saves and reloads a native three-class model before
classifying disjoint printed-digit images. Its new reusable C-SVC blocks preserve
class mappings and descriptor contracts; a same-length preprocessing mismatch
withholds predictions. This closes S2-20 without claiming general OCR accuracy.

## Vision tool priorities

The [image, video and point-cloud coverage checklist](vision-coverage-checklist.md) defines
the implementation scope and completion counts for stages 1–4. It assigns a
stable ID to each capability and records its APIs, shipped evidence or exit
check, and status. Tiled mosaics now render bounded windows from registered
layouts. Continuous centered stabilization now runs across long streams with
bounded look-ahead. Line-pair metrology now covers corners, angles and gaps.
Slanted-edge sharpness now provides selected ESF/SFR reports and explicit limits.
Aligned image comparison now covers selected ECC/normalized correlation and
constant-region policy. Signed Laplacian bands now support odd-size reconstruction
and multiscale edits. Floating TIFF and Radiance HDR round trips are qualified;
S1-125 remains partial until native EXR runs on a capable build. Prioritize
native QR encoding (S1-112).
[Image metadata round trips](<../examples/opencv tools/metadata_roundtrip/README.md>)
(S1-07) now preserve bounded EXIF/XMP/ICC/CICP packets through qualified PNG
and JPEG buffer/file APIs. Existing encoder, decoder, file source and writer
gain optional metadata ports; a fresh-metadata tool creates common tags with
orientation dropdowns. Canonical chunk contents survive native readback before
buffers or files are published. Existing files remain protected. All eight
orientation transforms preserve source coordinate/ticket semantics; raw chunks
after orientation are read-only and a fresh-metadata project verifies one rotation.
Validation: 80 targeted checks passed; two existing native EXR checks remain
skipped on this build. Four original public projects run through installed
Debug/Lean CLI checks, including intentional unsupported JPEG CICP. RGBA16
pixels, native file/buffer parity, repeated XMP framing, ICC compatibility,
bounded/invalid data, XML presets and contained failures are qualified.
Opaque camera tags are not generally rewritten after processing.
[Variational flow refinement](<../examples/opencv tools/flow_refinement/README.md>)
(S1-109) now provides a paired-image block with all seven native controls and an
explicit copied A-to-B seed. A (+2,+1)-pixel translation checks endpoint error
against independent truth and checks residual reduction on common support.
Selections restrict output writes/statistics while the native energy uses the
full frame. Trusted source/donor masks, border exclusions and optional residual
and supported-fraction gates expose invalid sampling; failed quality gates
withhold the field while preserving diagnostics. Source ticket/grid/pyramid
ownership and both pair identities feed the existing forward-warp block.
Validation: 49 targeted checks, including nine existing dense-flow consistency
regressions, and installed Debug/Lean runs of all three projects. Native solver
failures and invalid fields are contained. This single-scale baseline needs a
suitable seed and does not qualify visibility or captured-scene accuracy.
[Color-distribution region tracking](<../examples/opencv tools/color_region_tracking/README.md>)
(S1-108) now exposes MeanShift/CamShift with explicit shape/box teaching,
hue/HS histogram controls, native termination and qualified color/work-area support.
Translation and growing-ellipse references check independent centers and scale;
loss withholds primary events and expiry requires explicit reseeding. Accepted
updates alone can learn. Stream/context memory is bounded and copyable, with
current-ticket geometry, optional histogram export and explicit axial ambiguity.
CamShift sizes measure color mass, not physical part dimensions or identity.
Validation: 45 focused checks, 60 existing sequence/ticket regressions and installed
Debug/Lean runs of all three projects. A shared state correction keeps explicitly
failed blocks red when their primary outputs are withheld; successful warmup stays
waiting. No broad regression suite was repeated.
[Dense-flow warping](<../examples/opencv tools/dense_flow_warp/README.md>) (S1-95)
now distinguishes pull and forward fields, bounds inverse iterations/residuals,
and returns invalid, uncovered and reverse-disagreement evidence separately.
An original RGB16 translation has 11,439 exact donor pixels; a companion injects
20 invalid and 216 inconsistent vectors, and a coverage-gate project withholds
the image intentionally. Source acquisition tickets remain primary while an
explicit target or known pull grid supplies output coordinates/pyramid metadata.
Donor maps reuse the generic Remap block. Existing Farneback flow, strict .flo
loading, interpolation/border controls, source selections and graphics-off are
qualified. Global invertibility and physical occlusion are not established.
Validation: 47 focused checks and installed Debug/Lean runs of all three projects;
no broad regression suite was repeated.
[Masked color editing](<../examples/opencv tools/masked_color_change/README.md>)
(S1-89) now exposes all three native gradient multipliers. An independent
colored-step reference checks interior BGR values and unchanged boundaries;
unselected samples, alpha, image grids and ticket payloads remain intact.
[Contrast-preserving decolorization](<../examples/opencv tools/contrast_preserving_gray/README.md>)
(S1-90) now exposes native grayscale and color-boost images alongside the
ordinary baseline. Controlled equal-gray patches separate by 77 DN with full-image
fitting context; selected writes preserve the defined backgrounds. A native
two-color-only collapse fails explicitly and withholds processed images.
These are content-dependent appearance edits, not colorimetric calibration.
Validation: 78 focused checks and installed Debug/Lean replay of all three
projects cover native parity, independent color/contrast references, alpha/order,
selection holes and frames, no-ops, failed mapping containment, graphics-off and XML.
No broad suite was repeated.
[Masked photo correction](<../examples/opencv tools/masked_photo_correction/README.md>)
(S1-88) now exposes native illumination and texture controls, intersected
mask/region selection, gradient/change maps and source ticket/frame ownership.
An original surface reduces broad variation and weak texture while retaining
the known boundary position. Its report and tests make the accompanying
contrast loss explicit. Unselected samples and straight alpha remain exact;
thin/empty selections are no-ops, and unsupported depth, grids and allocations
are rejected before the native solve. The supported contract is uint8 color,
not radiometric calibration or clipped-glare recovery.
Validation: 49 focused checks and installed Debug/Lean runs passed, including
independent surface measurements, native parity, XML controls and graphics-off.
No broad suite was repeated.
[Lateral chromatic correction](chromatic-aberration.md) (S1-146) now applies
loaded or explicitly defined camera/lens polynomials with conservative support,
depth/order/alpha retention and green-grid ticket/frame ownership. An original
16-bit RGB fixture verifies exact supported pixels and a failed support gate;
real-lens coefficient fitting remains a captured-calibration step. Validation
includes 110 focused checks and installed Debug/Lean runs of both successful
graphs and an expected support-limit failure. No broad suite was repeated.
Reference-guided hard
shadows (S1-145) now use qualified regional gains, preserve defect samples and
expose masks and failure evidence. An original clear/cast-shadow replay restores
independent 16-bit RGB pixels exactly; inconsistent attenuation withholds the
corrected image. The model assumes aligned, neutral, region-constant attenuation.
Unicode annotations
(S1-144) now use FreeType coverage on the original image sample depth, preserve
alpha and frame/ticket ownership, and support configurable project fonts. An
original public replay qualifies Latin, Greek and Cyrillic in Debug and Lean;
complex scripts require an appropriate font and Raqm-capable Pillow build.
Two-region contrast
(S1-143) now measures explicit signal/background statistics and ratios, preserves
numeric scale and channel semantics, and distinguishes invalid evidence from
limit failures. Original color examples verify expected strong/weak outcomes.
Controlled distortion
fixtures (S1-142) now provide independent donor probes, seeded geometric stress,
conservative support and typed reusable maps. Native float64 cubic boundary
corruption is guarded; fixed-map sampling is qualified with explicit quantization.
Foreground component debounce
(S1-141) now confirms sustained moving masks through mutually unique spatial/area
gates, restarts ambiguous or missing observations, and retains bounded current-frame
state. Independent replay masks verify transient suppression and reconfirmation.
Controlled video encoder qualification (S1-140) now verifies six finalized lossless/lossy
streams, reports actual backends and guards known OpenCV MJPEG grayscale and
fractional-rate failures. Log-polar similarity
registration (S1-139) now verifies rotation/scale/translation in image space,
rejects distinct near-equal poses and preserves original source/target contexts.
Focus stacking (S1-138) now
provides hard/weighted fusion, original level indices, captured-neighborhood
coverage and explicit ambiguity fallback, with independently checked upstream
registration examples. Stack-built flat-field
references (S1-137) now cover bright/dark mean and median estimates, usable-pixel
and noise qualification, raw floating reference units and independent held-out
correction. Binary mask comparison
(S1-136) now measures directed boundary distances and missing/extra area,
with explicit tolerance, selection, empty-set and current-grid contracts.
Phase/magnitude reconstruction
(S1-135) now covers signed grayscale/color images, row/2D transforms, explicit
phase floors and normalization, source windows and zero-phase coordinate grids.
Geometric circle/arc fitting
(S1-134) now reports stable radial consensus, ordered angular extent and
conditional uncertainty, with explicit short-arc and conditioning gates.
Contour rasterization (S1-133)
now preserves selected holes/islands, unions independent roots and validates
source grids against exact pixel references. Source-over alpha
composition (S1-132) now handles straight/premultiplied association, color/depth
conversion, selections and transparent-boundary validation. Intensity-weighted moments
(S1-131) now compare centers/covariance against geometric support and export
frame-aware poses, with independent center/axis validity. Paired-frame
interpolation (S1-130) now validates motion support against independent image
truth and carries explicit media time and source lineage. Structure tensors
(S1-129) now expose selected eigenvalue, coherence and orientation maps with
explicit validity and verified ambiguity handling. Affine SIFT/ORB
(S1-128) now preserve source-space support geometry and verify tilted-view
correspondences against independent truth. Lossless PNG/JP2
(S1-126) preserve exact uint16 samples, while MIPI RAW10/12 (S1-127)
now decode padded frames and retain Bayer phase through integer geometry.
The checklist separates independently testable functions and links each capability to the [OpenCV scope map](opencv-scope-map.md).
Its subgroup audit records which inspection workflows need a different build or
a tested fallback. Point clouds include imported scans and reconstructed surfaces.
The [native point-cloud API crosswalk](opencv-scope-map.md#native-point-cloud-api-crosswalk)
groups exposed APIs and C++-only routines by their graph contracts. The
[native imaging, calibration and video API crosswalk](opencv-scope-map.md#native-imaging-calibration-and-video-api-crosswalk)
adds named entry points for codecs, photo, video I/O, tracking, calibration and
stereo. The [raster, feature, detection and camera geometry crosswalk](opencv-scope-map.md#native-raster-feature-detection-and-camera-geometry-api-crosswalk)
maps another 254 native names to graph workflows. The
[DNN and stitching crosswalk](opencv-scope-map.md#native-dnn-and-stitching-api-crosswalk)
maps 151 more. The [core and contrib crosswalk](opencv-scope-map.md#native-core-and-selected-contrib-api-crosswalk)
covers image arithmetic, extended filtering and features, flow, image quality,
stereo, structured light and RGB-D. The
[reconstruction, registration and stabilization crosswalk](opencv-scope-map.md#native-reconstruction-registration-and-stabilization-api-crosswalk)
covers five further contrib families. The
[recognition and restoration crosswalk](opencv-scope-map.md#native-ocr-shape-hashing-foreground-saliency-and-super-resolution-crosswalk)
covers OCR, shape, hashing, foreground, saliency and both super-resolution
modules. The [image ML, tracking, detection and restoration crosswalk](opencv-scope-map.md#native-image-ml-tracking-detection-and-restoration-crosswalk)
adds classical image learning, contrib tracking and detectors, and additional
restoration tools. The [color calibration and profile crosswalk](opencv-scope-map.md#native-color-calibration-and-image-profile-resampling-crosswalk)
covers camera color profiles and image-derived sample grids. The
[metrology and residual API review](opencv-scope-map.md#inspection-metrology-and-residual-image-api-review)
links remaining image API variants and adds line-pair, spatial-axis and
calibration-evidence workflows. The [camera model compatibility audit](opencv-scope-map.md#camera-model-compatibility-audit)
records wide-angle object pose and independently calibrated relative pose as
extensions of the shipped pinhole paths, and lists the multiview triangulation checks.
Use the checklist as the implementation baseline and add scope for independently
testable inspection workflows.

| Order | Milestone | Next work | Exit criteria |
| --- | --- | --- | --- |
| 1 | [OpenCV core and baseline 2D](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) | Close common gaps in image processing, geometry, replay, and measurement. Prioritize native QR encoding (S1-112). S1-07 metadata file/buffer round trips now have verified native chunk transport and explicit orientation semantics. S1-109 paired flow refinement now has independent motion/residual references and explicit support/failure evidence. S1-108 color-distribution tracking now has controlled translation/scale/loss evidence. S1-95 dense-flow warping and S1-88–S1-90 photo/color edits have controlled references and explicit support/failure evidence. Native EXR qualification for S1-125 remains open because this build has no EXR codec; then follow the remaining coverage checklist. Sequence and calibrated measurement follow-up is tracked in [#14](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14). | Common inspection workflows run headlessly with examples and validated parameter, pixel-depth, selection and coordinate contracts. Use the binding inventory to evaluate candidate APIs. |
| 2 | [ML and DNN](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) | Add multi-input and batched ONNX execution, model-specific detection and mask decoding, more LightGlue adapters, and K-nearest/random-forest image classifiers. Native SVM training, prediction and model I/O are available in [the classical ML workflow](classical-ml.md) (S2-20). | Versioned model and preprocessing fixtures reproduce CPU results. Measure accelerator support on each backend used. |
| 3 | 3D and calibrated geometry | Finish calibration revision management and offline hand-eye work in [milestone 2](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2). Extend stereo and prioritize cloud interoperability, cleanup, segmentation, comparison, mesh output and bounded reconstruction in [milestone 6](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6). | Held-out targets meet declared geometric tolerances. Recorded sequences demonstrate frame, unit and memory bounds. Live robot pose exchange follows offline geometry and acquisition timestamp validation. |
| 4 | Contrib expansion | Select extra-module tools by workflow value, starting with segmentation, texture, restoration and specialized feature or motion methods from the [coverage backlog](opencv-coverage.md). | Each selected tool has a qualified Python path on a supported build and ships with a CPU fixture, example and documented limits. |
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
[Stitch overlapping views](<../examples/opencv tools/stitch_image_pair/README.md>)
builds a registered two-view mosaic and emits coverage and source mappings.
[Choose a content-aware seam](<../examples/opencv tools/stitch_seam/README.md>)
demonstrates graph-cut selection of individual source samples in an overlap,
with a bounded seam work grid for larger overlaps.
[Stitch an ordered inspection scan](<../examples/opencv tools/stitch_scan_sequence/README.md>)
automatically registers adjacent views and checks the fit and coverage of each join.
[Tiled inspection mosaics](<../examples/opencv tools/tiled_scan_mosaic/README.md>)
separate registration from rendering, accept up to 64 bounded source views, and
produce selected windows with coverage and source-coordinate metadata.
[Close the loop on an inspection scan](<../examples/opencv tools/stitch_scan_loop/README.md>)
demonstrates bounded drift correction when the last view overlaps the first.
[Correct exposure in an inspection scan](<../examples/opencv tools/stitch_scan_exposure/README.md>)
demonstrates overlap-based exposure correction and feathering on 16-bit tiles.
DIS optical flow adds a dense motion estimator with native preset and patch
controls, masked outputs, and an offline PCB replay.
[Camera motion stabilization](<../examples/opencv tools/camera_stabilization/README.md>)
corrects translation or similarity jitter with bounded sequence memory,
coverage, and a measured synthetic replay.
[Recorded camera stabilization](<../examples/opencv tools/recorded_camera_stabilization/README.md>)
measures before and after over the same covered pixels and verifies
recovery after a coverage rejection.
[Check stabilization on a second plane](<../examples/opencv tools/stabilization_validation/README.md>)
tracks an independent area from the anchor and rejects accumulated drift.
[Check stabilization on recorded depth variation](<../examples/opencv tools/recorded_stabilization_validation/README.md>)
verifies rejection and recovery on depth-separated patches of a recorded scene.
[Smooth a recorded camera path](<../examples/opencv tools/offline_camera_stabilization/README.md>)
uses centered look-ahead within finite segments and preserves frame tickets.
[Continuous centered stabilization](<../examples/opencv tools/continuous_camera_stabilization/README.md>)
extends the same smoothing to a bounded rolling window, retains each frame's
ticket and timestamp, and flushes the remaining tail when playback ends.
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
- Single-call plugin captures of retained image, cloud and data inputs from one completed execution.
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
- Topology-aware contour/blob rasterization with ID/index selection, subtree holes/islands and binary source-grid masks.
- Geometric circle/arc fits with robust consensus, ordered angles, radial residuals and conditional uncertainty.
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
