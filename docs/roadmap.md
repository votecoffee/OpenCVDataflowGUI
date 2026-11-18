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
The current audit used committed source `9c9d6a7`: **8,034 passed, 36 skipped,
six failed** in 872.37 seconds offscreen (8,076 cases). All **8,844 tracked-file
hashes** and **61 installed-package versions** matched before/after; source and
checkout stayed clean. The expected skips remain **34 opt-in native ROS cases**
and **two unavailable EXR round trips**. Twelve existing scikit-image/NumPy
shape-assignment deprecation warnings remain.

All six failures were example/maintenance checks, subsequently repaired:
31 detector-evaluation, instance and rotated saved graphs acquire the five
new optional SSD-head defaults without changing any existing values or assets;
the SimpleBinPicking README digest follows its committed documentation; the
3D preview test checks every discovered project while retaining its original
coverage floor; and example authoring reads its exact sibling preparation file
instead of a cached same-name helper from another example. Three new isolation
cases exercise that last fix explicitly. All **293 focused rechecks** across the
eight affected suites pass, including all six original failing cases and their
numeric/project/provenance checks. Those scoped repairs changed only examples,
authoring and maintenance tests; no image-processing/runtime code changed in
the repair increment.
The broad invocation was not repeated after those scoped repairs. Its initial
six failures and the successful focused follow-up are reported separately.

This closes the twenty-increment cycle through S2-33. The next targeted cycle
includes constrained OCR decoding, learned single-image super-resolution,
restoration, monocular depth (S2-34 through S2-37), the optional-input plugin
snapshot repair, learned stereo disparity (S2-38) and learned optical flow
(S2-39), learned video tracking (S2-40), image-descriptor PCA (S2-41),
image mixtures (S2-42), image-feature outliers (S2-43) and visual-word retrieval
(S2-44), model-specific landmark skeletons (S2-45) and HOG sliding-window
detection (S2-46), cascade-based image detection (S2-47) and rotated crop
classification (S2-48), image-based regression (S2-49) and local anomaly maps
(S2-50) and input-shift evaluation (S2-51). It is now **19/20**, with learned
instance tracking masks (S2-52) next. The installed wheel matches all
**399 release-source modules**; Python 3.14.3/OpenCV 5.0.0
contrib 5.0.0.93 remain unchanged. This is software/offline evidence, without
native ROS or physical equipment qualification.

The previous broad checkpoint was `2a66fc8`: **6,986 passed, 36 skipped** in
602.21 seconds with the offscreen suite, Python 3.14.3, OpenCV 5.0.0 contrib
5.0.0.93, NumPy 2.5.4 and PySide6 6.12.0. All **6,879 tracked-file hashes**
and **61 installed-package versions** matched before and after. The committed
source stayed unchanged. Concurrent untracked example work was preserved outside
this publication; the whole checkout was therefore no longer clean at the end.
The audit wrapper's clean-checkout assertion flagged that concurrent work; pytest
itself returned zero. No source/package regression was found and no broad rerun
was needed for those untracked files.

The skips are **34 opt-in native ROS cases** and **two unavailable OpenEXR round
trips**. Twelve existing scikit-image/NumPy shape-assignment deprecation warnings
remain. Installed optional-plugin checks ran. This closes the twentieth increment
ending with instance segmentation (S2-11). Ordered landmarks (S2-12) and the
partial external SuperPoint interface (S2-15) advance the targeted cycle to
**20/20**, including qualified DISK (S2-16), ALIKED (S2-17), reusable clusters
(S2-19), K-nearest (S2-21), random forests (S2-22), classifier/detector
evaluation (S2-23), engine-dispatch reliability, pinned SSD/YOLOX exports (S2-25/S2-26)
and paired plugin-review snapshots with a separate read-only review action,
cross-tile box fusion (S2-27), appearance-aware detection tracking (S2-28) and
prompted segmentation (S2-29), tiled segmentation (S2-30) and pinned text
detection (S2-32) and batched region OCR (S2-33). The twentieth targeted increment
and its broad/focused maintenance audit are complete. S2-34 through S2-36 are
qualified below; learned monocular depth (S2-37) is also qualified and learned
stereo disparity (S2-38) and learned optical flow (S2-39) are also qualified;
learned video object tracking (S2-40) and image-descriptor PCA (S2-41) are
also qualified; texture/color mixture classification (S2-42) and image-feature
outlier detection (S2-43) and bag-of-visual-words retrieval (S2-44) are qualified.
Object landmarks with model-specific skeletons (S2-45) are qualified;
HOG sliding-window detection (S2-46) and cascade-based image detection (S2-47)
and rotated crop classification (S2-48) and image-based regression (S2-49) are
qualified; local anomaly maps (S2-50) and input-shift evaluation (S2-51) are
qualified; learned instance tracking masks (S2-52) are next. Permitted
real SuperPoint extraction remains open. This is software/offline qualification,
without native ROS or physical equipment acceptance.

The previous broad checkpoint was `e9b9aaf`: **5,949 passed, 36 skipped** in
349.62 seconds with the offscreen suite, Python 3.14.3, OpenCV 5.0.0 contrib
5.0.0.93, NumPy 2.5.4 and PySide6 6.12.0. All **5,880 tracked-file hashes**
and **61 installed-package versions** matched before and after; the checkout
stayed clean. The skips are **34 opt-in native ROS runtime cases** and **two
unavailable OpenEXR round trips**. Twelve scikit-image/NumPy shape-assignment
deprecation warnings remain. Optional installed-plugin checks ran, including
Simple3DMeasure 0.1.0.dev5, Simple3DLocate 0.1.0.dev19, SimpleBinPicking
0.1.0.dev3, SimplePathWidth 0.1.0.dev12, SimpleClassifier 0.3.1 and
SimpleReferenceInspect 0.1.0.dev3. This is software/offline acceptance, without
native ROS or physical equipment qualification.

The `e9b9aaf` checkpoint's first attempt identified three test failures: catalog fixtures
had not supplied the required TIFF path and camera/reference identities, and a
saved mosaic project lacked newer default fields. The catalog now uses explicit
required settings while separate tests retain rejection of unconfigured sources.
The mosaic project was refreshed without changing its images; regeneration now
runs in a temporary folder, compares normalized project text/exact image bytes,
and verifies the source collection stays untouched. **100 focused rechecks**
passed before that clean full rerun. It closed the cycle ending with S1-173.
The next completed cycle covered **20 targeted increments**: paced file
replay (S1-11), bounded live network acquisition (S1-12), media-time sampling
(S1-110), actual variable/unreliable video timing (S1-123), and whole
selections through nonlinear correction (S1-69), candidate visible support (S1-61),
HDR moving-region rejection (S1-150), unordered rotating views (S1-104), and
explicit camera refinement with reserved observations (S1-147), and review-only
stabilization border fills (S1-148), and controlled motion-kernel estimation (S1-84), and shared capture-setting negotiation/readback (S1-115), and controlled curved-QR decoding (S1-121), and image detection/decoding (S2-08), and complete named multi-input inference (S2-03), and ordered batched/dynamic-shape image inference (S2-04), and revisioned model/preprocessing/execution receipts (S2-05), and rotated detection/affine poses (S2-09), and source-grid semantic segmentation (S2-10), and independent instance masks (S2-11).

An earlier broad checkpoint was `1e1cccb`: **4,801 passed, 31 skipped** in
298.51 seconds with the offscreen suite, Python 3.14.3, OpenCV 5.0.0 and contrib
5.0.0.93. Before/after manifests matched for every tracked file and installed
package version. Installed optional-plugin checks ran, including Simple3DMeasure
0.1.0.dev5, Simple3DLocate 0.1.0.dev19, SimpleBinPicking 0.1.0.dev3,
SimplePathWidth 0.1.0.dev12 and SimpleReferenceInspect 0.1.0.dev3.
The skips are **29 opt-in native ROS runtime tests** and **two unavailable
OpenEXR round trips**. There were 12 scikit-image/NumPy shape-assignment
deprecation warnings in the skeleton fixtures. This checkpoint does not qualify
those skipped native paths or physical equipment.

That earlier checkpoint closed the cycle ending with run-linked contours (S1-152).
The subsequent completed cycle covered **20 targeted increments**: codec
availability/page inventory (S1-153), program lifecycle/managed ROS services,
optical-flow file round trips (S1-154), seeded MIL tracking (S1-156), bounded
encoded-video decoding (S1-157), indexed TIFF page access (S1-158), temporal
pixel statistics (S1-160), ArUco-backed QR reading (S1-161), and structured-append
QR recovery (S1-162), feature evaluation (S1-163), and stable calibrated mosaic
reuse (S1-164), registered-view replay (S1-165), and relative frequency response
(S1-166), saved motion trajectories (S1-167), neighbor restoration (S1-168),
rolling-shutter correction (S1-169), clean background references (S1-170),
image-track prediction (S1-54/S1-171, one increment), and periodic
detector refresh (S1-172), and common-pitch measured image profiles (S1-173).
The new checkpoint above closes this cycle. Program
lifecycle is application infrastructure and does not add OpenCV coverage. The
S1-155 pyramid binding audit records an unresolved consumer limit and adds no
capability completion. After
the frozen run, eight existing test files received syntax-tree/comment-preserving
formatting to resolve 33 line-length findings; their **108 focused checks passed**.
Full source/test Ruff checks were clean. That formatting follow-up did not change
application code or repeat the broad suite.

Previous broad checkpoints were `6cd14f3` (3,689 passed, two EXR skips, ending
with native QR generation and Save As changes) and `43afa2e` (2,859 passed,
13 skipped). Their pinned asset, recipe and persistence checks remain covered.

The subsequent `05d6be3` integration adds eight 3D line examples and a version-2
recipe Save As regression. Seventeen focused line/asset checks passed after that
integration; the broad suite was not repeated for these examples.

The completed 20-increment cycle covered:
animated image codec round trips (S1-111), large-radius smoothing (S1-159),
assistant application authoring with connected-input plugin captures, the
SimpleITK medical image/metadata foundation, optional ROS 2 Lyrical/Fast DDS
image/data integration, ROS PointCloud2 conversion/acquisition/publication,
ROS CameraInfo/per-acquisition registered RGB-D association, strict ROS
pose/TF conversion with run/reset acquisition identity, static CameraInfo reuse,
ordered timestamp pairing of ROS image streams, explicit CameraInfo intensity
rectification, source ROI/binning with independent rectified output grids,
conservative area-coverage mask resizing (S1-118), and paired differential exposure
compensation with defect residuals (S1-116), and frozen-reference camera health
monitoring with foreground suppression and trained fiducials, and independent
connection-health monitors with endpoint diagnostics, and dense/tiled multiband
mosaic composition (S1-117), typed in-memory multipage TIFF transport (S1-149),
signed-kernel binary hit-or-miss morphology (S1-151), and bounded run-linked
contour extraction (S1-152).
The animation increment's **108 targeted checks** cover
file/buffer, timing, format, provenance, XML, editor and saved examples.
All **12 installed Debug/Lean CLI runs** reproduced their expected outcomes,
including the deliberate GIF timing rejection. No broad suite was repeated.
The smoothing increment passed **76 targeted checks** for native/independent
references, data types, selections, payloads, controls, XML, examples and contained
failures. Its benchmark records native-call timing and noise/edge tradeoffs on a
1920x1080 float32 fixture; it excludes graph overhead and makes no speed guarantee.
All **10 installed Debug/Lean CLI runs** reproduced their expected outcomes,
including guarded native high-depth arithmetic. No broad suite was repeated.
The assistant increment passed **132 focused checks** for authoring, file revision
conflicts, replay bounds, nested graphs, plugin validation/captures, editor loading
and existing persistence contracts. All **17 installed CLI outcomes** matched,
including controlled pass/fail inspections under Debug, Balanced and Lean.
Two skill packages passed validation and were installed locally. No broad suite
was repeated and no OpenCV coverage row was added for this application API.
The SimpleITK increment passed **128 focused checks** for optional imports,
metadata/native buffer copies, scalar/vector types, oblique physical points,
OpenCV geometry/pyramids, image I/O, masks, labels, XML, toolbox/editor behavior
and API/path compatibility. All **24 installed Debug/Lean CLI outcomes** matched:
18 successful medical examples and six deliberate Dice-threshold failures.
The six original MRI/label asset hashes, deterministic project regeneration,
strict JSON and documentation links were verified. No broad suite was repeated.
Update the count as implementation continues. Example reorganizations and frozen
third-party reviews do not add vision capability completions.

The ROS 2 increment passed **70 focused offline/editor/industrial checks** and
**11 opt-in native localhost Fast DDS checks**, using the separate Lyrical
Python 3.12.3/rclpy 10.0.10 runtime. Exact color/high-depth/padded/endian image
bytes, structured pose/joint/clock/int64 data, subscriber matching, stale/latest
queues, ticket acquisition, simulated-clock stamping, worker cleanup, cancellation
and ambiguous publication handling were checked. All **eight installed Debug/Lean
CLI runs** matched the four public examples. The application was reinstalled;
no broad regression suite, simulator or robot operation was performed.

The PointCloud2 increment adds four ROS blocks, exact raw-message relay and a
typed XYZ/RGB/normals projection with units, endian, stride, field and finite-value
checks. Source pixel organization survives dropped points; canonical publication
uses meters and raw publication preserves all fields/bytes. Three additional
examples reuse real T-LESS RGB-D data and attribution. Focused verification covers
independent binary layouts, all eleven Lyrical field types, organized NaN holes,
raw packed-color NaN bits, ticket/payload/XML/editor behavior and native DDS
exchange. Point-cloud extras remain available in raw packets; remerging extras
after arbitrary point edits is a separate future feature.
Validation passed **114 focused offline/editor/industrial checks** and **five
native cloud checks**. The **eleven existing native DDS checks** also passed after
the worker extension. All **14 installed Debug/Lean CLI runs** matched the seven
public ROS examples, including the real T-LESS organized round trip. Source
assets and the original four examples stayed unchanged; text contracts, generated
controls, local links and Python 3.12 worker syntax were checked. The app was
reinstalled; no broad suite or robot/simulator operation was performed.

## Short-term focus

**Inspection input-shift evaluation (S2-51)** adds three reusable blocks for
controlled exposure/blur/affine views, current inspection observations and paired
robustness reports. Original tickets, pixel/truth/support hashes, known transforms
and spatial/pyramid maps remain explicit. Synthetic borders are excluded from both
inspection and evaluation. Model/decision revisions, exact variant membership and
truth/shared-support retention are checked before comparisons.

[Eleven original MIT projects](<../examples/opencv tools/input_shift_evaluation/README.md>)
reuse the raw-patch ONNX/reference controls. Stable dark-defect and uint16 pairs
retain **64-pixel truth** and **IoU 1.0** after a two-pixel pose shift and small
exposure change. Three sensitive clean controls reveal exposure, blur and odd-phase
pose false rejects: **50% overall**, **100% for the shifted view**. A loose threshold
has **100% false accepts** on labelled defects. Missing variants, mixed valid
reference revisions and cropped-away truth are explicit errors. These paired
synthetic views are correlated and are not independent physical camera captures.

Reports retain raw scores/deltas or explicitly declared probabilities, judgment
changes, source-grid truth IoU and mapped-baseline agreement on shared support.
Empty IoU unions and absent label populations remain null. Per-variant rates and
means separate sensitivity from the baseline. Independent NumPy translation,
separable Gaussian, exhaustive patch distances and mask arithmetic match graph
results within **3e-7**; native ONNX patch extraction agrees exactly.
All **10 asset pins** regenerate exactly.

**139 focused checks** and **59 installed checks** pass, including 90-degree pose
truth/pixels, high depth, parameter/identity/retention/budget guards, Debug/Lean,
JSON/XML/Save As, failed-input history and cold Run All teardown. All **399
source/wheel/installed modules** match; **61 package versions** remain unchanged.
ML/DNN is **50/62 (80.6%)**, overall **276/470 (58.7%)**, core **173/175 (98.9%)**.
Targeted cycle **19/20**, without a broad repeat. Next: **S2-52 learned instance
tracking masks**. See the [contract and qualification limits](input-shifts.md).

**Local image-model anomaly maps (S2-50)** add five blocks for explicit ONNX
feature grids, known-good reference training, source-grid inspection and sealed
reference save/load. NCHW/NHWC cell geometry, normalization, model/preprocessing
revisions and source pixel/ticket/frame identity are checked. Native Euclidean or
Manhattan nearest-reference search has explicit work/memory bounds; source-grid
distance, support and anomaly masks remain separate. Inclusive distance/area
decisions retain reference/threshold revisions without inventing confidence.

[Twelve original MIT projects](<../examples/opencv tools/local_anomaly/README.md>)
use a **268-byte** ONNX patch encoder and **768 reference vectors** from three
known-good synthetic surfaces. Held-out dark and missing-texture defects localize
to exactly **64 pixels**, matching independent truth. The normal maximum distance
is about **0.00555**. A weak one-pixel defect at **0.1098** is deliberately missed
by threshold 0.2, while a structural-defect-free brightness shift rejects all
**1024 pixels**. These raw pixel embeddings are controlled integration evidence,
not pretrained industrial-backbone accuracy or physical acceptance. Global bank
matching can miss rearranged known textures. Even-stride reduction is explicit;
it is not a representative coreset or a PatchCore reproduction.

Independent native ONNX patch extraction agrees exactly with NumPy, and native
distances match exhaustive float64 references within **2e-7**. All **15 asset
pins** regenerate exactly. **202 focused checks** pass, including Debug/Lean,
layouts, padding/selection, identities, model I/O, limits, high depth, failure
controls, JSON/XML/Save As, failed-input gating and cold Run All shutdown.
**58 installed checks** pass; **398 source/wheel/installed modules** match, with
**61 package versions** unchanged. ML/DNN is **49/62 (79.0%)**, overall **275/470
(58.5%)**, core **173/175 (98.9%)**. Targeted cycle **18/20**, no broad repeat.
Next: **S2-51 image-model evaluation under input shifts**. See the
[contract and limits](local-anomaly.md).

**Image-based regression (S2-49)** adds `ml.decode_regression` and
`ml.regression_metrics` around existing inference/receipts. Explicit vector or
NCHW scalar heads become named continuous values with units and scale/offset;
references yield signed residuals and optional value/error judgments. Model and
decoder/preprocessing revisions, source image/ticket/frame context and batch-row
identity remain recorded. Detailed single-image tensor hashes are checked; source
pixel hashes and capture independence are explicitly unverified. Aggregate reports
retain all residuals, MAE/RMSE/bias/max error and R², with unit/revision/dedup guards.

[Fifteen original MIT projects](<../examples/opencv tools/image_regression/README.md>)
fit seven synthetic widths with a native linear EPS-SVR and export a **298-byte**
ONNX mean-intensity width proxy. Graph-native SVR training is not part of this
increment. Three disjoint held-out widths have **MAE 0.000594 mm**, **RMSE 0.000710
mm**, maximum error 0.001065 mm under an assumed 0.1 mm/pixel scale. The dim control
has residual **-0.46366 mm**; the corresponding report has **MAE 0.1551 mm** and
fails. This exposes the trained lighting/area assumption; it is not physical
calibration, geometric bead measurement or production accuracy evidence.

Independent native SVR/ONNX and NumPy calculations agree within **2e-5** output
units for ONNX and **2e-7** for SVR. All **30 asset pins** reproduce exactly.
**184 focused checks** pass across decoding, receipts, reference/identity gates,
metrics, existing inference/batch/model/catalog contracts, Debug/Lean, high depth,
valid failures and invalid heads/models, row receipts, JSON/XML/Save As, failed
inputs and cold one-/three-image Run All teardown. **67 installed checks** pass;
all **397 release-source/wheel/installed modules** match, with **61 package versions**
unchanged. ML/DNN advances to **48/62 (77.4%)**, overall **274/470 (58.3%)**, core
**173/175 (98.9%)**. Targeted cycle **17/20**, without a broad repeat. Next:
**S2-50 image-model anomaly maps**. See the [contract](image-regression.md).

**Rotated crop classification (S2-48)** adds the reusable `cv.crop_pose` primitive,
then composes existing blob poses, DNN/class decoding and bounded parallel
inspection ToolBlocks. Affine pose scale/shear and half-pixel crop geometry remain
explicit. Masks/regions/alpha define measured support; strict rejects unsupported
samples, while Pad reports coverage and a normal judgment. Crop encoding/depth,
spatial/pyramid transforms, child tickets and parent/feature references remain
intact. An annotation-only step now emits a completed empty layer when graphics
are disabled, avoiding false unfinished-buffer errors in Lean per-part graphs.

[Eleven original MIT projects](<../examples/opencv tools/rotated_crop_classification/README.md>)
use analytically drawn rotated parts and a **4,313-byte** native-trained linear-SVM
ONNX classifier. Twenty-four canonical noisy windows train the model; held-out
scene seeds differ. Measured blob moments provide an unoriented 180-degree axis,
which does not change either local stripe class. Five textured parts pass at the
illustrative 0.9 threshold (minimum about **0.9966**); an unmarked part scores
**0.5014** and fails. No-alignment has zero passes; all-known has six; horizontal-
only acceptance has three passes/three failures. Serial/parallel results agree.
These related controls do not establish industrial accuracy, calibrated confidence
or general unknown-class rejection. All **33 asset pins** and source records remain.

Independent moment/coordinate/bilinear and native/NumPy logit references differ
by less than **0.501 uint8 code units** and **5.4e-7 logits**. **173 focused
checks** pass, including relevant existing crop/pose/annotation/parallel contracts,
Debug/Lean, high depth, masks, gates, strict/padded failures, exact regeneration,
nested JSON/XML/Save As and cold one-image worker teardown. **53 installed
checks** pass; all **396 release-source/wheel/installed modules** match and **61
package versions** remain unchanged. One existing editor check initially hit its run-time limit; its isolated
recheck and the remaining affected checks passed.
ML/DNN advances to **47/62 (75.8%)**, overall **273/470 (58.1%)**, core
**173/175 (98.9%)**. Targeted cycle **16/20**, without a broad repeat. Next:
**S2-49 image-based regression**. See the [contract](pose-crop-classification.md).

**Cascade-based image detection (S2-47)** adds `ml.cascade_detect`: bounded,
optionally SHA-pinned modern HAAR/LBP XML, native boxes/weights/neighbors modes,
scale/size/group controls, working equalization and box-center/coverage selection.
Negative stage weights remain intact unless their explicit filter is enabled;
neighbor counts remain separate. Native nearest-scale fallback is reported and
optional strict size filtering enforces requested bounds. Source pixels/depth/
encoding, ticket/frame payloads, graphics policies and shared failure handling
remain unchanged. The CPU scope is shared with HOG and restores thread preference.

[Fifteen projects](<../examples/opencv tools/cascade_detection/README.md>) include
unchanged OpenCV 4.13.0 model XML qualified under OpenCV 5, the public-domain NASA
portrait, a two-size composition and valid empty/count/weight/model-pin controls.
The HAAR portrait box is **[176,65,97,97]**, with weight about **5.82965** or **37
neighbors**; LBP returns [170,63,107,107]. Both overlap the coarse manual ROI above
0.7 IoU; the composed view yields two faces. Ungrouped HAAR returns 40 windows,
including extras with negative margins. A 97×97 request selects native 100×100;
strict size rejects it. These are bounded CPU integration controls, not face or
industrial accuracy evidence. Source terms, six asset pins and license are retained.

**151 focused checks** pass across native references, parameter/catalog contracts,
Debug/Lean, uint16/encoding/payloads, selections and resource bounds, invalid model/
native exception recovery, exact regeneration, JSON/XML/Save As, input gates and
cold Run All/parallel/preview teardown; existing HOG, KCF and native context checks
remain passing. All **45 installed checks** pass. The wheel matches all
**395 release-source modules**, with **61 package versions** unchanged. ML/DNN
advances to **46/62 (74.2%)**, overall **272/470 (57.9%)**, core **173/175 (98.9%)**.
Targeted cycle **15/20**, without a broad repeat. Next: **S2-48 rotated crop
classification**. See the [contract](cascade-detection.md).

**HOG sliding-window detection (S2-46)** adds four reusable blocks: model
configuration, native detection and strict detector save/load. Native people
presets, explicit float32 coefficients and binary linear HOG-SVM conversion share
a revisioned geometry/weights contract. Search exposes scale/stride/padding,
native grouping, mask/region filtering, raw margins and independent IoU suppression.
Source pixels, encoding, depth, ticket/frame context and graphics policy remain
shared; count failures are distinct from model/native exceptions.

[Thirteen projects](<../examples/opencv tools/hog_detection/README.md>) use original
MIT synthetic parts with two exact boxes, separate query noise and 23 pinned
assets. At threshold 0.5 both one/three-level searches retain the two known parts;
at zero, suppression retains six results including four extras. An independent
501-window descriptor-dot-product reference agrees on all native candidate boxes,
with maximum margin difference below **3.3e-8**. Native stock people presets are
checked only on blank images. This qualifies CPU integration, not general
industrial/pedestrian accuracy, calibrated confidence or a speed improvement.

Cold Qt worker testing reproduced a native OpenCL device-release abort. Native
context preparation now occurs before short-lived workers, including nested
ToolBlocks/parallel inspections; HOG CPU search restores its calling-thread OpenCL
preference. The existing KCF startup safeguard remains covered. Project loading
stays inert. Search-work/resource limits reject excess work before native search
and suppression, without silently truncating detections.

**188 focused checks** pass, including independent native references,
Debug/Lean, strict model failures, uint16 and payload preservation, input gates,
JSON/XML/Save As and cold Run All/parallel/preview teardown. All **39 installed
example/native-runtime checks** pass. The installed wheel matches all **394
release-source modules**, with **61 package versions** unchanged.
ML/DNN advances to **45/62 (72.6%)**, overall **271/470 (57.7%)**; core remains
**173/175 (98.9%)**. Targeted cycle **14/20**, without a broad repeat. Next:
**S2-47 cascade-based image detection**. See the [contract](hog-detection.md).

**Object landmarks with model-specific skeletons (S2-45)** extend the existing
decoder instead of adding a person-specific wrapper. Configurable row columns,
flat exports, separate visibility/presence activation and score aggregation,
an independent scalar object gate and explicit scaled-index geometry support
additional landmark heads. Named/reordered identities remain stable; absent
points and their incident links can be hidden. Current image pixels/context,
preview, diagnostics, parameter validation and failure behavior remain shared.

The [contract](object-landmarks.md) and
[nine projects](<../examples/opencv tools/object_landmarks/README.md>) reuse
existing NHWC preparation and sealed inference with an unchanged **5,557,238-byte**
OpenCV Zoo MediaPipe Pose model and an attributed public-domain NASA portrait.
The selected 33-point skeleton excludes six auxiliary rows and world coordinates.
At minimum visibility/presence 0.8, **28/33 points and 28/35 links** remain;
indices **16,18,19,20,22** are suppressed. Presence alone accepts all 33.
Blank input has object score about **7.3e-12** and no accepted points. Missing
right wrist produces no pose. Independent direct native and NumPy mapping/score/
topology references match within **0.0002 source pixels** and **1e-6 score**,
with exact accepted identities/links. The supplied cropped portrait nevertheless
produces confident unsupported body predictions. These are configured-adapter
controls, not annotated localization/visibility accuracy or physical pose proof.

**159 focused checks** pass across the existing decoder and twelve original
graphs, generic score/column/layout/origin controls, parameter/resource bounds,
native examples and catalog/reference contracts. All **23 installed example
checks** pass, including exact assets/regeneration, JSON/XML/Save As, pinned
model/preprocessing integrity, input gates and cold one-image Run All teardown.
Only eight optional default fields were added to the original twelve graphs;
their existing values, connections, images and models remain unchanged.
All **392 release-source/wheel/installed modules** match and **61 package versions**
remain unchanged. The release includes the committed FoundationPose example
publication; unrelated ongoing plugin work is preserved. ML/DNN is
**44/62 (71.0%)**, overall **270/470 (57.4%)**, core **173/175 (98.9%)**.
Targeted cycle **13/20**, without a broad repeat. Next: **S2-46 HOG sliding-window
object detection**. Real SuperPoint, EXR, useful LK-pyramid reuse and accelerator
qualification remain open.

**Visual-word image retrieval (S2-44)** adds seven reusable feature, vocabulary,
signature, gallery, ranking and portable model I/O blocks under OpenCV → ml.
SIFT uses deterministic initialization and native Euclidean k-means; ORB retains
actual farthest-first binary representative words and uses Hamming/Hamming2.
Saved active detector settings, codebook revisions, normalization and assignment
thresholds prevent incompatible signatures from sharing a gallery. Exact ranking
exposes four metrics, inclusive thresholds, stable ties and explicit empty results.
Current images remain unchanged; optional drawable centers/labels and histogram
plots reach existing review/filmstrip tools. Lean preserves numeric decisions.

The [contract](visual-words.md) and
[twelve projects](<../examples/opencv tools/visual_words/README.md>) train only on
six original MIT part-like motif views. Three held-out rotated/noisy/darker queries
rank the intended label first with SIFT, ORB and Hellinger signatures. Raw-frequency
cosine distances are **0.0119/0.0321/0.0396** for SIFT and
**0.0057/0.0158/0.0871** for ORB. Independent native extraction and NumPy
Euclidean/bit-count assignment have no differing word IDs; SIFT roundoff is below
**2.8e-5**, ORB distances agree exactly and saved pipelines match the reference
signatures/ranks. A shuffled-quadrant image still ranks as washers: spatial layout,
pose and pickability are not verified. An intentionally strict threshold rejects
a known view. These controlled results do not establish production accuracy.

**148 focused checks** pass across native/reference words and metrics, WTA 2/3/4,
selection/alpha/depth, active identities, model integrity, input gates, failures,
resource bounds, shared descriptor/PCA helpers and catalog/reference contracts.
All **29 installed example checks** pass, including exact regeneration, portable
JSON/XML/Save As and cold three-frame Run All teardown. All **392 release-source/
wheel/installed modules** match; **61 package versions** remain unchanged. The
release includes the committed GDRNPP worker fix and excludes another chat's
unfinished FoundationPose example work. ML/DNN is **43/62 (69.4%)**, overall
**269/470 (57.2%)**, core **173/175 (98.9%)**. Targeted cycle **12/20**, without
a broad repeat. Next: **S2-45 object landmarks with model-specific skeletons**.
Real SuperPoint, EXR, useful LK-pyramid reuse and accelerator qualification remain open.

**Image-feature outlier inspection (S2-43)** extends the native covariance and
Mahalanobis tools to image descriptor batches with exact preprocessing/PCA
identity, saved row normalization, float64 fitting, explicit sample/population
and fixed-center policies, regularization and portable model save/load. Current
review images retain their pixels/ticket/grid and receive drawable window-level
judgment outlines. The outline does not localize defects; Lean omits graphics.

The [contract](feature-outliers.md) and
[thirteen projects](<../examples/opencv tools/feature_outliers/README.md>) use
nine original known-good training images and separate held-out controls. Raw
known-good distances are **2.41–2.53**, versus **17.09** for the crack and **29.63**
for the missing arc at the illustrative limit **4**. The subtle edit scores
**2.53 and passes**: this is a documented miss, not a sensitivity claim. Independent
NumPy covariance/quadratic-form and native references agree within **3e-14**.
L2, singular/no-ridge, changed descriptor and same-size wrong PCA revision cases
expose normalization and compatibility limits. Default 324-value HOG also works
directly; PCA is optional under the configured work/workspace bounds.

**141 focused checks** pass across covariance, examples, shared descriptor/PCA
helpers, catalog and references. All **32 installed example checks** pass, with
exact regeneration/assets, JSON/XML, portable Save As, input gates and a cold
three-frame Run All worker. All **391 release-source/wheel/installed modules**
match and **61 package versions** remain unchanged. The release includes the
other chat's committed GDRNPP review; its unfinished FoundationPose example work
is preserved outside this increment. ML/DNN is **42/62 (67.7%)**, overall
**268/470 (57.0%)**, core **173/175 (98.9%)**. Targeted cycle **11/20**, without
a broad repeat. Next: **S2-44 bag-of-visual-words image retrieval**, with a trained
codebook, image-level histograms and held-out ranking. Real SuperPoint, EXR,
useful LK-pyramid reuse and accelerator qualification remain open.

**Image color/texture mixtures (S2-42)** add four native EM training,
classification and portable model I/O tools under OpenCV → ml. Normalized color
or selected local mean/std features expose model-specific components, native
posteriors, absolute density, accepted/unknown/support masks and explicit judgment
limits. Auto and manual Gaussian-responsibility starts, covariance structures,
regularization and bounded work are configurable. Spherical fits use equivalent
diagonal prediction; native multidimensional trainE binding remains unqualified.

The [contract](image-mixtures.md) and
[thirteen projects](<../examples/opencv tools/image_mixtures/README.md>) include
original MIT surfaces with separate training/query noise and independent edit
masks. The color example rejects exactly **120 edited pixels**, with no extras or
misses, even though every winning posterior is **1**. Disabling density intentionally
accepts the edit. Independent native predict2 and Gaussian references agree within
**2.3e-13**. Texture has **3,400 supported centers** from **4,480 selected pixels**;
it flags **192 of 256 edited pixels**, misses **64** at mixed neighborhoods and adds
none. These controls expose limits rather than establish production sensitivity.

**93 distinct focused checks** pass for native/reference fitting and scoring,
all covariance structures, encoding/depth/alpha, support/region/ticket/pyramid
contracts, bounded failures/recovery, model integrity, catalog and saved projects.
All **31 installed example checks** pass, including exact regeneration, JSON/XML,
portable Save As and a cold one-image Run All worker with clean shutdown.
All **387 release-source/wheel/installed modules** match; **61 package versions**
are unchanged. The release copy excludes another chat's unfinished pose-plugin
changes, which remain untouched in the shared checkout. ML/DNN is **41/62 (66.1%)**,
overall **267/470 (56.8%)**, core **173/175 (98.9%)**. Current targeted cycle
**10/20**, without a broad repeat. Next: **S2-43 image-feature outlier detection**,
qualifying the existing covariance/Mahalanobis tools on known-good variation and
held-out image defects. Real SuperPoint, EXR, useful LK-pyramid reuse and accelerator
qualification remain open.

**Image-descriptor PCA (S2-41)** qualifies the existing native fit/project
tools on image descriptors and adds portable model save/load. Bounded row groups,
source preprocessing, exact mean/basis revisions, count/variance/L2/fixed-center
controls and per-row reconstruction errors support matching downstream classifiers.
Changed HOG settings and a different valid same-size PCA model are rejected.
Explicit dimensions-only operation is unverified and gets a distinct output
contract. Reconstructed features also keep a distinct identity, including L2 units.
Legacy numeric models and the existing table project remain compatible.

The [contract](descriptor-pca.md) and
[ten projects](<../examples/opencv tools/descriptor_pca/README.md>) use original
MIT stripe windows: six for fitting and three with disjoint pixel hashes for
held-out checks. Native HOG has **324 values**; **two PCA components** retain
**99.9764%** training variance. Both known held-out orientations retain their raw
nearest-neighbor labels. Their reconstruction RMSE is **0.00407/0.01636**, versus
**0.14491** for the diagonal control, above the chosen **0.03** demonstration
limit. The unguarded classifier still assigns the diagonal to a known class.
An independent native HOG/NumPy SVD reference checks the saved subspaces and
labels. These controls establish neither general accuracy nor a speed benefit.

**138 distinct focused checks** pass across native/SVD parity, contracts,
float64/fixed-center and L2 behavior, strict/bounded models, failures/recovery,
graphics/provenance, existing numeric and K-nearest behavior, catalog and references.
All **26 installed example checks** pass, including exact regeneration, portable
Save As/XML and a cold two-frame Run All worker with clean shutdown. All
**386 source/wheel/installed modules** match; **61 package versions** are unchanged.
ML/DNN is **40/62 (64.5%)**, overall **266/470 (56.6%)**, core **173/175 (98.9%)**.
Current targeted cycle **9/20**, without a broad repeat. Next: **S2-42 texture and
color mixture classification**, using persisted native EM models and held-out image
samples. Real SuperPoint, EXR, useful LK-pyramid reuse and accelerator qualification
remain open. Concurrent completed Deflectometry examples are indexed from their
owner-reported publication; this increment does not repeat their validation.

**Learned video box tracking (S2-40)** adds `sequence.track_vit` through the
existing seeded native sequence lifecycle. The small unchanged Apache ViTTrack
model is included with exact publisher hash/notice, so all nine projects run
offline. Native mean/std, response threshold and DNN controls are exposed; explicit
Classic/New CPU import, immutable snapshots and sealed settings/preparation are
recorded. Changed model bytes cannot silently reuse the old native state.

The [contract](learned-tracking.md) and
[nine projects](<../examples/opencv tools/learned_tracking/README.md>) qualify
warm-up, score/contrast/jump/selection gates, loss/expiry, explicit detector
reseed, reset/stream/context changes, independent/editor handle ownership and
bounded crop/model allowance. Current image/status and a current native score
remain available while unready/lost boxes and poses are withheld. Score is a
window-weighted response, not a probability; pose angle follows image axes.
Lean skips graphics while preserving native state/acceptance.

Independent native evaluation on artificial translations of licensed public
T-LESS imagery gives minimum box **IoU 0.93284** and maximum center error
**1.11804 pixels** across six updates. A blank frame has score **0.09838**, below
the configured **0.2** threshold. This is controlled clip/box evidence, not
recorded camera motion, segmentation ground truth or physical part pose.

**104 focused checks** pass across native behavior, all projects, model contracts,
registry and documentation. **23 installed example checks** and a cold installed
Run All worker pass, with seven frame events for seven inputs and clean teardown.
Sealed Save As/XML starts fresh native state and retains exact weights/notices.
All **386 source/wheel/installed modules** match; **61 package versions** are
unchanged. ML/DNN is **39/62 (62.9%)**, overall **265/470 (56.4%)**, core
**173/175 (98.9%)**. Current targeted cycle **8/20**, without a broad repeat.
Next: **S2-41 image-descriptor dimensionality reduction**: qualify the existing
PCA tools in a persisted descriptor pipeline on held-out images and retain the
feature model revision. Real recorded tracking/occlusions, real SuperPoint,
native EXR, useful LK-pyramid reuse and accelerators remain open.

**Learned optical flow (S2-39)** adds `ml.optical_flow` over shared ONNX
snapshots/contracts/receipts. Two named or concatenated frames expose explicit
layouts, normalization, xy/yx order, fixed component affine rules and independent
head/network/source-pixel scaling. Output vectors index the original first grid
and map first to second. Both tickets/grids/recipes/pyramids survive; consecutive
frames need not share a ticket. Padded/invalid contributors and fractional endpoint
masks reduce support rather than clipping vectors. Optional content-bound image
and field manifests let existing motion consumers reject wrong directions/images.

The [contract](learned-flow.md) and
[twelve projects](<../examples/opencv tools/learned_flow/README.md>) qualify nine
arithmetic controls and three unchanged pinned BSD/MIT RAFT Classic/CPU paths.
The full-size output is selected by name because native terminal order differs
between engines. Independent source-grid references on artificial translations
of licensed public T-LESS texture report mean endpoint errors **0.01861**,
**0.01457** and **0.00297 source pixels** for positive/negative/zero motion.
All three retain the fixed evaluation interior; this is not recorded optical
motion, physical displacement or occlusion qualification. Supported ramp warps
and NaN-preserving native flow-file round trips provide separate consumer evidence.

Drawable arrows reference the first image, with a bounded sampling stride.
Lean/graphics-off skips previews, drawings and unexposed magnitude/angle maps
without changing numerical vectors/support. Empty selections skip model I/O;
native/shape exceptions clear the cache without a new output event.

**148 focused checks** pass across quantities, pairing, consumers, saved examples,
shared model contracts, registry and reference documentation. **29 installed
example checks** and a cold installed native worker pass, including sealed Save
As/XML and clean teardown. All **385 source/wheel/installed modules** match;
**61 package versions** are unchanged. ML/DNN is **38/62 (61.3%)**, overall
**264/470 (56.2%)**, core **173/175 (98.9%)**. Current targeted cycle **7/20**,
without a broad repeat. Next: **S2-40 learned video object tracking**, with pinned
assets and annotated init/update/loss/reset evidence. Real SuperPoint, native
EXR, useful LK-pyramid reuse and accelerator qualification remain open.

**Learned stereo disparity (S2-38)** adds `ml.stereo_disparity` over shared
ONNX model contracts/receipts. Two named or concatenated views expose explicit
layouts, normalization, head/network/source pixel units and affine/sign conversion.
Disparity stays on the original left grid as `x_left - x_right`; rounded-letterbox
padding, raw invalid contributors and predicted right-view mask/boundary support
remain invalid rather than clipped. Ticket policy, both camera frames, recipe and
pyramid references are retained; rectification is caller-supplied, not inferred.

The [contract](learned-stereo.md) and
[twelve projects](<../examples/opencv tools/learned_stereo/README.md>) qualify nine
original arithmetic controls and three unchanged pinned Apache HITNET ETH3D
New/CPU controls. The exact artifact imports/executes with New; Classic rejects
GatherND, so its engine is selected explicitly without silent retry. A known
3-pixel plane and 0.1 m baseline reproject to independently calculated camera rays
at 8/3 m depth. Native pairs use artificial shifts of a licensed public T-LESS
image, not physical stereo captures or those objects' measured depths.

Independent direct-native source-grid references report **0.03025/0.03049 source
pixel MAE** for 8/16-pixel shifts with full interior retained support. The zero
plane has **0.20038 pixel MAE** over the full evaluation region and **94.12%**
retained support under nonnegative-contributor/correspondence gating. Metrics
and support are reported separately; coverage is not matching confidence.

**144 focused checks** pass across quantity/geometry, saved examples, shared model
contracts, registry and documentation. **29 installed example checks** and a cold
installed native worker pass, including sealed Save As/XML, Debug/Lean and exact
artifact/preparation evidence. All **384 source/wheel/installed modules** match;
**61 package versions** remain unchanged. ML/DNN is **37/62 (59.7%)**, overall
**263/470 (56.0%)**, core **173/175 (98.9%)**. Current targeted cycle **6/20**,
without a broad repeat. Next: **S2-39 learned optical flow**, with explicit paired
frame direction, displacement scale, source support and known motion. Physical
stereo/occlusion accuracy, real SuperPoint, native EXR, useful LK-pyramid reuse and
accelerator qualification remain open.

**Optional-input completed plugin review** now accepts an explicit
`snapshot_processed_execution(input_ports=(), ...)` after a valid Debug
completion that consumed no wired inputs, such as a file/manifest-driven tool.
The result/report packets retain executed evidence and private copy isolation;
capture never inserts an input, executes processing or invokes training.
Required-input, owner-thread, active/closed, exception and missing-output guards
remain. An internal per-completion input-retention marker prevents a profile
switch or restored snapshot from recreating discarded history. Default and
input-only captures retain their previous availability checks. See
[plugin review contracts](plugin-dialogs.md#review-one-completed-execution-without-rerunning).

**104 focused checks** pass across paired/connected/input/cloud captures, review
dialogs and run profiles, including nine new optional-input/retention cases.
The installed wheel passes all **27 paired snapshot checks** plus **28 monocular
example checks**; all **383 source/wheel/installed modules** match and **61 package
versions** are unchanged. This shared-contract repair was checked against the
affected suites without repeating the broad audit. It is outside the vision
denominator; current targeted cycle **5/20**. Next remains **S2-38 learned stereo
disparity**.

**Learned monocular depth (S2-37)** adds `ml.monocular_depth` over shared
ONNX contracts and receipts. Fixed affine head values, explicit relative/metric
and inverse conventions, m/mm, output layout, normalization, resize and bounds
are exposed. Relative maps use a data port and never emit physical Depth.
Metric conventions are caller declarations with `metric_validated: false`.
Source-center resampling conservatively rejects padded/invalid contributors;
selection gates output while preserving source context, tickets and coordinates.
Coverage judgments are separate from model confidence and diagnostic display.

The [contract](monocular-depth.md) and
[eleven projects](<../examples/opencv tools/monocular_depth/README.md>) qualify
nine arithmetic controls, exact metric camera-ray XYZ and two explicitly
prepared unchanged MIT MiDaS v2.1 Small Classic/CPU paths. The real CC BY 4.0
T-LESS capture has 387,684 positive reference depth pixels; reference depth is
never supplied to inference. Independent native/bilinear samples agree, and
near/far ordering is **97.20%** on 3,609 eligible pairs. An evaluation-only
inverse-depth fit on the left half gives **51.36 mm** RMSE on the right half,
worse than the **40.70 mm** constant-depth baseline. This is one acquisition,
not independent metric calibration; MiDaS output remains relative.

**124 focused checks** pass across quantity/geometry contracts, all saved
examples, model contracts, registry persistence and reference documentation.
Sealed Save As/XML, ignored pinned model preparation, Debug/Lean and cold native
background teardown are checked. **28 installed example checks** and an installed
cold worker pass; all **383 source/wheel/installed modules** match and **61 package
versions** remain unchanged. ML/DNN is **36/62 (58.1%)**, overall **262/470 (55.7%)**,
core **173/175 (98.9%)**. Current targeted cycle **4/20**, without a broad repeat.
Next: **S2-38 learned stereo disparity** with paired-image preparation, known
disparity, source masks/frames and explicit pixel units. Real SuperPoint, native
EXR, useful LK-pyramid reuse and accelerator qualification remain open.

**Learned image restoration (S2-36)** adds `ml.restore_image` over shared ONNX
inference, model snapshots and revisioned receipts. Same-grid direct/residual
heads expose normalization, NCHW/NHWC, depth/color/alpha and source selection.
Whole-image or bounded context-tile processing writes disjoint cores without
resizing or moving source coordinates. Empty selections/blend zero skip model
I/O; native/shape errors reset the model and emit no new output event.

The [contract](learned-restoration.md) and
[thirteen projects](<../examples/opencv tools/learned_restoration/README.md>)
qualify nine original arithmetic controls and four explicitly prepared native
DnCNN-25 paths. The MIT source weights remain unchanged; a deterministic
dynamic-shape ONNX export agrees with independent source tensor arithmetic
within 2.2352e-6 normalized units. No training or remote Python code runs, and
application inference uses OpenCV. Private local preparation retains exact
source/export hashes and sealed settings; authoring packages are not new runtime
dependencies.

Independent clean/degraded references report Gaussian/median/learned errors,
flat/texture behavior, false/erased crack pixels and tile seams. Learned MSE
improves from 628.3195 to 14.4515 on the noisy control, but erases all 29 faint
crack pixels. On low-noise input MSE worsens from 4.1516 to 11.8902. A 17-pixel
context matches whole inference within 2e-6 normalized units on these controls;
zero context differs by up to 30 integer samples. Deliberate fidelity and seam
checks fail independently of successful inference. Other/nonlocal models need
their own tile and defect-sensitivity qualification.

**128 focused checks** pass across restoration, saved examples, model contracts,
registry persistence and offline documentation. **31 installed example checks**
and a cold background worker pass, including nine current native context calls
and clean process exit. All **382 source/wheel/installed modules** match;
**61 package versions** are unchanged. Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93
remain in use. ML/DNN is **35/62 (56.5%)**, overall **261/470 (55.5%)**, core
**173/175 (98.9%)**. Current targeted cycle **3/20**, without a broad repeat.
Next: S2-37 learned monocular depth with explicit relative/metric conventions
and known depth imagery. Real SuperPoint, native EXR, useful LK-pyramid reuse
and accelerator qualification remain open.

**Learned single-image super-resolution (S2-35)** adds `ml.super_resolution`
over native `dnn_superres`, with an unchanged pinned Apache FSRCNN x2 CPU model.
The [contract](super-resolution.md) and
[eight projects](<../examples/opencv tools/super_resolution/README.md>) qualify
exact native pixels, source color/alpha, explicit high-depth quantization,
mask/region output selection, nominal spatial/pyramid mapping and byte-preserving
model assets. A bounded per-block native cache imports private hashed snapshots;
source bytes are verified every inference activation. Independent instances can
execute concurrently, and native creation is guarded around layer registration.

The independent retained benchmark compares learned, linear, cubic and Lanczos
results against original known pixels, feature sizes and flat-field noise.
Learned output lowers clean-control MAE (1.1711 versus cubic's 1.7218), but still
loses the one-pixel crack, shrinks a thresholded circle from 73x73 to 71x71,
erases 27 dark-detail pixels versus cubic's 26, and changes 30 initially uniform
samples. On noisy input its flat standard deviation exceeds linear's. These
controls expose reconstruction changes, not added measured resolution or physical
dimensional accuracy. Other exports/scales/devices and multi-output S2-58 remain
separate qualification.

**76 focused checks** pass across native contracts/examples, registry persistence
and offline reference documentation. Fresh-process background checks caught and
fixed an unnecessary CPU device-enumeration path that initialized OpenCL on a
worker and failed during driver teardown. CPU execution now avoids that query;
the cold worker and installed example suite exit cleanly. No global OpenCL
setting was changed. The installed package passes **21 example checks**;
all **381 source/wheel/installed modules** match, and all **61 package versions**
are unchanged. Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93 remain in use.

The S2-35 checkpoint reached ML/DNN **34/62 (54.8%)**, overall **260/470 (55.3%)**,
core **173/175 (98.9%)**, targeted cycle **2/20**, without a broad repeat.
The separately owned
[SimpleBinPicking completed Results review](<../examples/Simple Plugins/bin_picking/completed_runtime/README.md>)
is indexed; its plugin/native validation is owner-reported. The next item was S2-36 learned
image restoration. Real SuperPoint, native EXR, useful LK-pyramid consumption
and accelerator qualification remain open.

**Constrained OCR decoding (S2-34)** adds `ml.recognize_constrained_text`
without changing existing greedy OCR defaults or sealed projects. Alphabet and
optional exact-word constraints support raw greedy verification or bounded
label-prefix beam search. Retained sequences receive exact log-domain CTC
forward rescoring; ranking keeps original model mass, never dictionary-normalized
confidence. Optional alternatives, selected Viterbi time-step evidence, pruning
counts and lexicon identity remain inspectable. Blank-dominance and optional
greedy-relative gates expose/reject forced dictionary reads.

The [contract](constrained-ocr.md) and
[eleven projects](<../examples/opencv tools/constrained_ocr/README.md>) qualify
independent exhaustive alignment references, image-dependent native probability
and logit controls, explicit rejection cases and an unchanged pinned Apache CRNN
reading three printed labels. **166 focused checks** pass across new decoding,
native examples, existing batch OCR and model contracts. They cover repeats,
nonzero blanks, multi-character labels, extreme/zero probabilities, finite search
limits, resource recovery, Debug/Lean, source tickets/pixels, pinned lexicon files
and JSON/XML relocation. Finite beam search has no global-optimum guarantee;
exact label-sequence scores are not calibrated correctness or a sum over all
tokenizations of the same text. Character spans are model steps, not pixel boxes.

The S2-34 checkpoint reached ML/DNN **33/62 (53.2%)**, overall **259/470 (55.1%)**,
core **173/175 (98.9%)**, targeted cycle **1/20**. At that checkpoint all **380
source/wheel/installed modules** matched; **61 package versions** were unchanged.
The installed package passes **25 new example checks** and a cold background
worker returns three native reads in one acquisition. Python 3.14.3/OpenCV 5.0.0
contrib 5.0.0.93 remain in use. The separately owned
[Simple3DMeasure completed Results review](<../examples/Simple Plugins/3d_measure/completed_runtime/README.md>)
is indexed; its plugin/native validation is owner-reported. The next item was S2-35 learned
single-image super-resolution; real SuperPoint, native EXR and usable LK-pyramid
consumption remain open. No broad rerun was needed for this targeted increment.

**Batched region OCR (S2-33)** adds `ml.recognize_text_batch` over the existing
native batch preparation/inference path. Explicit TNC/NTC heads normalize to
ordered per-region CTC rows. Source quads/IDs/tickets/recipes, crop homographies,
raw texts, token step spans, local blank/confidence/skipped status and accepted
aggregate reads remain inspectable. A shared output-axis extension leaves
ordinary image inference on its original axis-zero contract. The older
single-region recognizer now uses the engine-preserving snapshot loader.

The [contract](batch-ocr.md) and
[eight projects](<../examples/opencv tools/ocr_batch/README.md>) qualify original
MIT dynamic arithmetic models on N=3 and variable/padded tails, independently
matching single-crop paths/confidence. An unchanged hash-pinned Apache CRNN
reads `lot2048`, `pnax12`, `pass`, agrees with the separate recognizer and
explicitly remains fixed N=1; unsupported batch declarations do not silently
fall back. **145 focused checks** pass across the new OCR, existing batch/model
contract and single-region OCR suites. They cover glyph/CTC references,
Debug/Lean, RGB/uint16 preservation, metadata conflicts, vocabulary digest,
limits/recovery and JSON/XML relocation. Confidence is the geometric mean of
emitted run peaks, not a calibrated string probability or summed CTC posterior.
The block handles multiple regions of one acquisition, not multi-ticket input
collection. Lexicon/prefix decoding is separately qualified by S2-34 above.

ML/DNN is **32/62 (51.6%)**, overall **258/470 (54.9%)**, core
**173/175 (98.9%)**. The twenty-increment cycle is closed by the broad audit and **293** scoped
rechecks above; the next targeted cycle began at **0/20**. At that checkpoint all **378 source/wheel/installed modules** matched,
and all **61 package versions** remain unchanged. The installed package passes
**20 example checks** and a cold background worker returns three current native
reads from one acquisition. Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93 remain
in use. The separately owned
[Simple3DLocate completed Results review](<../examples/Simple Plugins/3d_locate/completed_runtime/README.md>)
is indexed; its native/plugin validation is owner-reported. The next item was S2-34
constrained OCR decoding, while real SuperPoint, native EXR and useful
LK-pyramid consumption remain open.


**Pinned text detection (S2-32)** qualifies the Apache-2.0 PP-OCRv3 English
model through existing DNN inference and `ml.text_regions`. Preparation verifies
the exact 2,423,490-byte artifact, seals BGR normalization/white-letterbox/CPU
settings and remains explicitly offline once prepared. The observable 736x736
map feeds external-contour scoring, optional DB-style rectangle expansion,
reference ROI and count/truncation judgments. The decoder now correctly maps
head centers through network edges to source centers when scales differ;
tickets, known roots/root dimensions and recipes must agree.

The [contract](text-detection.md) and
[eight examples](<../examples/opencv tools/pinned_text_detection/README.md>)
qualify original MIT industrial-style glyph fixtures with independent polygons,
known 12-degree rotation, joined/separate text and ordinary blank/count/
truncation failures. **49 focused tests** pass, including the existing OCR
suite, Debug/Lean native cases, Classic/New CPU agreement, at least 95% polygon
IoU against native DB on an identity-grid reference, exact half-pixel/DB-offset
references, budget/context rejection/recovery and JSON/XML relocation. Numeric
regions and unchanged source format/depth/payload survive graphics suppression.
No OCR text is invented: character recognition, natural-image application
accuracy and accelerator qualification remain separate. Installed module and
package verification passes: all **377 source/wheel/installed module files**
match, and all **61 package versions** remain unchanged. The installed package
passes **21 example checks** and the cold background worker returns three current
regions from one acquisition. Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93
remain in use.

ML/DNN is now **31/62 (50.0%)**, overall **257/470 (54.7%)**, core
**173/175 (98.9%)**. Targeted cycle **19/20**; no broad suite repeated. Next:
batched OCR recognition and per-region order/confidence qualification (S2-33).
Real SuperPoint extraction, native EXR and useful LK-pyramid consumption remain
open. The separately owned [SimpleClassify completed-run Results review](<../examples/Simple Plugins/image_classification/completed_runtime/README.md>)
is indexed; its plugin validation remains owner-reported.


**Tiled semantic segmentation (S2-30)** adds optional array-backed class evidence
from `ml.decode_segmentation` and `ml.fuse_tile_segmentation` after ordinary
Crop/Resize, DNN and Join branches. Probability means or nearest-label votes use
uniform/feather weights on the reference grid. Known tickets, roots, ordered
classes, pyramid contexts and recipes must agree. Unknown pixels remain `-1`;
failed tiles, incomplete coverage and area limits retain current failing outputs
with reasons. Reference image format, depth and auxiliary payload stay intact.

The [contract](tiled-segmentation.md) and
[nine projects](<../examples/opencv tools/segmentation_tile_fusion/README.md>)
qualify original MIT pixel-dependent ONNX equations against independent labels
and a separate whole-image execution. Red parts cross both overlap boundaries
without splitting. Classic/New CPU engines, downscaled tiles, ROI, empty/area/
coverage/failed-tile controls, bounded arrays/work, context rejection/recovery,
Debug/Lean graphics and JSON/XML relocation are checked. **147 targeted tests**
pass, including the existing semantic decoder suite. Full class export stays off
by default; label votes explicitly report vote fractions rather than model
probabilities. Supported maps are crop/scale/flip/right-angle; learned accuracy,
context halos, instance identity fusion and gigapixel streaming are outside this
qualification. The rebuilt/installed wheel matches all **377 source modules**;
all **61 package versions** remain unchanged. Its **20 example checks** and cold
background worker pass (one acquisition, 788 selected pixels). Python 3.14.3 and
OpenCV 5.0.0 contrib 5.0.0.93 remain in use.

ML/DNN is now **30/62 (48.4%)**, overall **256/470 (54.5%)**, core
**173/175 (98.9%)**. Targeted cycle **18/20**; no broad suite repeated. Next:
qualify a pinned text detector (S2-32), then batched OCR recognition (S2-33).
Real SuperPoint extraction, native EXR (S1-125) and usable LK-pyramid consumption
(S1-155) remain open. The newly published, independently owned
[SimpleSegmentation completed-run review](<../examples/Simple Plugins/pixel_segmentation/completed_runtime/README.md>)
is indexed in the example catalog; its plugin qualification is owner-reported.


**Prompted segmentation (S2-29)** adds separate source prompt, EfficientSAM ONNX
inference and mask decoding blocks. The pinned Apache 2.0 April 2025 Ti export
consumes full uint8 RGB images, foreground/background points and box corners in
six completely rebound slots. Three array-backed native logit planes remain
inspectable through a group selector. Pixel/format signatures, prompt/model
revisions and ticket/frame/pyramid/recipe checks prevent mixed source requests.
Independent decoder settings expose quality/index selection, prompt consistency,
nearest-mask/linear-logit mapping, output selections, exact regions, optional
logits/graphics and distinct normal quality/prompt/area failures. Failed current
geometry remains available; malformed or mixed data emits no current result.

Eight [runnable projects](<../examples/opencv tools/prompted_segmentation/README.md>)
use an unchanged credited CC0 photograph and original shapes with independently
constructed truth. Rectangle Dice is 0.977 for box/nearest and 1.0 for point/linear
on the qualified fixture. **95 unique focused checks pass across targeted runs**,
including independent native preprocessing/prompts/logits/masks, Debug/Lean,
negative-point failure, region/quality gates, source/request/budget guards, warm
rebinding and stable old buffers, model contracts, XML, portable Save As,
offscreen controls and a cold GUI worker using the existing single-image Run All
fallback. The fused encoder reruns per request; split embedding reuse, other SAM
exports, accelerators and general image/industrial accuracy remain unqualified.
The installed application passes all **16 Debug/Lean project outcomes** and the
cold worker check. All **376 Python module files** match source/wheel and all
**61 installed-package versions** remain unchanged.
ML/DNN reaches **29/62 (46.8%)**, overall **255/470 (54.3%)**; core remains
**173/175 (98.9%)**. Targeted cycle **17/20**, no broad suite repeated.
Next: S2-30 tiled segmentation.

**Detection tracking with appearance (S2-28)** extends the Kalman/global association
tools with optional named-space vectors, cosine gates/costs, bounded matched-only
template updates, class IDs and current detector IDs. Appearance defaults off.
Required/if-both policies, feature/model changes, prediction/expiry and detector
versus local-flow evidence remain explicit. `ml.box_appearance` supplies bounded
uint8 HSV/grayscale ROI histograms with inset/mask/region/alpha support and original
pixels/records. External learned vectors share the contract; no re-identification
model is qualified. `evaluate_tracking` supplies bounded global class-compatible
IoU matching, identity switches, fragments, MOTA and per-frame offline reports.

Six [annotated projects](<../examples/opencv tools/detection_tracking/README.md>)
use actual blob/YOLOX detections. The hidden swap retains identities with appearance
and records two switches with motion alone. A dim-part miss retains three false
negatives and one fragment; shorter expiry adds one identity switch on re-entry.
Pinned YOLOX on translated CC0 photographs retains the cup through a hidden frame.
**117 focused checks pass**: Debug/Lean, independent histogram/normalization/metric
references, missing features, class/frame/ticket gates, bounds, exception recovery,
worker isolation, tracker regressions, XML, offscreen controls, portable assets,
two existing source-example regenerators and a cold Run All worker. All twelve
installed Debug/Lean sequence runs and the cold worker also pass. All **374 Python
module files** match source/wheel; all **61 package versions** remain unchanged.
Ten existing saved projects retain default-off appearance; original images remain
unchanged. Controlled CPU scope, without a real-video benchmark, physical identity
claim or HOTA/IDF1. ML/DNN reaches **28/62 (45.2%)**, overall **254/470 (54.0%)**;
core remains **173/175 (98.9%)**. Targeted cycle **16/20**, no broad suite repeated.
Next: S2-29 prompted segmentation.

**Cross-tile detection fusion (S2-27)** adds `ml.fuse_tile_detections` for grouped
box collections from crop/resize detector branches. Known root/ticket/recipe
checks precede exact pixel-center-to-edge mapping. Crop, anisotropic scale, flip
and right-angle maps are supported; arbitrary rotations/shear/perspective reject.
Explicit local/unclipped box sources, reference selection/border policies,
optional internal-edge exclusion and class-aware/agnostic native hard NMS expose
confidence, IoU, eta and top-k. Separate raw/candidate/result budgets reject
instead of silently truncating. Retained local observations remain in provenance,
while fresh geometry and output payloads use the reference image grid. Count and
failed-tile judgments retain available geometry; malformed groups and native
exceptions emit no current output event and recover on a valid activation.

Eight [runnable projects](<../examples/opencv tools/detection_tile_fusion/README.md>)
use a pinned Apache 2.0 YOLOX-S model on overlapping crops of a CC0 photograph,
a half-size branch and a black negative. Explicit preparation seals a local model
bundle; no runtime downloads occur. Independent native preprocessing/grid/box
references, hand-marked truth, duplicate false positives, exact seam/flip/rotation
controls, invalid identities/budgets, failure recovery, serial/parallel parity,
Debug/Lean, XML, portable Save As and a cold offscreen CPU worker pass **80 focused
checks**; **97 existing decoder/collection/payload/toolbox checks** also pass.
Four installed Debug/Lean runs and a fresh installed worker pass; all **372 Python
modules** match source/wheel and all **61 package versions** remain unchanged.
See [contracts](tile-detection-fusion.md). This is axis-aligned hard-NMS
qualification, without fragment reconstruction, weighted fusion, general
homographies or dataset accuracy claims. ML/DNN reaches **27/62 (43.5%)** and
overall **253/470 (53.8%)**; core remains **173/175 (98.9%)**. Targeted cycle
**15/20**; no broad suite repeated. Next: S2-28 detection tracking.

**Read-only plugin result review (application API/editor reliability)** adds
`Block.plugin_review_dialog_label` and
`create_plugin_review_dialog(self, parent, context)`. Opted-in blocks receive a
Results-tab action with a separate modeless popup and capture context. It never
invokes the training factory, reads/validates a configuration draft, stages values
on accept, applies settings or runs the block. Plugins choose when to capture the
completed inputs/outputs and how to present their retained model/fit evidence.
Training and review subscriptions receive coalesced publications independently;
either popup can close without disabling the other. Owner lifetime events dispose
the context, and factory/subscriber failures remain contained and retryable. The
assistant schema advertises the review capability without opening a factory.

Verification: **179 unique focused editor/plugin/capture/profile/API checks pass
across targeted runs**. The installed wheel passes **66 capture/dialog checks**
in 5.60 seconds. These cover actual worker publications, no automatic capture or execution,
invalid drafts left untouched, independent training/review contexts, normal
selection/graph/hide/close disposal, and invalid factories/subscriber exceptions.
Release is idempotent after Qt has destroyed the notification timer. The deliberately
destructive native-deletion control runs in a fresh Qt process, with exit zero and
no teardown traceback. An earlier mixed-process attempt encountered native Shiboken
collection failure, so native deletion is isolated from later graph-worker fixtures.
The installed application is refreshed; all **371 Python module files** match
source/wheel and all **61 dependency versions** remain unchanged. This is generic host support, not qualification
of any plugin's own review implementation. Vision counts stay unchanged and the
targeted cycle at that checkpoint reached **14/20**; S2-27 was next.

**Paired plugin-review snapshots (application API reliability)** add
`PluginDialogContext.snapshot_processed_execution(input_ports=None, output_ports=None)`.
`PluginProcessedExecution` contains private selected input/output `ValuePacket`s
from one completed execution, plus the existing identity/counter/judgment fields,
score and message. Arbitrary output payload tokens remain paired with their values;
plugins can bind their immutable completed model/fit cache without another search,
training run or configured-recipe reload. Input/output subsets avoid unnecessary
overlay or tensor copies. The existing input-only API keeps its return type and
behavior through the shared capture path.

One nonblocking block lock and the dialog's owning thread guard the joint copy.
Failed inspections remain available, rejected inputs preserve the previous pair,
and exceptions make it unavailable. Missing/duplicate/undeclared ports and absent
optional values reject atomically. Both preview retentions are required; a private
per-completion output-availability marker prevents Lean summaries from becoming
apparent data after a profile switch, including no-input source blocks. It adds no
per-frame image copies, primary fields, payload fields or project schema changes.

Verification: **178 focused plugin/runtime/coordination/ticket/API checks pass** in
24.82 seconds, including actual offscreen editor publication. The installed wheel
passes **50 plugin capture/dialog checks** in 2.56 seconds and captures matching
input images and detection collections/reports from both raw and decoded YOLO
projects. All **371 Python module files** match wheel/source and all **61 package
versions** are unchanged. The installed app is refreshed. This adds no vision
completion credit; targeted cycle advances to **13/20**. The read-only action
was subsequently added in the increment above.

**Pinned YOLO output contracts (S2-26 Complete, stated YOLOX baseline)** qualify
the existing DNN and object-detection blocks rather than introduce another wrapper.
The pinned OpenCV Zoo YOLOX-S raw grid/objectness export runs under explicit
Classic / OpenCV / CPU. Local preparation also appends standard ONNX grid/size
decoding, objectness-times-class scores and an axis transpose, retaining original
learned initializers and nodes. This separately pinned derivative exposes decoded
rows `[1,8400,84]` and channels `[1,84,8400]` under explicit New CPU. It is modified
YOLOX, not a YOLOv8 model or qualification of every other YOLO family.

Twelve [YOLO projects](<../examples/opencv tools/yolo_detection/README.md>) compare
those layouts on a credited CC0 photograph. A known portrait crop complements
vertical padding with horizontal padding and unequal rounded resize scales.
Independent native preprocessing, explicit grid expansion and native batched NMS
check source boxes, scores, class IDs and original row identities. Separate
hand-marked cup truth exposes seven duplicate false positives when NMS is disabled.
Class-aware versus cross-class suppression, strict score/objectness gates,
region exclusion, valid empty results and malformed-layout containment/recovery
exercise inspection semantics. Class controls also label the wooden table's known
full-image extent, exposing its loss under cross-class suppression. Photo/crop
share one capture group, without an accuracy
or independent-sample claim. Sealed model/settings bundles and declared source
credits/license support portable Save As; loading/running never downloads or
converts weights. Existing app code and installed package versions are unchanged.

Verification: **123 unique focused checks pass across targeted runs**: 53 YOLO
qualification checks and 70 existing decoder/example checks. All twelve installed
projects reproduce Debug/Lean judgments; validation distinguishes duplicate false
positives, a table missed by cross-class NMS, threshold/selection misses and an
explicit layout error. Native decoded math uses a same-engine raw reference;
separate scored-box checks bound Classic/New floating-point differences. Exact
export regeneration retains every original learned initializer/node, and exact
project/crop regeneration, XML, portable model/credit/license assets and an installed
cold New-engine worker (exit zero) are checked. All **371 installed Python module
files** match source and **61 installed-package versions** remain unchanged.
Generated references, strict JSON/local links, source/model hashes, original text
contracts and repository sanity are checked. No broad suite or native equipment run.

ML/DNN advances to **26/62 (41.9%)**, overall **252/470 (53.6%)**; core remains
**173/175 (98.9%)**. Targeted cycle **12/20**. Next: S2-27 cross-tile source-grid
box fusion, then remaining image-model adapters. Permitted real SuperPoint,
other YOLO families/anchors/quantized heads and accelerator/ORT qualification
remain open.

**Pinned SSD detection (S2-25 Complete, stated export)** extends
`ml.decode_detections` with explicitly named box/class/score heads, optional
valid-count trimming and an XYXY/YXYX dropdown. It reuses existing score/class
filters, source-grid mapping, regions/masks, border handling, native NMS,
ticket checks, count judgments and drawable result objects. Storage shapes,
single-image batches, count/range/row limits and scored labels/coordinates reject
with contained failures; no export schema or anchor grid is guessed.

Five [SSD projects](<../examples/opencv tools/ssd_detection/README.md>) use the
unmodified ONNX Model Zoo SSD ResNet-34 opset-12 export, pinned by repository
revision, exact size and SHA-256. An explicit preparation script creates an ignored
sealed model/preprocessing bundle. The model runs with OpenCV 5 New / OpenCV / CPU;
Classic rejects this export. Independent native preprocessing/tensor/source-box
references and a separately hand-marked CC0 coffee cup qualify the integration
at IoU 0.6. Region exclusion, confidence gating, unmet count and a black-image
negative distinguish inspection failure, missing truth and valid empty results.
The model already suppresses proposals; additional NMS is disabled. Result
overlays, current ticket/grid context, validation sets, portable model/credit/license
assets, XML and cold-worker shutdown are checked. Detailed receipts retain the
native int64 label dtype; these small class IDs travel exactly through the existing
float32 tensor contract. This is a bounded integration baseline, not an accuracy
study or qualification of arbitrary SSD/quantized/accelerated models.

Verification: **186 focused checks passed** in 44.31 seconds, covering both SSD
formats, the prior detection/model-contract/engine fixtures and exact regeneration.
All **ten installed Debug/Lean sequences** matched, with 20 graph activations and
ten image-set validations. An installed cold New-engine validation worker exited
zero. Three installed module files match wheel/source bytes, all **61 package
versions** remain unchanged, and the app is refreshed. The unsupported Classic
import can emit a native Windows fault diagnostic; its rejection probe is isolated,
while sealed settings reject width/engine changes before import and recover after
restoration. Public photo/model hashes, ignored weights, source credits/license,
JSON/XML/portable Save As, generated references, local links, repository sanity and
original text contracts are checked. No broad suite, ROS or equipment operation.

ML/DNN advances to **25/62 (40.3%)**, overall **251/470 (53.4%)**; core remains
**173/175 (98.9%)**. Targeted cycle **11/20**. Next: S2-26 YOLO export qualification,
then the remaining image-model adapters; permitted real SuperPoint extraction and
accelerator/ORT qualification remain open.

**Explicit classic ONNX dispatch (S2-01/S2-05 reliability)** corrects the
OpenCV 5.0.0 vector-buffer branch that could route Classic through automatic engine
selection. Classic now imports the already verified bytes through a unique owned
temporary file and the explicit native path overload, deleting the file before
forward or on a native import exception. It never reopens the mutable original
path after hashing or retries a failed classic model with New. Auto/New retain
memory loading; global engine environment variables are not changed.

Execution receipts include portable import transport, engine argument and frozen
byte count. Native handles/import evidence follow the existing cache/reset/close
rules and are absent from saved project configuration. Native engine observation
and per-layer device placement remain unverified. CPU-only imports retain the
previous accelerator-discovery/shutdown fix.

Five [public engine projects](<../examples/opencv tools/dnn_engines/README.md>)
compare the same RGB-mean bytes under Classic/New/Auto and distinguish actual classic
rejection from New acceptance of shared-constant proposal heads. An audit of 21
bounded public model files/100 references found two classic-rejected control models;
their **21 saved projects** now explicitly choose New. The migration changes only
engine values, retaining every other saved field, layout and all original model/image
bytes. Generation scripts match. The optional pinned real YOLOX coffee project keeps
Classic and reproduces its native import/forward through the corrected path.

Verification: **748 focused DNN/validation tests passed** in 44.16 seconds.
The **61 existing local-feature checks** also passed in 13.80 seconds, covering
the remaining bounded public model family affected by this shared loader.
All **10 installed Debug/Lean project sequences** matched, including 20 graph
activations and ten image-set validations with explicit rejection controls.
An installed cold classic-worker check exited zero. Both installed module files
match wheel/source bytes and all **61 package versions** stayed unchanged.
Engine-only migration was verified byte-for-byte for all 21 projects; model/image
hashes, fixture regeneration, JSON/XML/Save As, original text contracts and generated
parameter/binding/coverage references are checked. The installed app is refreshed.

See [loading contracts](onnx-engine-loading.md). This adds no completed capability:
ML/DNN stays **24/62 (38.7%)**, overall **250/470 (53.2%)**, core **173/175 (98.9%)**.
Targeted cycle **10/20**. Next qualify pinned real-model SSD/YOLO paths (S2-25/S2-26),
retaining permitted real SuperPoint extraction and accelerator/ORT qualification
as open. No full-suite repeat, ROS or physical equipment operation.

**Classifier/detector evaluation (S2-23 Complete, bounded local baseline)** now
includes source-image object labels, current ticket/grid-checked box capture and
`ml.evaluate_detections`. Class-aware score-greedy one-to-one IoU matching reports
true/false positives, misses, per-class precision/recall/F1, all-point or 101-recall
AP, mAP and score/IoU sweeps. Fully labeled negative images differ from unlabeled
images; missing/error/cancelled images retain their truth and fail. Retained last
outputs cannot count as new detections, and unobserved dimensions remain unknown.

The optional validation tab has a source-box label table and reports for classes,
images, score sweeps and IoU cutoffs. `ProjectSession.run_validation()` and the
assistant CLI use the same workflow without altering production history. The graph
block accepts lists, batches or full validation reports, emitting a diagnostic PR
plot with independent coordinates and optional graphics-off execution. JSON/XML
and portable Save As retain annotations/configuration and model assets.

Installed cold-worker qualification exposed a native shutdown abort after a CPU
run enumerated OpenCL devices on a transient worker. CPU/default-backend inference
and CPU fallback now avoid unrequested accelerator discovery. Receipts identify
CPU-only availability rather than a complete device inventory; requested accelerators
retain explicit discovery and native placement stays unverified. A fresh-process
worker check verifies clean shutdown independently of successful image assertions.

The examples explicitly use the new CPU engine. An additional upstream audit found
that the OpenCV 5.0.0 vector-buffer classic branch forwards through the default engine
([source](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/onnx/onnx_importer.cpp#L3942)).
This discovery is addressed by the explicit-dispatch increment above. CPU inference
remains the qualified scope; selected settings do not certify native placement.

Eight [public detector projects](<../examples/opencv tools/detector_evaluation/README.md>)
use an original pinned ONNX proposal fixture with image-driven scores and independently
scaled truth boxes. Correct, duplicate, upstream-miss, localization, wrong-class,
negative-FP and decoder-error controls reproduce exact independent counts/AP; the
eighth project exposes actual duplicate records/curves. This is software qualification,
not a trained detector accuracy claim. It completes the declared baseline together
with the six classifier/group-split projects. Crowd/ignore, oriented/segmentation IoU,
benchmark importers, full COCO protocol and folds/bootstrap confidence remain separate.

Verification: **733 focused validation/DNN tests passed** in 61.94 seconds after
the shared CPU-discovery fix. All **16 installed Debug/Lean CLI runs** matched
expected outcomes; **four installed offscreen/cold-worker checks** passed with
process exit zero. Seven installed module files match the built wheel/source;
all **61 package versions** stayed unchanged. Original text contracts, generated
references, strict JSON, local links and public attribution are checked. The first
installed window assertions had passed before OpenCV aborted at process shutdown;
that failure is now covered by an independent cold-process regression.

See [detector contracts](detector-evaluation.md). ML/DNN is **24/62 (38.7%)**,
overall **250/470 (53.2%)**, core **173/175 (98.9%)**. Targeted cycle **9/20**;
The follow-up dispatch audit is addressed above; pinned real-model SSD/YOLO
paths and permitted real SuperPoint extraction remain open. No broad repeat, native ROS or physical equipment run.

**Classifier evaluation foundation (S2-23, `8d11d1a`)** adds grouped metadata splits and
confusion/per-class metrics through `ml.split_image_groups` and
`ml.evaluate_classification`. The existing validation image-set runner now optionally
records class truth, capture groups, pinned image hashes and exactly one current
ticket-matching recognized-label/score event per image. Wrong classes, rejection,
missing output and execution errors remain distinct and retain labeled-image
denominators. Optional score sweeps cannot recover already rejected candidates and
do not imply calibrated probabilities. Declared training/evaluation groups cannot
overlap; group metadata alone cannot prove model training provenance.

The validation window has an optional classifier tab, advanced score/group settings,
a confusion table and per-class support/precision/recall/F1. API callers use
`ProjectSession.run_validation()` or `opencv-dataflow-api run-validation`, preserving
the open/session production history. Cancelled runs retain missing expected IDs and
fail as incomplete. JSON/XML and portable Save As retain case metadata/configuration.
Six [public native projects](<../examples/opencv tools/classifier_evaluation/README.md>)
exercise disjoint original HOG/KNN images, actual swapped-label training, rejection,
preprocessing errors, split plans and a diagnostic matrix. Training feature rows are
checked against the declared images; all assets are pinned and regeneration is exact.

Verification: **228 targeted tests passed** in 13.27 seconds, **12 installed Debug/Lean
CLI runs** matched the expected correct/rejected/wrong/error outcomes, and an installed
offscreen background validation/report check passed. Six installed module files match
the built wheel and source exactly; all **61 package versions** stayed unchanged.
Generated parameter/binding/coverage references and original text contracts are checked.

At this classifier-only increment, S2-23 remained Partial: ML/DNN **23/62 (37.1%)**,
overall **249/470 (53.0%)**, core **173/175 (98.9%)**, targeted cycle **8/20**.
The detector counterpart is now qualified above. See [classifier contracts](classifier-evaluation.md).
This software/offline evidence does not claim production classifier accuracy.

**Random forest image classification (S2-22)** adds native train/predict
and portable JSON save/load blocks using the shared image-descriptor contracts.
Split depth/minimum samples, active features, hard tree limit, bounded optional
OOB stopping, class priors, row weights and normalization are exposed. Optional
native permutation importance has a numerical output and diagnostic plot; its
extra work is bounded before training. Runtime training does not set the caller's
RNG seed or opt into parallel branches. Stateless inference uses owned native
models and ordinary parallel scheduling.

Models retain explicit mappings, preprocessing, training settings/summary and
native trees. The loader checks the supported ordered-feature schema, tree/node
counts, binary preorder topology and split indices before native reading. Some
native Python getters are unavailable and native serialization omits termination
settings; the wrapper retains settings separately and reads the saved OOB statistic
only as a requested internal metric, not held-out accuracy. Without requested OOB
work that statistic is null. SHA revisions and sealed Save As assets are retained.

Prediction checks native class-vote headers/totals and candidates, exposes inclusive
vote-fraction/margin gates and default tie rejection, withholds unknown scalar
labels and distinguishes known disallowed classes. Current image context/pixels,
drawable filmstrip annotations, Lean omission, admission and exceptions use the
existing contracts. Votes are not calibrated probabilities or general novelty
detection.

Ten original public HOG projects train/reload ordinary, normalized, importance,
weighted and bounded-OOB variants. Frozen fixture votes are [60,4]/[2,62] on the
held-out orientations and [12,52] on the diagonal control; the declared 0.9 cutoff
separates them. Actual native fixture preparation is reproducible in an isolated
seeded child and never alters GUI runtime seeds. No external data/models were
downloaded. See [contracts](rtrees-classification.md) and
[ten projects](<../examples/opencv tools/rtrees_classification/README.md>).

**94 focused checks passed**: 63 forest checks, 23 catalog/offscreen inspector
checks and eight native K-nearest regressions. Native class/vote/importance and
prior comparisons, controlled vote boundaries, malformed-tree rejection before
native reading, context/admission, budgets, seed isolation, exception recovery,
JSON/XML/Save As and exact fixture regeneration are covered. All **22 installed
outcomes** matched: 20 Debug/Lean project outcomes and two held-out K-nearest API
regressions (16 passes, four intentional inspection failures, two deliberate
preprocessing errors). Three package/installed modules match source and all 61
package versions are unchanged. Thirteen strict JSON files, nine image/two model
hashes, 719 local links, ten original text contracts, generated references/counts,
Ruff and repository hygiene passed. Root README reviewed unchanged. Additional
generic method-name inventory hits earn no extra capability credit.

ML/DNN is **23/62 (37.1%)**, overall **249/470 (53.0%)**; core remains
**173/175 (98.9%)**. Targeted cycle **7/20**, no broad repeat. Next:
classifier/detector evaluation (S2-23), starting with image-level splits/confusion
reports. Actual permitted SuperPoint extraction, categorical/missing-value feature
adapters, regression, incremental training, calibrated confidence and production
accuracy remain separate qualification.

**K-nearest image classification (S2-21)** adds native train/predict and
portable JSON save/load blocks. Exact brute-force search retains class strings,
float32 rows, default K, optional per-row L2 and descriptor preprocessing contracts.
Shared public descriptor/normalization/revision helpers preserve existing SVM
formats and validation limits. Native KD-tree output did not meet the classifier
layout contract in the installed probe, so that backend remains outside this
qualified baseline; its controls are not exposed as a misleading alternative.

Prediction exposes K, four Euclidean distance statistics, inclusive distance and
vote-fraction cutoffs, default vote-tie rejection and accepted-class judgments.
Raw candidates/neighbors and squared/native distances remain available. Unknown
labels are null and scalar Label emits nothing; a recognized but disallowed class
still emits its label with an inspection failure. Votes are agreement, not
probabilities. Optional image review preserves current pixels/context, draws a
label for filmstrip composition and skips drawings in Lean. Context conflicts,
input admission and bounded work prevent invalid native prediction.

Nine original public projects train/reload two HOG orientations on six images,
inspect separate stripe positions, reject an unseen diagonal pattern and equal
votes, distinguish a known disallowed class and reject wrong preprocessing.
Both ordinary and L2 models are included without external downloads. Model files
contain training feature rows and follow the source data's sharing policy.
See [contracts](knn-classification.md) and
[nine examples](<../examples/opencv tools/knn_classification/README.md>).

**129 focused checks passed**, including 39 SVM regressions, native KNN response/
neighbor/distance/vote comparisons, all distance policies, inclusive cutoff/tie
behavior, context-before-native guards, unknown-label emissions, resource limits,
exception recovery, saved graphs, exact regeneration, JSON/XML and portable
Save As. Offscreen catalog/inspector checks cover new controls. All **20 installed
outcomes** matched: 18 Debug/Lean project outcomes plus two held-out SVM API
regressions (12 passes, six intentional inspection failures, two deliberate
preprocessing errors). Known fixture distances are below 0.28; the unseen diagonal
is above 2.8 against the declared 0.5 cutoff. Four package/installed modules match
source and all 61 package versions are unchanged. Twelve strict JSON files, nine
image/two model hashes, 712 local links, eleven original text contracts, generated
references/counts, Ruff and repository hygiene passed. Root README reviewed
unchanged. The inventory's ambiguous Subdiv2D findNearest name hit adds no extra
capability credit.

At S2-21, ML/DNN reached **22/62 (35.5%)**, overall **248/470 (52.8%)**;
core remained **173/175 (98.9%)**, cycle **6/20**. Random forests (S2-22) follow above. Actual permitted SuperPoint extraction, KD-tree adaptation,
incremental/weighted training, regression, calibrated confidence and production
classification/OCR accuracy remain separate qualification.

**Reusable image clusters (S2-19)** adds six native tools for palette or
descriptor training, fixed-center prediction and shared JSON save/load. Labels
are sorted once at training and remain attached to the stored center order.
Descriptor L2 normalization and preprocessing contracts persist. Color palettes
use fixed full-scale depth conversion and normalized BGR/gray or scaled native
Lab, preserving original source pixels, alpha, encoding and current context.
Masks/regions intersect; missing connected selections block work. Training's
evenly spaced sample cap is explicit and does not promise rare-color coverage.

Inclusive distance and stored-ID filters produce unknown masks/flags and a
fraction judgment with reasons. Accepted-count histograms retain their model and
filter contract, with all selected samples as denominator. Distances are not
probabilities and scaled Lab is not Delta E. Quantization is a separate numeric
image; drawable unknown outlines feed filmstrip review. Lean omits optional
drawings/maps. Strict bounded models contain only numerical centers/settings;
sealed assets move with Save As and native exceptions remain contained.

Nine original public projects demonstrate palette training/reload, rearranged
colors, a known 320-pixel defect, RGB16, Lab, explicit region exclusion and native
HOG vocabulary training/query on disjoint stripe positions. The same-length
wrong-HOG contract deliberately errors without predictions. No images or model
weights were downloaded. See [contracts](reusable-clusters.md) and
[examples](<../examples/opencv tools/reusable_clusters/README.md>).

**129 focused checks passed**, including existing K-means regressions, catalog
and offscreen inspector controls. Four targeted checks passed again after internal
helper names were clarified. Independent native fit/Lab/HOG references, stable
reload IDs, large-coordinate distances, alpha/depth/context, unknown fractions,
budgets, JSON/XML/Save As, strict files, admission and exception recovery are
covered. An initial test exposed a histogram payload omission; histograms now
retain the current source coordinate context as well as their feature contract.
All **18 installed outcomes** matched (14 passes, two intentional inspection
failures and two deliberate preprocessing errors). Three package/installed
modules match source and all 61 package versions are unchanged. Thirteen strict
JSON files, 12 image/three model hashes, 701 local links, ten original text
contracts, generated references/counts, Ruff and repository hygiene passed. Root
README reviewed unchanged; no binding-inventory or extra capability credit.

At S2-19, ML/DNN reached **21/62 (33.9%)**, overall **247/470 (52.6%)**;
core remained **173/175 (98.9%)**, cycle **5/20**. K-nearest classification
(S2-21) follows above. Actual permitted SuperPoint extraction,
incremental/multi-reference cluster training, calibrated confidence and
production accuracy remain separate qualification.

**ALIKED subpixel features and matching (S2-17)** adds
`ml.aliked_features` and `ml.lightglue_aliked` under Custom ML. Real local-weight
Kornia n16/n16rot/n32 presets produce 128-value descriptors for matching LightGlue
weights. Model/mode/NMS/native threshold controls are explicit, and an independent
inclusive Minimum score prevents native mean-fallback behavior from silently
weakening a requested hard confidence cutoff.

Shared learned-feature lifecycle, normalized image conversion, source-center
mapping, post-detection cap/ROI/alpha filters, source pixels/depth/encoding/context,
hashes/caches/recovery and result graphics are reused through small backend hooks.
ALIKED replicate-pads internally to multiples of 32 and unpads its maps; the
padded-grid budget accounts for that native allocation. Every learned checkpoint
parameter must be present. Only the deterministic non-learned DKD radius grid,
absent from the older official files, may be initialized by the native constructor.
The optional ML extra now requires Kornia >=0.9; installed dependency versions
remain unchanged.

Thirteen projects cover the three presets, independent known translations,
resizing, RGB16, source/local regions, threshold fallback, strict score/empty/count
failures and genuine SIFT comparison. The same attributed VisA PCB photograph pair
checks four approximate mounting-hole centers. Its initial CPU fit has 212
matches/194 inliers and a 1.781-pixel maximum checkpoint error within the unchanged
4-pixel image-space limit. Eight copied public images and four local model hashes
are pinned; this does not qualify calibrated physical metrology.

**124 focused checks passed**: 70 ALIKED/catalog/offscreen-control checks and
54 DISK regressions. Four strengthened context/admission assertions also passed
their targeted recheck. Public native rows/descriptors/scores, all presets,
source geometry, alpha/depth/encoding, direct capture, hashes/cache/model
completeness, exception recovery, JSON/XML/Save As and exact regeneration are
covered. All **28 installed outcomes** matched (22 passes/six deliberate failures),
including older DISK/SIFT regressions; CLI checks used two CPU threads. Three
packaged/installed runtime modules and the new optional dependency metadata match
source. Fourteen strict JSON files, eight image/four model hashes, 681 local links,
13 original text contracts, TOML 1.0 parsing, generated metadata, Ruff and repository
hygiene passed. The app was refreshed without changing installed dependency
versions; root README reviewed unchanged. The binding inventory's new ambiguous
method-name hit is not evidence of a FaceRecognizer block or extra completion.

At S2-17, ML/DNN reached **20/62 (32.3%)**, overall **246/470 (52.3%)**;
core stayed **173/175 (98.9%)**, cycle **4/20**. Reusable clusters (S2-19)
follow above; actual permitted SuperPoint extraction remains open. The CPU baseline does not qualify t16 matching, arbitrary exports,
cross-version parity, accelerators, repeated-instance discovery or physical
localization. See [contracts](aliked-features.md) and
[thirteen examples](<../examples/custom tools/aliked_matching/README.md>).

**Real DISK extraction and matching (S2-16)** adds `ml.disk_features`
and `ml.lightglue_disk` under Custom ML. Local depth/epipolar checkpoints run
actual 128-descriptor Kornia DISK inference; matching LightGlue weights fit a
planar transform/pose. Native top-N, odd NMS, raw threshold and padding controls,
explicit downscale interpolation and half-pixel source mapping are exposed.
Score values remain raw, without being mislabeled probabilities.

Mask/region/alpha filter centers after native top-N; descriptor context remains
whole-image. Source and padded-network pixel budgets precede allocation/inference.
Primary pixels/depth/encoding/ticket/recipe/spatial/start context survive.
Portable reference capture does not fine-tune weights. Model path/hash/device
changes invalidate owned caches; native exceptions recover and reset/close release
references. Failed-input admission retains the last processed image and does no
new work; absent wired selections prevent unmasked fallback. Missing/failed
matching withholds poses, and optional drawable overlays feed the filmstrip.
Lean skips painting.

Ten projects use independent known translations, resizing, both detector variants,
RGB16, ROI/pose inspection, SIFT comparison and deliberate empty/count failures.
Separate attributed CC BY 4.0 VisA PCB photographs check four approximate visual
mounting-hole centers; the initial CPU fit has 298 matches/270 inliers and a
3.624-pixel maximum checkpoint error within a declared 4-pixel image-space limit.
This is not calibrated metrology. Eight source images and three pretrained model
digests are pinned; optional preparation is explicit and model files stay local.

**177 regression checks passed, followed by two focused admission checks**.
Native Kornia rows/descriptors/raw scores match the public API; source-coordinate,
mask/alpha, dtype/encoding, train/capture, context, cache/hash/exception recovery,
JSON/XML/Save As, exact regeneration and offscreen metadata checks pass. Initial
fixture assertions were corrected for grayscale review-channel expansion and
direct-training versus execution metadata; no shared-runtime change was needed.
All **22 installed outcomes** matched (18 passes/four deliberate empty/count
failures), including older SIFT and external-feature regressions. CLI checks used
two CPU threads. Three packaged/installed runtime modules match source. Eleven
strict JSON files, eight image/three model hashes, 669 local links, 11 original
text contracts, generated metadata, Ruff and repository hygiene pass. The app was
refreshed without dependency changes; root README reviewed unchanged.

At S2-16, ML/DNN reached **19/62 (30.6%)**, overall **245/470 (52.1%)**;
core stayed **173/175 (98.9%)**, cycle **3/20**. ALIKED (S2-17) follows above.
Real permitted SuperPoint extraction remains open.
The declared CPU baseline does not qualify arbitrary exports, cross-version
repeatability, accelerators, repeated-part discovery or physical localization.
See [contracts](disk-features.md) and
[ten examples](<../examples/custom tools/disk_matching/README.md>).

**External SuperPoint feature contracts (S2-15, Partial)** adds
`ml.pack_superpoint_features` and `ml.lightglue_superpoint` under Custom ML.
Explicit XY, 256-value descriptor and probability heads map through declared
resize/letterbox geometry into source pixel centers. Score/source/ROI filtering
retains corresponding raw-row identities. Optional L2 normalization, descriptor
family, empty heads, source-image and learned-attention row limits are explicit.

The optional matcher uses locally supplied matched LightGlue weights, an optional
SHA-256, weights-only loading and native robust planar geometry. Query body/wire
context agrees with the current image; references retain their own acquisition
identity. Missing/failed fits emit no pose. Source pixels/depth/encoding/tickets
survive; exceptions are contained and subsequent valid work recovers. Drawable
feature/match markers feed filmstrip overlays; Lean now skips optional painting
for the existing SIFT matcher too. Its IDs, defaults and extraction remain intact.

SuperPoint's detector, inference file and weights have separate restrictive terms
documented by the upstream project. No detector is downloaded or bundled here.
The explicit preparation option fetches only Apache-2.0 LightGlue matcher weights,
checks the pinned release hash and preserves existing artifacts/receipts. A real
permitted SuperPoint extractor/export/repeatability fixture remains open, so this
row earns **no completion credit**.

Seven original MIT ONNX controls use prescribed coordinates/descriptors, with a
small image-mean-dependent descriptor perturbation. They qualify tensor flow and
matcher geometry, not SuperPoint detection. The same textured +8/+6 image pair is
used by genuine SIFT extraction in the comparison project. Mask/score, empty/count
failures, JSON/XML/Save As and exact regeneration are covered.

**125 focused checks passed**, including 61 new interface checks, existing SIFT
native tests, catalog persistence and offscreen controls. All **16 installed
outcomes** matched (12 passes/four intentional empty/count failures), including
the older SIFT demo and landmark regression. Installed CLI checks used two CPU
threads. Three packaged/installed runtime modules match source. Eight strict JSON
files, three original asset hashes, 656 local links, 12 original text contracts,
generated metadata, Ruff and repository hygiene passed. The app was refreshed
without dependency changes; root README reviewed unchanged. The newly published
FoundationPose setup guide is linked from the example index; its collaborator's
native evidence was not rerun as part of this qualification.

At S2-15, ML/DNN remained **18/62 (29.0%)** (nine Partial), overall
**244/470 (51.9%)**; core **173/175 (98.9%)**, cycle **2/20**. Qualified DISK
extraction/matching (S2-16) follows above; actual SuperPoint extraction remains open. See [contracts](local-features.md) and
[seven controls](<../examples/custom tools/local_features/README.md>).

**Ordered landmarks and local poses (S2-12)** adds `ml.decode_landmarks`
and `ml.landmark_pose` under OpenCV DNN. Explicit single-object heatmaps and
X/Y/confidence rows retain their configured identity, names and raw channel IDs.
Declared axes, head alignment, normalized/pixel coordinates and exact inference
resize/letterbox inverse produce source pixel centers. Optional selections can
restrict peak search or filter the global result. Equal peaks reject by default;
bounded quadratic/log-quadratic fits check curvature/residual/offset and report
fallback status. Confidence remains the original sampled score.

Rejected records stay in their ordered slots. Points defaults to no event on
incomplete correspondence; an explicit accepted-only policy supplies indices.
Count/require-all gates give normal failure reasons. Accepted landmark pairs or
fixed angles define ordinary poses, including local offsets and shared-root
mapping for pose-relative inspection. Missing points withhold the pose. Original
image depth/encoding, tickets, processing start, recipe and pyramid context
survive. Known context conflicts and malformed/budgeted heads reject; native
exceptions emit no result and subsequent valid activations recover. Drawable
markers/labels/links feed the filmstrip; Lean skips diagnostic generation.

Twelve original MIT uint16 image-dependent ONNX projects cover four output
conventions, analytic Gaussian centers, independent finite-image centroids,
integer/refined/letterboxed/subset/ROI results, local poses, deliberate partial
and empty failures, and a sealed model/settings bundle. Moving an input spot
changes its inferred source coordinate. JSON/XML/Save As and exact independent
regeneration preserve the source collection and pinned assets.

**171 focused checks passed**, including 81 landmark checks and offscreen editor
controls. All **26 installed outcomes** matched (22 passes/four deliberate
partial/empty failures, including model-contract and instance-mask regressions).
Four packaged/installed runtime modules match source. Fourteen strict JSON files,
seven asset hashes, 686 local links, 12 original text contracts, generated
parameters/inventory/coverage, Ruff and repository hygiene passed. The installed
app was refreshed without dependency changes; root README reviewed unchanged.
The published experimental FoundationPose worker guide is linked from the example
index with its supplied-mask and rejected-result limitations; that collaborator's
native evidence is not part of this landmark qualification.

After S2-12, ML/DNN scope reached **18/62 (29.0%)**, overall **244/470 (51.9%)**;
core remained **173/175 (98.9%)**, targeted cycle **1/20**. The partial S2-15
increment follows above; the broad checkpoint remains current. S2-12 qualifies
explicit decoding and geometry, not arbitrary exports, multi-instance discovery,
learned task accuracy, accelerators or physical localization. See
[contracts](landmarks.md) and
[twelve projects](<../examples/opencv tools/landmarks/README.md>).

**Independent instance masks (S2-11)** adds `ml.decode_instances`,
`ml.pack_detection_heads` and `ml.select_instance_mask` under OpenCV DNN.
Explicit ROI/class-indexed masks and packed coefficient/prototype heads reuse
ordinary detector score/class selection and box NMS while preserving original row
IDs. ROI masks keep their original unclipped box support; full-network prototype
fields map through exact rounded preprocessing. Activation order, sampling,
thresholds, box crop, pixel/center ROI selection and area/count limits are explicit.
Separate boxes/scores/classes pack into detector rows while retaining other heads.

Each retained item carries an exact source-grid raster region. Overlap remains
in every individual mask, independently of the union/summary labels/overlap count.
Sort/filter preserves masks and supports custom mask area/confidence/centroid
metrics; a selector emits one exact mask/region for further inspection. Missing
connected selections prevent unmasked fallback. Source pixels/encoding/depth,
tickets, recipe, pyramid and processing start survive. Known conflicts reject;
native errors publish no current result and valid subsequent input recovers.
Tile, prototype-operation and total-run budgets reject without partial masks.
Optional outlined masks/holes/labels feed filmstrip overlays; Off/Lean skips them.

Correctly shaped zero-row outputs now survive inference serialization, detector
and instance adapters, separate-head packing and batch unpacking. General input
creation/classification/embedding retain strict nonempty rules. Explicit zero-row
data and stubbed native-array handoffs qualify this transport/ownership contract;
they do not claim an unsupported native model can execute zero-shape operators.

**484 focused regression checks passed**, followed by **102 focused checks** for
final instance, zero-row, required-selection and catalog/reference contracts.
Independent scalar masks/packed-pixel truth cover ROI clipping, class/layout,
prototype mixing/crop, activation order, overlap, selection, filters, malformed
heads/budgets, source context, JSON/XML/Save As and exact regeneration. Twelve
original MIT image-dependent ONNX projects use fixed proposals/templates and
RGB score/gain arithmetic, with no learned recognition claim. Explicit tiling
avoids an observed scalar-broadcast importer failure in the fixture.

All **26 installed outcomes** matched (22 passes/four intentional count failures,
including existing model-contract/semantic regressions); nine packaged/installed
modules match source. Strict JSON, three asset hashes, local links, 18 original
text contracts, generated parameters/inventory/coverage, Ruff and repository
hygiene passed. App refreshed and root README reviewed unchanged. See
[contracts](instance-segmentation.md) and
[twelve projects](<../examples/opencv tools/instance_segmentation/README.md>).

At the S2-11 checkpoint, ML/DNN scope was **17/62 (27.4%)**, overall
**243/470 (51.7%)**; core remained **173/175 (98.9%)**. It closed targeted
cycle **20/20**, with the broader checkpoint recorded above. S2-12 followed
as the first increment of the new cycle. Arbitrary export conventions, trained task
accuracy, accelerators and physical occlusion/pose remain separate qualification.

**Source-grid semantic segmentation (S2-10)** adds `ml.decode_segmentation`
under OpenCV DNN. Explicit multiclass logits/probabilities, binary foreground
scores and decoded class IDs use six declared layouts and a single-image batch.
Nearest winner labels or native linear raw-score sampling map full-network heads
through exact rounded resize/letterbox transforms to source-size labels/masks.
Ignored/unknown/below-confidence pixels become -1, class zero remains a valid
semantic ID, and ignored winners never promote a runner-up. ID-only heads have
no invented confidence. Class statistics and count/ROI-fraction limits retain
complete current outputs on normal `segmentation.area` failures.

Source encoding/depth, ticket/frame, recipe, pyramid and processing start survive;
known conflicts reject. Mask/region selections gate outputs after full-image
inference. Tensor, image and mapped-score limits bound decoding work, including
flat-value validation before conversion and a qualified head above two million
values. Optional semantic color maps and drawable outlines/holes feed filmstrip
overlays; contour/vertex limits flag graphics truncation without changing masks.
Off/Lean avoids optional graphics. Base execution owns timing/history/diagnostics;
native errors emit no result and a valid next invocation recovers.

**214 focused checks passed**, covering independent nearest/linear pixel
references, layouts and score meanings, extreme logits, binary thresholds,
ignored classes/ROI/confidence, count/fraction failures, unknown IDs without
confidence, malformed heads, resource guards, context conflicts, original RGB
uint16/pyramid/recipe/timing data, JSON/XML/Save As, exact regeneration, mask-to-blob
handoff and existing inference/detection/catalog/reference regressions. Twelve
original MIT color-arithmetic ONNX projects use source-grid PNG truth calculated
without executing inference or the decoder. Their numerical masks qualify this
adapter, not learned recognition accuracy. A scalar float-map reference also
caught and corrected the older 1/32-phase assumption for this CPU implementation.

All **24 installed Debug/Lean example outcomes** matched (20 passes, four
intentional area-failure judgments), including class counts/fractions, source
tickets and absent ID confidence. Two installed model-contract/rotated-pose
regressions passed; six packaged/installed modules match source. Strict JSON,
15 model/image/reference hashes, local links, 14 original text contracts,
generated parameters/binding inventory/coverage, Ruff and repository hygiene
passed. Application refreshed, root README reviewed unchanged; no broad suite or
physical-equipment/accelerator qualification repeated. See
[contracts](semantic-segmentation.md) and
[twelve projects](<../examples/opencv tools/semantic_segmentation/README.md>).

At the S2-10 checkpoint ML/DNN reached **16/62 (25.8%)**, overall
**242/470 (51.5%)**; core remained **173/175 (98.9%)**, targeted cycle **19/20**.
The next increment above qualifies instance segmentation (S2-11).
Arbitrary/cropped/multilabel/promptable exports and industrial accuracy remain
separate qualified capabilities.

**Rotated detection and affine poses (S2-09)** adds
`ml.decode_rotated_detections` under OpenCV DNN. Explicit center/size/angle
score/objectness/class-ID heads, row/channel layouts, radians/degrees,
clockwise/counterclockwise angles and 180/360-degree semantics remain declared.
Class-aware/agnostic suppression operates on unclipped network rectangles;
source footprints preserve the exact affine image instead of fitting new boxes.
Non-uniform resize can produce parallelograms, with full model-frame poses retained
through collection sort/filter, Pose from feature and local inspection regions.
Source encoding/depth, tickets, recipe/pyramid context and processing start survive.
Masks/regions select original centers; polygon clipping preserves predicted poses.

Default polygon IoU corrects the installed native full-containment shortcut;
optional `NMSBoxesRotated` retains OpenCV behavior. Nested controls retain the
small inner proposal only in corrected mode; crossing thin proposals survive
oriented suppression. Conservative bounds skip disjoint pairs, and explicit
candidate/result/pixel/raw-row/pair budgets reject excess work. Native exceptions
emit no current outputs and valid subsequent input recovers. Drawable outlines,
axis markers and labels feed explicit filmstrip overlays; Off/Lean skips drawings.

**152 focused checks passed**, including independent source corners/IoU, all
head/layout/unit conventions, corrected/native overlap, class filters, border
policies, count failures, precision guards, known provenance conflicts, source
pixels, supplied-pose offsets and pyramid rescaling, JSON/XML, exact regeneration
and existing pose/axis-detection/catalog/reference regressions. Twelve original
MIT projects have native image-dependent score heads and analytic corner truth;
positions/angles remain fixed controls rather than learned recognition.
All **24 installed Debug/Lean example outcomes** matched (20 passes, four
intentional count-failure judgments), with source-corner/ticket and affine ROI
checks. Two installed model-contract/pretrained-photo regression spot checks
also passed. Five packaged/installed runtime modules match source. Thirteen
strict JSON files, four model/image hashes, 853 local links, fourteen original
text contracts, generated references/counts, Ruff and repository hygiene passed.
Application refreshed and root README reviewed unchanged. No broad suite or
physical-equipment/accelerator qualification repeated.
See [contracts](rotated-detection.md) and
[twelve projects](<../examples/opencv tools/rotated_detection/README.md>).

At the S2-09 checkpoint ML/DNN reached **15/62 (24.2%)**, overall
**241/470 (51.3%)**; core remained **173/175 (98.9%)**, targeted cycle **18/20**.
The next increment above qualifies semantic segmentation (S2-10). Arbitrary rotated exports, industrial accuracy,
physical pose calibration and throughput remain application-specific work.

**Revisioned model/preprocessing/execution receipts (S2-05)** extend all three
inference interfaces with project-contained model/settings bundles, optional
upstream preparation declarations, strict byte/configuration checks and hidden
Receipt outputs. Summary/Detailed/Off controls avoid unnecessary tensor hashing;
bound model bytes are still verified once per activation before cache reuse.
Same-size/restored-timestamp changes reject. Recipes use their effective settings
against the active contract. The additive `export-model` CLI and Python export
function publish a fresh two-file bundle without processing; explicit new
revisions can reuse verified model bytes. Save As copies both assets without
overwriting conflicts; JSON/XML omit native state.

Receipts record actual tensor shapes/preparation, loaded bytes, environment/build,
native cache reuse, requested/selected execution, target availability IDs and
application-selected fallback. Detailed mode adds dtype/order-qualified input and
output hashes. Batch rows retain call/row/group references with source tickets.
Native per-layer placement remains explicitly unverified; CPU fallback is opt-in
for unavailable targets and never silently retries arbitrary native failures.
Ten original MIT projects demonstrate six valid controls and four contained
errors, including byte/settings/upstream-preparation drift and strict target policy.
**185 targeted checks passed**, including independent tensor hashes/arithmetic,
same-size/restored-mtime tampering with receipts Off, settings/new-revision export,
strict JSON/path limits, explicit availability/fallback, row/ticket ownership,
cache/native errors, Save As conflicts, XML and existing single/multi/batch/detector
and CLI workflows. Old example regeneration checks compare reopened definitions
with new optional defaults while preserving every original source/asset byte.
All **20 installed Debug/Lean example outcomes** matched (12 pass, eight contained
invalid activations), plus two installed batch/pretrained-photo regression passes.
Installed CLI bundle export also passed without processing. Six packaged/installed
runtime modules match source; 17 strict JSON files, 15 model/manifest/image hashes,
851 local links, 20 original text contracts, generated references/counts, Ruff and
repository hygiene passed. Application refreshed, root README reviewed unchanged;
no broad suite or physical-equipment/accelerator qualification repeated.
See [contracts](model-contracts.md) and
[ten projects](<../examples/opencv tools/model_contracts/README.md>).

At that S2-05 checkpoint ML/DNN reached **14/62 (22.6%)**, overall
**240/470 (51.1%)**; core remained **173/175 (98.9%)**, targeted cycle **17/20**.
The next increment above qualifies rotated detection (S2-09). Native placement, arbitrary exports,
signed provenance and general recognition accuracy remain separate qualification.

**Ordered batched/dynamic-shape image inference (S2-04)** adds `ml.dnn_batch`
and `ml.unpack_inference` under OpenCV DNN. Existing ticketed image groups feed
bounded NCHW calls with declared exact names/axes, independently prepared image
sizes/colors/normalization, masks/regions and explicit variable/padded/rejected
tails. Source mode qualifies changing common H/W between activations; configured
mode resizes mixed source sizes. Native leading output axes must match submitted
counts, and padding outputs never become acquisitions. Each real row retains
its source image, ticket/grid/pyramid/companion/recipe/start time and preparation
maps. Member outcomes stay independent; conflicting recipes still reject.
Selection or serial unbatching exposes ordinary image/tensor ports to existing
decoders. Actual signatures key native handle reuse, every call rebinds input,
late failures publish no partial results, and subsequent valid input recovers.

**121 targeted checks passed** for native CPU auto/classic/new image arithmetic,
variable N/H/W, fixed native reshaping, output subset/order, tail policies,
preallocation/cumulative budgets, RGB/uint16/letterbox/selection semantics,
per-image status/timing/coordinate ownership, failed-input gates, warm failures,
recipe conflicts, serial downstream execution, JSON/XML/Save As and exact
regeneration. Existing named multi-input/single-image inference, shared metadata
and native-reference checks passed. Ten original MIT projects include six valid
controls and four contained invalid activations. Native 40×24 float32 pooling
uses an explicit 2e-5 absolute tolerance; 8×8 controls use 2e-6. The shared helper
can split native arrays before portable row conversion, removing an unnecessary
full-batch list round trip without claiming a measured performance improvement.
All **20 installed Debug/Lean batch outcomes** matched (12 pass, eight deliberate
invalid-input failures), with numerical rows, source ticket/grid ownership and
serial execution counts verified. Two installed regression spot checks also
passed: paired multi-input inference and the pinned pretrained YOLOX coffee
example. Five packaged/installed runtime modules match source bytes. Eleven
strict JSON files, ten model/image hashes, 833 local links, fourteen original
text contracts, generated references/counts, Ruff and repository hygiene passed.
The application was refreshed and root README reviewed unchanged.
See [contracts](batch-inference.md) and
[ten projects](<../examples/opencv tools/batch_inference/README.md>).

At that S2-04 checkpoint ML/DNN reached **13/62 (21.0%)**, overall
**239/470 (50.9%)**; core remained **173/175 (98.9%)**. Targeted cycle was
**16/20**, without a broad suite. The following increment above qualifies S2-05.
This is the
declared single-input float32 CPU batch baseline, without arbitrary exports,
paired multi-input batches, model semantic-order guarantees or accelerator
qualification. Native allocated work and padding effects remain model-dependent.

**Complete named multi-input inference (S2-03)** adds `ml.image_tensor`,
`ml.make_tensor` and `ml.dnn_multi` under OpenCV DNN. Independent image branches
share existing preprocessing with explicit NCHW/NHWC, color/units, mean/std,
selection and rounded resize/letterbox maps. Numeric/boolean data or configured
constants become bounded float32 tensors. Up to four slots bind exact model names,
selected tensor records and declared axes; missing, extra, wrong-type or mismatched
inputs produce no partial result. Native input-shape/name signatures participate
in the cache key, every activation rebinds every input, and failed native binding/
forward paths discard the handle. Model work is retained only per block instance.

Current/Reference dropdowns and full-ticket/value/unchecked policies distinguish
live acquisitions from cached auxiliaries. Different source grids remain valid;
primary input owns the output ticket/grid and optional preprocessing. Per-input
names, roles, shape/type, ticket/recipe/grid and preparation remain in `input_info`.
Shared runtime recipe consistency stays enforced, including references. Matching
values are an association convention, not physical synchronization or ledger merge.
The common inference helper also works around this wheel's complete-output-set
requirement: one native forward computes terminal outputs before requested subset/
order selection. Retained-output limits do not bound unselected native allocations.

**93 focused checks** passed for native RGB/gray subtraction, numeric gains,
NHWC and four-input sums, independent arithmetic, exact name/shape binding,
complete rebind/cache reuse, native errors/missing bindings and recovery, source
identity/reference rules, timezone-normalized tickets, recipe conflicts, pixel/
element guards, masks, float32 conversion, provenance, JSON/XML/Save As and
regeneration. Existing single-input inference and detector examples (including
the pinned pretrained photo path), output selection and catalog/reference checks
passed. Nine original MIT projects retain seven successful controls and two
contained invalid activations. All **18 installed Debug/Lean outcomes** matched:
14 passes and four deliberate invalid-input failures. Five packaged/installed
runtime modules match source bytes. Original asset hashes, strict JSON, generated
references/counts, local links, Ruff and text contracts passed; root README reviewed
unchanged and application refreshed. No broad suite, physical synchronization or
accelerator qualification repeated. See [contracts](multi-input-inference.md) and
[nine projects](<../examples/opencv tools/multi_input_inference/README.md>).
At that S2-03 checkpoint ML/DNN reached **12/62 (19.4%)**, overall
**238/470 (50.6%)**; core remained **173/175 (98.9%)**, targeted cycle **15/20**.
The following increment above qualifies batched/dynamic-shape image inference
(S2-04); revisioned model/preprocessing/execution receipts (S2-05) remain next.
This baseline covers declared float32 bindings, not automatic ONNX signature
discovery, every model input type/operator, arbitrary batch workflows or accelerators.

**Image object detection and NMS (S2-08)** adds `ml.decode_detections` under
OpenCV DNN. Explicit XYWH score/objectness, XYXY/score/class and SSD row layouts
support row/channel-first ordering, normalized coordinates, class labels/offsets,
probability/logit choices and optional exact YOLOX stride decoding. Resize and
rounded letterbox inverses map floating pixel-edge rectangles back to the original
image. Known ticket/grid/recipe mismatches reject; missing provenance remains
explicit in the report. Center selections, class filters, native per-class or
class-agnostic hard NMS, adaptive eta/top-K and border policies remain separate.

Outputs are ordinary sortable/filterable box collections, counts, source image
pass-through and structured reports. Drawable labels/outlines share source
coordinates and can be composited into the filmstrip. Count criteria return
normal failed inspections; bad tensors/geometry and exceeded image/row/candidate/
result budgets emit no partial collection. Stride counts are checked before grid
allocation, and boxes with negative coordinates cannot suppress another class.
DNN model paths now declare their ordinary project reference so Save As rebases
them with image references. Model/inference receipt work remains S2-05.

**84 focused checks** passed: native ONNX image-dependent control heads,
independent stretch/letterbox edge references, normalized/explicit row layouts,
YOLOX grid/size math, probability/objectness/logit scoring, stable ties, border
clipping, selection cell conventions, source identity, resource bounds, native
exception recovery, graphics, collection sorting, JSON/XML and Save As. Existing
DNN/classifier/embedding and catalog/reference checks passed. Nine original MIT
control projects contain fixed proposals with RGB-mean-dependent scores and do
not claim learned recognition. The optional real-photo project uses hash-pinned
Apache-2 OpenCV Zoo YOLOX-S and the existing CC0 Rachel Michetti coffee photo;
its CPU cup box matches independent grid expansion/native batched NMS. Temporary
regeneration preserves model/image hashes and original programs.

All **20 installed Debug/Lean outcomes** matched (16 passes, four deliberate
inspection failures), including both pretrained photo runs. Five changed packaged/
installed runtime modules match source bytes. Strict JSON, model/photo/asset hashes,
generated references/counts, links, Ruff and original text contracts passed.
The 35,858,002-byte pretrained model stays ignored and is downloaded only by the
explicit preparation script. App refreshed; root README unchanged. No broad suite,
accelerator or physical equipment qualification repeated. See [contracts](object-detection.md)
and [ten projects](<../examples/opencv tools/dnn_detection/README.md>).
ML/DNN is **11/62 (17.7%)**, overall **237/470 (50.4%)**; core remains
**173/175 (98.9%)**. At that checkpoint the targeted cycle was **14/20** and multi-input inference
(S2-03) was next; the following increment closes it. Batch/dynamic shapes and
model/preprocessing receipts (S2-04/S2-05) follow.
The remaining core EXR and precomputed LK-consumer limits stay open. This baseline
qualifies the declared adapters and CPU fixtures, not every detector export,
general semantic/industrial accuracy, rotated/segmentation heads or accelerators.

**Curved QR qualification (S1-121)** completes the existing standard/curved
reader path with seven original pinhole/cylinder label projects, independent
forward-projected surface corners and stored original module matrices. Five
fixtures decode; severe curvature and a 27-degree case remain deliberate
failures. The flat decoder reads only the mild case. Three finder-associated
corners are within three pixels of analytic references, while the native fourth
corner differs from the true curved surface corner by over 40 pixels in the
strong-wrap cases. Result geometry is explicitly a native quadrilateral, with
image-plane axes rather than a calibrated surface pose. A quarter-turn fixture
retains a sampled module error despite recovering the correct text.

The existing reader gains hidden-by-default structured `info`, backend/mode and
corner-model fields, per-observation module roots and `qr_observation` source
spatial/pyramid/corner provenance. Source ticket and recipe survive, but no
inverse surface map or error-corrected module-image claim is made. Valid misses
emit no stale modules. Native exceptions are contained and empty selections skip
decoding. Old saved projects retain their selected/exposed controls and connectors.

**94 focused checks** passed for native reads/declared misses, analytic geometry,
original module values, source and module grids, mask/region intersection,
RGB(A)/gray, ticket/recipe/pyramid ownership, graphics/filmstrip behavior,
exception recovery and JSON/XML. Existing flat/ArUco readers, linear/optional
barcode adapters and six backend-comparison projects passed their selected
regressions. Original hashes and full temporary regeneration preserve the source
collection. All **14 installed Debug/Lean CLI outcomes** matched (ten passes,
four deliberate inspection failures), with the changed packaged/installed module
matching source bytes. Strict JSON, generated references/counts, local links,
Ruff, original text contracts and repository hygiene passed. App refreshed; root
README unchanged, no broad suite or physical camera/print test repeated.
See [contracts](curved-qr.md) and [seven projects](<../examples/opencv tools/curved_qr/README.md>).
Core is **173/175 (98.9%)**, overall **236/470 (50.2%)**; cycle **13/20**.
The remaining core items need a capable EXR codec (S1-125) and a reusable LK
pyramid consumer (S1-155). At that checkpoint, ML/DNN image detection/decoding (S2-08) was next; the
following increment closes that baseline. Multi-input/batch/model-receipt
contracts (S2-03–S2-05) follow. Continue to 3D and selected
contrib in the declared order; this is scoped roadmap coverage, not every API.

**Capture-setting negotiation and readback (S1-115)** extends `input.camera`
and adds `input.capture_file` under OpenCV Video I/O. Shared typed settings cover
size/rate/FourCC/buffers, explicit native exposure/gain and auto controls, manual
focus/white balance and whitelisted additional controls. Reports retain setter
success, after-set/after-frame readback, actual decoded geometry/type, independent
host conversion, backend identity and ambiguous zero/minus-one values. Strict
setter/readback policies withhold images and emit failed tickets with reasons;
report-only preserves usable actual frames with warning details. Handles reuse
unchanged settings, reapply on reopen, and release after failure/reset. Geometry
is always observed; optional per-frame readback and graphics follow explicit
policies. Diagnostic charts have independent grids and Lean skips their generation.

**69 focused checks** passed: controlled backend accepts/ignores/clamps/rejects,
unavailable/nonfinite/exception replies, tolerances, manual-control ordering,
geometry disagreement, cached/current observations, resource guards, recovery,
legacy/new JSON/XML persistence, Save As, toolbox/reference metadata, unchanged
Video file and selected paced-source behavior. Native FFmpeg verifies exact FFV1
pixels, actual resolution/rate/codec, rejected camera settings, host conversion,
range/loop/EOF and five original MIT projects. Example regeneration retains source
assets and compares exact pixel hashes and normalized saved programs. Ten installed
Debug/Lean Run All outcomes cover all 30 ticketed frames (18 pass, 12 deliberate
fail), with six packaged source modules matching the checkout. No physical camera
was opened; readback is software/backend evidence, not device/driver acceptance.
Native camera calls do not acquire a hard timeout in this increment.
See [contracts](capture-negotiation.md) and [five examples](<../examples/opencv tools/capture_negotiation/README.md>).
At that checkpoint, core was **172/175 (98.3%)**, overall **235/470 (50.0%)**;
cycle **12/20**. Curved QR (S1-121) is closed by the following increment. Native EXR (S1-125) and reusable LK pyramid
consumption (S1-155) remain dependent on capable native paths. Then proceed to
ML/DNN priorities. The root README remains concise and unchanged; no broad suite
was repeated for this increment.

**Controlled motion-kernel estimation (S1-84)** adds
`inspect.estimate_motion_blur` under Custom > Image processing > Restoration.
One degraded image provides disjoint complete selected tiles, split into fit
and validation groups. A bounded centered line-PSF grid is compared using native
DFT power, periodic Hann window effects, radial log-power whitening and a
positive nuisance floor. Fit observations alone select length, angle and floor;
validation scores that model without refitting or retrying. Contrast, fit/validation
score, distinct-hypothesis margin and search-boundary gates withhold PSF/length/
angle on rejection. Model floors are not measured SNR. Image remains unchanged;
kernel/source-coordinate metadata, current ticket/start time and pyramid units
are explicit. Masks/regions and straight alpha admit only fully supported tiles.
Spectrum diagnostics own their grid and drawable annotations use image pixels.
Accepted kernels wire directly to Wiener deblur; its connected PSF requirement
prevents rejected estimates from falling back to a configured default.
Eleven original MIT projects use an independent 257-point trapezoidal
pixel-space line integration and separate clean truth. Horizontal/diagonal/
vertical motion, mild noise, uint16 and selection pass; flat, sharp, heavily
noisy, ambiguous coarse-grid and mixed fit/validation textures reject. Accepted
controls meet one-pixel/five-degree parameter limits and restored MSE below
80% of degraded MSE. Circular boundaries are controlled, not inferred.
All **71 targeted checks** passed, including exact centered weights/moments,
color/depth/alpha and tiny/HDR units, changing validation observations without
changing the chosen fit, whole-tile selections, ticket/start/grid/pyramid
provenance, invalid source/frequency controls, resource guards, native failure/
recovery, graphics off, XML/offscreen controls, exact regeneration and catalog
persistence. Focused rechecks covered malformed images and a later rejected
acquisition not reusing an earlier accepted PSF, followed by normal recovery.
All **22 installed Debug/Lean outcomes** matched (12 passes, 10 expected
inspection failures). Three packaged modules match working bytes. Source hashes,
strict JSON, generated references/counts, local links, original text contracts
and repository hygiene passed; root README reviewed unchanged. App environment
refreshed; no broad suite repeated.
This completes the declared controlled uniform-motion scope. It is our own
algorithm built from OpenCV operations; examples and toolbox classification
reflect that. Anisotropic texture can mimic blur and disjoint patches can share
a wrong model. General arbitrary, defocus, spatially varying and rotational
kernels are not estimated, and neither scores nor synthetic MSE certify real
defect recovery or physical camera motion. See [contracts](blur-estimation.md)
and [eleven projects](<../examples/custom tools/motion_blur_estimation/README.md>).
At that checkpoint, core was **171/175 (97.7%)**, overall **234/470 (49.8%)**;
cycle **11/20**. The following capture negotiation increment closes S1-115;
curved-QR qualification (S1-121) is next. Native EXR (S1-125) and reusable LK pyramid
consumption (S1-155) remain dependent on capable native paths.

**Review-only stabilization border fills (S1-148)** adds
`sequence.fill_stabilization_borders` under OpenCV video analysis. It pulls
measured samples from bounded neighboring corrected frames onto the current
grid, using shared-root spatial mappings or an explicit registered-pixel-grid
assertion. Known different segments skip or reject. Required per-member coverage
and known ticket/grid checks prevent padded or mismatched masks from donating.
Nearest temporal priority and convex means retain depth/encoding; straight alpha
and full interpolation support are checked. Measured-overlap consistency rejects
donors with excessive changes; a local mismatch halo, optional mask/region and
distance limits restrict fill centers. Border-only and all-hole policies are
explicit. Current Image/Coverage remain measured and unexpanded; Display alone
contains review samples, with fill/missing/support maps, original donor indices,
current ticket/pyramid identity and donor provenance. Its own review output is
rejected as a new measured input. Availability judgments distinguish strict
completion from explicitly allowed partial fills.
Seven original MIT projects use independently constructed world pixels and
integer-jitter captures. Sealed known corrections, shifted corrected grids,
changed donors, uint16 means, inconsistent neighbors, distance limits and native
motion fitting are exercised. Corrected crops can have corners no donor still
contains: six remain missing in the shifted-grid control, and the native fixture
allows at most eight while requiring 98% fill. No corners are fabricated.
All **64 targeted checks** passed: analytic integer/subpixel and tiny/HDR/alpha
references, unchanged measured pixels/masks, masks/regions, donor gating,
metadata and resource validation, contained native exception/recovery,
graphics-off, XML/offscreen controls, exact regeneration and shared catalog
persistence are covered. All **14 installed Debug/Lean outcomes** matched
(10 passes, 4 intentional availability failures). Four packaged modules match
working bytes. Eleven original PNG hashes and six sealed trajectories, strict
JSON, generated references/counts, local links, original text contracts and
repository hygiene passed; root README reviewed unchanged. App environment
refreshed; no broad suite repeated.
The wheel has a videostab namespace, but the six audited native inpainter/flow
fill bindings are absent. This is a tested application equivalent, without
native inpainter parity or proof of current content in unseen pixels. See
[contracts](stabilization-fill.md) and
[seven projects](<../examples/opencv tools/stabilization_fill/README.md>).
At that checkpoint, core was **170/175 (97.1%)**, overall **233/470 (49.6%)**;
cycle **10/20**. Controlled motion-kernel estimation followed in S1-84 above.

**Explicit panorama refinement and reserved validation (S1-147)** extends
`cv.stitch_rotating_panorama` with native homography camera estimation,
none/ray/reprojection bundle adjustment, original-source initial intrinsics,
supported per-parameter refinement flags, termination controls and optional wave
correction. Automatic mode retains the existing high-level path. Explicit mode
partitions bounded SIFT observations by seeded rounded pixel center before native
matching; duplicate orientations/scales at one center stay together. Native
matching, estimation and adjustment receive only fitting observations. Reserved
observations alone establish validation overlap/model agreement afterward;
sparse validation never falls back to fit data or searches seeds automatically.
Reports distinguish the two modes and retain partition counts, fit pair evidence,
initial/fitted work intrinsics, refinement masks and reserved source-pixel errors.
Source-dependent priors reject before detection; fit descriptor work is bounded
before matching. Initial rotations are projected to proper rotations. Fixed K
survives rounded work grids and restores original source units. Ray controls
enforce its native centered/square, always-refined-focal limitations.
Nine original MIT projects reuse the known-camera assets without duplication:
native cost choices, fixed uint16/plane rendering, initial-only fitting, a refined
390-pixel focal estimate, wrong fixed intrinsics, disconnected views, horizontal
wave correction and insufficient reserved evidence. Six pass and three fail
intentionally. Updated earlier projects include current defaults and retain exact
original images/truth.
All **64 targeted checks** passed: native-call instrumentation proves reserved
centers are absent from matching/estimation/adjustment; known focal/yaw/world
landmarks, exact fixed K under work reduction, depth, validation rejection,
configuration/source preflight, descriptor budgets, normal native failure,
contained exception/recovery, XML/offscreen controls, original panorama contracts
and example regeneration are covered. All **18 installed Debug/Lean outcomes**
matched (12 passes, 6 intentional failures). Three packaged source modules match
working bytes. Generated references/counts, source hashes, strict JSON, local
links, original text contracts and repository hygiene passed; root README
reviewed unchanged. App environment refreshed; no broad suite repeated.
This completes the declared rotating-view camera workflow. Reserved keypoints
share source images and descriptor neighborhoods; they are not independent
captures. Free intrinsics can be underconstrained with yaw-only captures even
when residuals are small. The tools do not establish general physical camera
calibration or robot extrinsics. See [contracts](rotating-panorama.md) and
[nine projects](<../examples/opencv tools/panorama_refinement/README.md>).
At that checkpoint, core was **169/175 (96.6%)**, overall **232/470 (49.4%)**;
cycle **9/20**. Review-only stabilization border fills followed in S1-148 above.

**Unordered rotating-camera panoramas (S1-104)** adds
`cv.stitch_rotating_panorama` and `cv.panorama_points` under OpenCV stitching.
Native camera estimates are anchored independently of input order and checked
against separate SIFT/homography overlap evidence. Contradictory rotation models
fail; disconnected components have explicit reject, failed-partial and
allowed-partial policies. Spherical/cylindrical/plane rendering retains original
gray/RGB/BGR depth, supported samples and current batch identity. Masks/regions
restrict registration and output support. Camera intrinsics/rotations, source
tickets/grids, work scales, projection/origin and geometry identity are public
JSON data. The point mapper carries compatible cropped source measurements onto
the independent nonlinear panorama grid with optional source/target masks.
Drawable boundaries/markers aggregate on the image; connectivity charts have
their own grid and are omitted when graphics are disabled. Bounds precede mask
preparation and warp-map allocation; the work estimate excludes native workspaces.
Eight original MIT projects use known 480×360 cameras, a 420-pixel focal length,
±36-degree yaw and independent world landmarks. The first two inputs have no
direct overlap; the third bridges them. Cases cover reordering, rejected and
partial components, a distorted view, uint16 plane rendering and masks.
All **99 targeted checks** passed, including analytic ray pixels/projections,
native focal/rotation/landmark references, masks, source ownership, ticket/crop
and grid identity, resource guards, native failure/recovery, XML/offscreen
controls and existing scan/pair/catalog behavior. All **16 installed Debug/Lean
outcomes** matched: ten passes and six intentional inspection failures. Six
packaged modules match working bytes. Exact project/asset regeneration, strict
JSON, source hashes, generated references, local links and original text
contracts were checked; root README reviewed unchanged. The app environment is
refreshed. No broad suite repeated.
S1-147 is **Partial**: native high-level fitting and wave correction now ship,
while manual solver/refinement controls, intrinsic priors and truly reserved
validation observations remain. Current separate matching can reuse fit
observations. These are relative rotating-pinhole estimates, not physical lens
calibration, metric translation or robot extrinsics. See
[contracts](rotating-panorama.md) and
[eight projects](<../examples/opencv tools/rotating_panorama/README.md>).
At that checkpoint, core was **168/175 (96.0%)**, overall **231/470 (49.1%)**;
cycle **8/20**. Explicit refinement and reserved validation followed in S1-147 above.

**Actual video clock diagnostics (S1-123)** adds `sequence.video_timing` under
OpenCV Video I/O. The stream monitor preserves current image/ticket/coordinates
while checking raw/effective time consistency, missing/repeated/backward clocks,
index order, stride-normalized cadence, nominal-rate agreement and an optional
independent reference. Valid VFR is accepted by default; constant/nominal policies,
bounded row history, session counters, explicit loop Reset, latching/recovery and
graphics-off operation are configurable. Raw decoder positions now accompany
existing file, paced, encoded-group and network effective clocks without changing
their timing/ticket policies; the encoded-video helper retains its two-value API.
Six original Matroska/MJPEG controls encode independently specified block times,
including unequal/duplicate/backward/zero clocks, omitted DefaultDuration and a
wrong declared FPS. A separate parser checks the actual binary time/codec fields.
Eight annotated graphs cover those controls, a wrong reference and paced VFR
loop/reset, with current-part measurements and selected timing/blob overlays.
All **207 targeted checks** passed; the final **43 focused monitor rechecks**
also passed after enforcing matching Info tickets and raw/effective decoder
consistency. Native controls, independent parser/exact regeneration, pacing,
start/stride/close-resume, bounded time sampling, encoded groups, history eviction,
failure recovery/reset, extreme metadata, copies, chart ownership, XML/editor and
existing video/network/catalog behavior are covered. All **16 installed Debug/Lean
outcomes** matched **128 activations**, including **62 intentional latched failed
judgments** with their reports, images and reasons retained.
Seven packaged modules match exact source bytes. Six complete MKV hashes,
independent binary parsing/exact regeneration, nine strict JSON files, 609 local
links, generated controls/counts and fifteen original text contracts passed;
root README reviewed and unchanged. See
[contracts](video-timing.md) and
[eight projects](<../examples/opencv tools/video_timing/README.md>).
These checks qualify authored MJPEG containers on this native FFmpeg build;
monotonic decoder reports are not authenticated PTS/exposure times, frame-drop
proof or physical synchronization. The independently published SimpleFringeScan
retained reviews are linked in the example index; their owner's native evidence
was not rerun or added to host vision coverage here.
At that checkpoint Core was **164/175 (93.7%)**, overall **227/470 (48.3%)**;
cycle **4/20**.
No broad suite repeated. The next increment below closes whole selections through
nonlinear lens correction (S1-69). Native EXR (S1-125) and cached LK-pyramid consumption (S1-155)
remain open.

**Moving-region rejection in HDR brackets (S1-150)** adds `cv.merge_hdr_motion`.
It computes response-aware weighted linear radiance from aligned timed uint8
brackets, rejecting inconsistent channels and insufficient contributors. Optional
Debevec/Robertson response curves, per-member masks, reference regions, clipping,
consistency/coverage thresholds, weighting, display fallback and bounded work use
normal block controls. Default clipped-code bounds can detect a contradictory
saturated moving part without using its clipped value as measured radiance.
An explicit Ignore policy demonstrates the lost evidence. Image is zero outside
Valid mask; separate reference-filled Display and Fallback mask stay review-only.
Native Align exposures now declares original/target geometry and valid support,
so zero padding does not become photometric evidence. Labels, member/aggregate
tickets, reference grids and source provenance are retained.
Six original MIT timed projects independently specify exact static radiance,
three moving-part positions including a saturated long exposure, clipped missing
support, strict failed coverage and native integer-shift recovery. The masked
statistics use measured pixels; filmstrip overlays show current diagnostic fills.
All **76 targeted checks** passed, including independent scalar weighted/channel
and nonlinear-response references, exact masks/counts/means, crop/padded native
MTB support, explicit grid assertions, source masks/regions, stale metadata,
native exception/recovery, work/parameter limits, XML/offscreen controls and
existing HDR/exposure/catalog behavior. A final **9 focused metadata/catalog
rechecks** passed after documenting the alignment-support output in the API.
All **12 installed Debug/Lean outcomes** matched: ten passes and two intentional
strict-coverage failures, with current filmstrip diagnostics retained. Five
packaged source modules match exact working bytes. Exact regeneration, eighteen
PNG hashes, seven strict JSON files, generated references/counts, 551 local links,
twelve original text contracts and repository hygiene passed; root README
reviewed unchanged. App environment refreshed. No broad suite repeated. The independently
published SimpleDeflectometry file-only reviews are indexed without rerunning the
owner's standalone estimator or adding host coverage for that separate work.
This is a photometric consistency proxy, not recovered hidden/clipped information,
physical synchronization or absolute radiometry. See [contracts](hdr-motion.md)
and [six projects](<../examples/opencv tools/hdr_motion/README.md>).
At that checkpoint, core was **167/175 (95.4%)**, overall **230/470 (48.9%)**;
cycle **7/20**. Unordered rotating views followed in S1-104 above.

**Visible support before pick selection (S1-61)** adds `inspect.visible_support`.
It samples candidate appearances on the selected template grid, retains detector
scores and reports matching/observed/in-frame counts before filtering candidates.
Missing or masked scene pixels remain in the expected denominator. Pattern
transforms, fixed bounds/subpixel locations and centered/custom-origin poses map
through known crop/resize frames; a separately loaded training grid needs either
matching known frames or an explicit pixel-alignment assertion. Current ticket
checks, raw-depth RGB/BGR comparison, alpha exclusion, bounded work, conservative
border support and optional drawable outlines/text use the common block contracts.
Four original MIT examples run native template detection, support qualification,
clearance/downstream pick selection and filmstrip overlays. Clear and touching
parts each retain 616 expected samples; the specified 210-sample cover leaves
406/616 and excludes the downstream part. Touching neighbors remain fully visible
but fail clearance. A required-three gate fails normally, keeps diagnostics and
suppresses picking through the configured failed-input policies.
All **92 targeted checks** passed, including independent sample/pick truth,
analytic projective sampling, 90-degree rotations/custom origins, compatible crop
frames, template/source selections, raw 16-bit/float64 channels, metadata/tickets,
malformed/native errors, recovery, work limits, graphics-off, XML and offscreen
controls plus existing pattern, template, pick and catalog behavior. A final
**34 focused rechecks** passed after avoiding unnecessary alpha/channel copies.
All **8 installed Debug/Lean outcomes** matched: six passes and two intentional
failed count gates, with current filmstrip diagnostics and suppressed picks.
Four packaged source modules match working bytes. Exact regeneration, five PNG
hashes, five strict JSON files, generated references/counts, 545 local links,
eleven original text contracts and repository hygiene passed; root README
reviewed unchanged. App environment refreshed.
No broad suite repeated. This is an appearance-support proxy: it does not recover
true hidden area or certify a robot grasp. Linear float64 comparison explicitly
uses a 1/32-phase grid for both appearance and support; output image pixels remain
unchanged. See [contracts](visible-support.md) and
[four projects](<../examples/opencv tools/visible_support/README.md>).
At that checkpoint Core was **166/175 (94.9%)**, overall **229/470 (48.7%)**;
cycle **6/20**. HDR moving-region rejection is qualified above. Native EXR
(S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Whole selections through nonlinear correction (S1-69)** adds
`cv.remap_selection` and its packed-map equivalent. Image pixels, masks and regions
share one remap; exact scanline runs preserve selection holes. Independent validity
excludes filled borders, with complete-footprint or thresholded coverage policies.
Tickets, color/depth encoding and corrected spatial grids are retained, and bounded
raster regions work with existing composition, transforms and measurements.
Five original RGB16 graphs cover nonlinear lens correction, fractional coverage,
packed maps, a wider field with replicated borders and rejected mixed map pairs.
Analytic Brown5 coordinates and independent four-tap references qualify exact mask
hashes, areas, centroids and channel means in Debug and Lean. The examples expose
OpenCV 5 float-map versus packed 1/32-phase precision rather than assuming equality.
Cubic/Lanczos support is deliberately conservative; counts remain target pixels,
not calibrated physical area. No physical depth reconstruction is claimed.
All **156 targeted checks** passed, including native sampling, nested raster
budgets, parameter/exception guards, color/coordinate/ticket ownership, existing
region consumers, XML and offscreen inspector controls. All **10 installed
Debug/Lean outcomes** matched, including two intentional contained map-pair errors.
Seven packaged source modules match working bytes; exact fixture regeneration,
seven strict JSON files, generated references/counts, 545 local links and thirteen
original text contracts passed. The root README was reviewed and remains unchanged.
The SimpleFringeScan index now links its owner's retained setup and execution
projects without rerunning native plugin acceptance or adding host coverage for
that separate work.
See [contracts](selection-remap.md) and
[five projects](<../examples/opencv tools/selection_remap/README.md>).
At that checkpoint Core was **165/175 (94.3%)**, overall **228/470 (48.5%)**;
cycle **5/20**. No broad suite repeated. Visible-support scoring is qualified above. Native EXR (S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Video sampling by media time (S1-110)** adds `input.video_at_time` under OpenCV
Video I/O. A configured or connected millisecond request selects nearest,
at-or-before or at-or-after frames. Native `CAP_PROP_POS_MSEC` positioning is
checked against the actual decoded time, with bounded forward brackets or
explicit start-scan fallback when positioning overshoots/rejects or lacks time.
Preroll, strict/estimated timing, signed error and acceptance limits, native/ordinal
index provenance and read/pixel/working budgets are exposed. Each independent
sample receives a fresh ticket and pixel grid after successful conversion; no
playback cursor or clock is silently advanced. Missing/changed files, invalid
settings, cancellation and native errors close the decoder and emit no stale frame.
Five original MIT projects cover exact/nearest/earlier/later selection and a
deliberate excessive-time-error rejection with blob measurements and filmstrip
overlays. All **115 targeted checks** passed, including exact lossless pixels and
annotated times, independent full-decode parity on a verified interframe MPEG-4
clip, nonsequential requests, missing timing/index properties, explicit estimates,
read/size guards, cleanup, request/external tickets and output-frame ownership,
explicit native-probe estimate labels, XML/editor/Save As and related
file-video/runner/catalog controls. The final compressed-fixture spot check also
passed after asserting that P/B frames are present. All **ten installed Debug/Lean
outcomes** matched **20 activations**, including four intended rejections with no
source image/ticket or downstream filmstrip output. Four packaged modules match
exact source bytes. Original clip hash, exact
regeneration, six strict JSON files, 597 local links, generated controls/counts and
thirteen original text contracts passed; root README reviewed and unchanged.
See [contracts](video-time-seeking.md) and
[five projects](<../examples/opencv tools/video_time/README.md>).
Native qualification is for constant-rate FFV1/MPEG-4 through this FFmpeg build;
position properties are not an independent demuxed-PTS guarantee. Native local
calls remain cooperatively cancellable and memory estimates exclude decoder/RSS.
Core is **163/175 (93.1%)**, overall **226/470 (48.1%)**; cycle **3/20**.
No broad suite repeated. Next qualify actual variable-frame-rate/unreliable-timing
clips (S1-123). Native EXR (S1-125) and cached LK-pyramid consumption (S1-155)
remain open.

**Network video input (S1-12)** adds `input.network_video` under OpenCV Video I/O.
An owned FFmpeg worker and one bounded shared frame slot keep native open/read
calls outside the graph worker, with separately enforced startup/open/read response
deadlines and cancellable reconnect delays. Stop ends its private worker rather
than waiting for a long backend timeout. Only successfully decoded/converted
current frames receive tickets. Per-connection spatial roots and an optional
Stream start signal make sequence resets explicit; neither stale frames nor
invented exposure times are emitted after recovery. The original managed-service
MJPEG projects verify normal reception and deliberate four-frame disconnection,
known blob centers/areas, tracking resets and final filmstrip overlays.
All **116 targeted checks** passed, including native loopback reads, bounded
open/read stalls, unexpected worker exit, shared-slot/process cleanup, retry
exhaustion, cancellable delay, offscreen Runner Stop/Resume, external tickets,
runtime copies, XML/editor validation, exact asset/project regeneration and
existing file-video/lifecycle regressions. Four installed Debug/Lean outcomes
matched **32 activations**, and the owned server port closed after each run.
Six packaged modules match exact source bytes. Eight original JPEG hashes, exact
regeneration, three strict JSON files, 599 local links, generated controls/counts
and thirteen original text contracts passed; root README reviewed and unchanged.
See [contracts](network-video.md) and
[two projects](<../examples/opencv tools/network_video/README.md>).
Qualification covers Windows loopback HTTP MJPEG; exposed RTSP TCP/UDP and HTTPS
controls still need their actual server/build qualification. Response deadlines
are not hard real time, native buffers are not a process RSS bound, and physical
camera FIFO/latency/losslessness is not established.
Core is **162/175 (92.6%)**, overall **225/470 (47.9%)**; cycle **2/20**.
No broad suite repeated. Next qualify media-time video seeking (S1-110), while
actual VFR diagnostics (S1-123), native EXR (S1-125) and cached LK-pyramid
consumption (S1-155) remain open.

**Timestamp-paced file replay (S1-11)** adds `input.paced_video_file` alongside
the existing fast source. A monotonic cancellable media clock preserves original
selected-frame timing, stride gaps, index/pixels and ticket provenance while
exposing Speed, decoder/FPS policy and explicit lateness/gap handling. No implicit
frame dropping occurs. Stop during a clock wait emits no ticket/image event and
keeps the pending frame; Resume reanchors its wall schedule. Cycle start marks
loop boundaries for explicit sequence Reset connections. Sixty focused checks
cover independent fake-clock variable gaps/overhead/speed/fallback/lateness,
native FFmpeg/MJPEG, queue order, offscreen Runner Stop/Resume, runtime copies,
parameter/XML/editor guards and three original lossless replay graphs. Six
installed Debug/Lean runs matched all 38 activations; five packaged modules match
source bytes. The original clip hash, exact regeneration, four strict JSON files,
581 local links, generated controls/counts and original text contracts passed.
See
[contracts](paced-video.md) and
[three projects](<../examples/opencv tools/paced_video/README.md>).
This is cooperative local-file scheduling, without arbitrary native-call
interruption, actual VFR qualification or hard-real-time claims.
Core is **161/175 (92.0%)**, overall **224/470 (47.7%)**; new cycle 1/20.
Next qualify bounded network video replay/reconnection (S1-12). Native EXR
(S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Common-pitch measured profiles (S1-173)** adds `cv.resample_image_profile`.
Exact own/explicit/reference grids, optional current rectified-workplane scale,
Float64 rate-aware Kaiser sinc/FIR filtering and linear interpolation preserve
source image coordinates, current tickets and channel-specific invalid support.
Null gaps are not bridged; boundary extension is flagged, endpoints never stretch
the requested pitch, and closed paths wrap one exact period. Three original
16-bit projects compare pixel and physical grids plus masked decimation.
Thirty-nine focused checks verify analytic amplitude/peak/alias behavior,
different sampling densities, current/reference provenance, physical scale,
boundaries/gaps, work/shape/value guards, XML and offscreen controls. All six
installed Debug/Lean project outcomes matched. Two original asset hashes,
exact regeneration, five strict JSON files, 583 local links, generated controls
and original text contracts passed.
The installed native signal resampler retained 0.6298 RMS in a 4:1 alias test and
uses a different distance convention, so the supported block is an original
tested equivalent under Custom tools; native binding coverage is not claimed.
See [contracts/native audit](profile-resampling.md) and
[three projects](<../examples/custom tools/profile_resampling/README.md>).
Core is **160/175 (91.4%)**, overall **223/470 (47.4%)**; cycle 20/20,
with the clean broader checkpoint recorded above. Next qualify timestamp-paced file video
replay (S1-11). Native EXR (S1-125) and cached LK-pyramid consumption (S1-155)
remain open.

**Periodic detector refresh (S1-172)** adds `flow.detection_cadence` and
`sequence.track_refreshed`. Any detector can receive selected cadence images;
its current results correct/reseed existing tracks and discover entering objects.
Between refreshes, bounded ROI corners, forward/backward native LK and robust
similarity fits refine only their original identities. Lost evidence predicts
then expires without learning replacement background; independent detector-age
frame/time limits expire even successful local tracks. Explicit detector/local/
prediction flags and current tickets/grids remain on results and poses. Three
annotated projects compare four versus sixteen independently counted detector
executions, entry/loss and deliberately short freshness. Native translation and
rotation/scale references, controlled injected drift correction, no local births,
selection/ticket/depth/work guards, contained native failures, forced cadence,
worker/XML and offscreen controls cover the surrounding contracts. All **104
targeted checks** passed. Six installed Debug/Lean runs matched all **96
activations**; sixteen image hashes, exact regeneration, four strict JSON files,
572 local links, generated controls/counts and original text contracts passed.
See [contracts](refresh-tracking.md) and
[three projects](<../examples/opencv tools/detector_refresh/README.md>).
Core is **159/175 (90.9%)**, overall **222/470 (47.2%)**; cycle 19/20.
Next qualify common-pitch image-profile resampling (S1-173). Native EXR
(S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Image-track prediction (S1-54/S1-171)** adds `sequence.track_image` with
native Float64 Kalman prediction/correction, Joseph covariance and elapsed-time
center/log-size/angle state. Global gated assignment reuses the conveyor solver;
center-only crossings and reordered detections retain identity. Wrapped 180/360
angles, partial measurements, label/scale/angle gates, ambiguity flags, heuristic
confidence and explicit missed/time/uncertainty/offscreen expiry keep estimates
separate from observations. Pattern model scale and offset outlines, current
image tickets and coordinate frames survive prediction. Missing events hold;
explicit empty detections advance. Native/input errors retire memory before
recovery. Two annotated twenty-frame projects use actual native blob detections
and independent intended motion. Seventy-four targeted checks cover analytic
transition/Joseph covariance, irregular time, wrapping, crossing, reacquisition,
expiry, metadata, memory workers/XML, failure containment, graphics and offscreen
controls. Four installed Debug/Lean runs matched all 80 activations; six packaged
modules match source. Twenty asset hashes, exact regeneration, strict JSON,
566 local links, generated controls/counts and original text contracts passed.
See [contracts](image-tracking.md) and
[two examples](<../examples/opencv tools/image_tracking/README.md>).
Synthetic accuracy and identity checks qualify this constant-velocity baseline,
not arbitrary occlusion or production tracking performance.
Core is **158/175 (90.3%)**, overall **221/470 (47.0%)**; cycle 18/20.
Next qualify periodic detector refresh and local-tracker handoff (S1-172).
Native EXR (S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Saved foreground references (S1-170)** adds clean-reference capture/publication
and sealed file restoration to `sequence.mog2` and `sequence.knn_background`.
Declared camera/calibration IDs, size/depth/encoding, numerical grid/pyramid and
explicit recipe binding gate native allocation. Bounded replay must recover the
clean background exactly before the first current image can emit; current tickets
and geometry survive, without reviving saved acquisition identities. Frozen
MOG2 retains its model. This wheel's KNN can update at zero learning rate, so its
frozen workflow replays a disposable query model per activation. Guarded updates
require a foreground mask and replace marked samples with trusted background
before native learning, preventing the observed KNN mask-update leakage. The
workaround cost and native scope are explicit. Fifty new checks and **102 targeted
checks** including existing native-background, sequence and reference regressions
passed. All **16 installed Debug/Lean workflow outcomes** matched across **24
activations**, including deliberate mismatch rejection and reference publication
in separate writable copies. The app was reinstalled; four packaged modules match
working bytes. Thirteen original asset hashes, eleven strict JSON files, exact
regeneration, generated controls/counts, local links and text contracts were checked.
Eight original MIT projects
cover first-image MOG2/KNN, protected updates,
five-image replay, Float32 TIFF, deliberate camera mismatch and reference capture.
Independent/native masks, persistent parts, fresh roots/tickets, context/format
rejection, corruption/pins, cache/reset, readiness/work guards, native failure
recovery, atomic publication, XML/Save As/reopen and offscreen controls qualify
this clean-image reconstruction baseline. Native model/mixture history and
physical camera identity/cleanliness are not established by a saved reference.
See [contracts](background-reference.md) and
[eight projects](<../examples/opencv tools/background_reference/README.md>).
Core is **156/175 (89.1%)**, overall **219/470 (46.6%)**; cycle 17/20.
Next qualify periodic detector refresh and local-tracker handoff (S1-172).
Native EXR (S1-125) and cached LK-pyramid consumption (S1-155) remain open.

**Reference-based rolling-shutter correction (S1-169)** adds
`inspect.scanline_motion` feeding the existing `cv.warp_flow` sampler. Native
Shi-Tomasi/LK correspondences feed robust degree 1-5 horizontal/vertical row
polynomials; separate held-out tracks, row span/gap limits, explicit extrapolation
and displacement/row-order bounds gate every field. Failed fits withhold Flow.
Content/direction-bound image pairs prevent stale sampling; current pixels, depth,
encoding, tickets and video time survive on the reference grid, with explicit
coverage and nonlinear output-to-source maps. Six original MIT projects cover
known cubic row motion, six-frame FFV1 replay, 16-bit BGR, a current moving part
fitted from a static strip, and deliberate no-texture/row-gap failures.
**57 new checks** and **110 targeted checks** including dense-warp, stabilization
and reference regressions passed. All **12 installed Debug/Lean workflow outcomes**
(**22 activations**) matched, including deliberate failures. The app was reinstalled;
four packaged source modules matched exact working bytes; eight original PNG/video
hashes, exact regeneration/readback, generated controls/counts, strict JSON, local
links and text contracts were checked. Independent truth verifies near-edge geometry,
substantial error reduction, current-part preservation and exact video/timing
transport within timestamp tolerance. Range/shape/encoding/finite/resource errors,
robust/held-out rejection, all polynomial degrees, unsafe/out-of-model geometry,
paired-image corruption, concurrency/reset, XML/reopen and offscreen controls are
covered. See [contracts](rolling-shutter.md) and
[six projects](<../examples/opencv tools/rolling_shutter/README.md>).
This is a reference-based row-translation baseline; it does not claim general
projective correction, physical readout calibration or production qualification.
Core is **155/175 (88.6%)**, overall **218/470 (46.4%)**; cycle 16/20.

**Restore detail from aligned neighbors (S1-168)** adds
`sequence.restore_neighbors` for finite registered captures, explicit horizontal/
vertical box or Gaussian target blur, local sharpness qualification and a
channel-wise blur-prediction/change-halo gate. Target coverage never expands;
selection fallbacks and detected foreground changes retain exact source samples.
Best or weighted fusion reports original donor indices, actual contribution
masks and capture tickets while preserving the current target ticket, pixel
depth, channel order, alpha and coordinate/pyramid payload. Controlled moved,
appearing and disappearing parts verify that detected changes do not duplicate
or erase parts. Six original MIT projects cover alternating sharp/blurred/sharp
captures, Gaussian blur, 16-bit BGR with coverage/region selection and explicit
no-donor pass-through/failure. **69 focused checks** qualify independent sharp
truth and error reduction, supported neighborhoods, extreme floating values,
current-ticket identity, concurrency, XML/reopen, offscreen dropdowns and recovery.
All **110 targeted checks**, including related focus-fusion/reference regressions,
passed; **12 installed Debug/Lean outcomes** matched, including intentional failure.
The app was reinstalled; four packaged source modules, eight original image hashes,
strict JSON, deterministic regeneration, local links and text contracts were checked.
The native video deblurrer is unavailable in this Python wheel; this application
equivalent requires known registration/blur and cannot certify low-contrast or
model-indistinguishable changes. See [contracts](neighbor-restoration.md) and
[six projects](<../examples/opencv tools/neighbor_restoration/README.md>).
Core is **154/175 (88.0%)**, overall **217/470 (46.2%)**; cycle 15/20.

**Saved image-motion trajectories (S1-167)** adds Capture, Save, Load and Replay
tools for versioned source-to-target affine corrections. Ordered pixel/type/encoding
fingerprints, source coordinate/pyramid geometry and media indices/timestamps bind
the complete current clip before warping. Nominal indexed timing is an explicit
alternative. Validity/reasons and segment target grids persist; current tickets
and custom metadata survive without reviving capture identities. Replay has no
cursor or fit cache. Reject, failed-partial and explicit-skip policies handle
invalid/low-coverage frames; all-invalid clips cannot pass. Strict bounded JSON,
canonical clip/revision IDs, optional file SHA256, exclusive/atomic publication
and Save As assets qualify persistence. The native videostab motion-file classes
are absent from this Python wheel; this is a validated application record format.
Ten original MIT projects use known integer shifts in PNG/FFV1, uint16, actual
media timing, explicit invalidity, native fit/capture parity and deliberate
pixel/order/time rejection. **67 focused checks** include independent pixels and
coverage, changed acquisition roots/current tickets, timestamps, segment geometry,
single-frame clips, inverse/validity guards, all-invalid/coverage policies, native
failure recovery, safe file publication, XML/reopen and offscreen controls. All
**20 installed Debug/Lean outcomes** matched. App reinstall, six packaged source
modules, sixteen original image/video/trajectory hashes, deterministic
regeneration, strict JSON, generated references/counts, local links and text
contracts were verified. Core is **153/175 (87.4%)**, overall **216/470 (46.0%)**;
cycle 14/20, no broad suite repeated. Records are bounded to 1–64 frames and affine
corrections; unbounded append recording and native videostab file interoperability
are not claimed. Content/timing association does not prove physical camera identity
or stabilization accuracy. See [contracts](motion-trajectory.md) and
[ten projects](<../examples/opencv tools/motion_trajectory/README.md>).

**Relative image frequency response (S1-166)** adds `inspect.frequency_response`
for aligned sample/reference captures. Native float64 DFT/division produces
gain/phase maps only where the reference excites the selected frequency band/cone.
Shared masks/regions, optional weighted centering and None/Hann/square-root Hann
windows define the aperture. Coherent amplitude floors and a pre-centering
numerical floor suppress weak/constant references; unit-normalized denominators
avoid native epsilon bias. Sample-energy gating distinguishes zero gain from
undefined phase. Radial means/circular phase, inclusive gain/phase judgments and
explicit no-support failures are exposed. Original sample depth/encoding is
preserved; diagnostic maps have a separate frequency grid with both source
contexts/tickets retained. Raw/unsigned-unit scaling and semantic color channels
are configurable. Ten original MIT projects use exact uint16 periodic binomial
blur, known shifts/brightness, a sparse grating, masked exterior and deliberate
support/gain/phase/grid/budget failures. **73 focused checks** cover independent
NumPy/analytic references, odd/even grids, gray/color/alpha/high depth, 1e±100
signals, selections/windows, numeric floors, payloads, guards, native exceptions,
parallel instances, XML/reopen and offscreen controls. All **20 installed
Debug/Lean outcomes** matched. App reinstall, four packaged source modules,
original asset hashes/regeneration, strict JSON, generated references/counts,
local links and text contracts were verified. At that completion core was **152/175 (86.9%)**,
overall **215/470 (45.7%)**; cycle 13/20, no broad suite repeated. This direct
single-pair measurement does not prove physical alignment, optical MTF
certification or statistical coherence. Windows/selections can bias the ratio.
See [contracts](frequency-response.md) and
[ten projects](<../examples/opencv tools/frequency_response/README.md>).
Saved motion trajectories follow in S1-167 above.

**Registered-view replay (S1-165)** adds `cv.replay_registered_views`: independent
images and matching coverage groups on full, union or common-valid cropped grids.
Per-member tickets/timestamps, custom data, source/pyramid provenance and inverse
coordinate maps survive; Unbatch enables serial downstream inspections without
blending images. Stable-ID/legacy layouts share the existing missing-view and
strict-grid rules. Exact support handles invalid borders and mask holes; no
common support is a failed status with no frame events. The default preserves
uint8/uint16/float32/float64 gray/RGB/BGR(A). Optional native detail placement uses
the AS_IS/CROP factories with exact uint8 gray/color conversion and explicit mask
enforcement. A direct native crop-constructor fault was isolated; the working
factory is used exclusively. Dense-workspace guards precede warping/allocation.
Nine original MIT projects cover union/common crops, native masks, missing/reordered
inputs, uint16, per-ticket statistics and deliberate no-support/depth/budget
failures. **115 focused checks** include independent pixels, half-pixel averages,
rotation, support/coordinate/ticket/pyramid contracts, native reset, independent
parallel instances, exception recovery, XML/reopen and offscreen dropdowns. All
**18 installed Debug/Lean outcomes** matched. App reinstall, four packaged source
modules, nine original image/layout hashes, deterministic regeneration, strict
JSON, generated references/counts, local links and text contracts were verified.
The binding inventory also refreshes accumulated existing QR references; these
are reference observations and add no capability completions. At that completion core was
**151/175 (86.3%)**, overall **214/470 (45.5%)**; no broad suite repeated.
Replay does not establish physical camera identity or synchronized exposures.
See [contracts](registered-view-replay.md) and
[nine projects](<../examples/opencv tools/registered_view_replay/README.md>).
Relative frequency response follows in S1-166 above.

**Stable calibrated mosaic reuse (S1-164)** extends `cv.register_scan_layout`
and `cv.render_mosaic_tile` with opt-in source IDs from labels, per-member payload
metadata or an explicit list/input. Version-2 layouts seal calibrated geometry;
reordered and missing acquisitions retain calibrated blend priority and a fixed
full-canvas root, including when the anchor camera is absent. Strict grid checks
reject changed dimensions/depth/encoding and declared crop/frame changes. An
explicit pixel-index assertion is available and reported. Missing views can
reject output, emit failed partial data, or allow an explicitly incomplete tile.
Coverage/overlap, current contributor tickets and source/tile transforms remain
available. Legacy ordered projects retain their original version-1 behavior.
Load/Save mosaic layout adds bounded strict JSON, optional exact-byte SHA256,
no-overwrite/atomic replacement and project-relative Save As asset copying.
Nine original MIT projects cover controlled fixed geometry, reordering, missing
middle/anchor/single views, explicit rejection, masked multiband tiles and separate
native registration. The **122 focused checks** include exact independent pixel
references, coverage/priority, feather/multiband tile seams, stable coordinates,
native fitting versus reuse, huge virtual bounds, persistence, exceptions and
offscreen controls. All **18 installed Debug/Lean outcomes** matched. App reinstall,
six packaged source modules, original image/layout hashes, deterministic
regeneration, strict JSON, references/counts, local links and text contracts were
verified. At that completion core was **150/175 (85.7%)**, overall **213/470 (45.3%)**; no broad suite
repeated. IDs and declared grids do not prove physical camera identity or
synchronized exposures; this is planar transform reuse. See
[contracts](mosaic-layout-reuse.md) and
[nine projects](<../examples/opencv tools/mosaic_layout_reuse/README.md>).
Registered-view replay follows in S1-165 above.

**Feature repeatability and match accuracy (S1-163)** adds
`inspect.evaluate_features` for existing ORB/BRISK/AKAZE/SIFT/KAZE feature sets.
The legacy evaluation functions are absent from the Python wheel; the replacement
explicitly measures center reprojection under a caller-supplied independent image
transform, with bidirectional common support and optional selections. Maximum
one-to-one geometric pairing defines repeatability; unique correct accepted pairs,
duplicate penalties and grouped thresholds define precision/recall. Native
distance/ratio matching or existing matcher records feed the benchmark. Undefined
metrics emit no scalar; no-support cannot report a passing evaluation.
Eight original MIT projects use independently rendered identity/rotation/projective
and occluded image pairs, existing radius matching and deliberate wrong-transform,
empty-feature and pair-budget failures. Exhaustive 3×3 graph oracles and independent
no-augmenting-path certificates verify maximum assignments and threshold behavior.
The **59 focused checks** include supported native descriptor families, source
grids/tickets, budgets, error recovery, parallel instances, XML/editor controls,
graphics-off and odd-width/half-pixel preview mappings. All **16 installed
Debug/Lean outcomes** matched. App reinstall, five packaged source modules,
original image hashes/regeneration, strict JSON, generated references/counts,
local links and text contracts were verified. At that completion core was
**149/175 (85.1%)**, overall **212/470 (45.1%)**; no broad suite repeated. Legacy ellipse-overlap scores
are a separate metric; caller-supplied transform independence cannot be proven by
the block. See [contracts](feature-evaluation.md) and
[eight projects](<../examples/opencv tools/feature_evaluation/README.md>).
Stable source identity and missing-view reuse follow in S1-164 above.

**Structured-append QR recovery (S1-162)** adds `inspect.read_qr_append`
for a single scene or an atomic image group, with native standard/ArUco detection
and per-symbol byte decoding. Actual QR module headers, exact format copies and
deinterleaved RS codeword integrity qualify sequence indices/count/parity without
trusting generator metadata or visual order. Unique complete indices, payload XOR,
optional expected SHA256 and strict ASCII/UTF-8/hex conversion gate full outputs;
missing/duplicate/mismatched/unverified sets emit status and no partial payload.
The header verifier conservatively rejects damaged raw codewords even when native
OpenCV repairs the payload. QR parity alone cannot distinguish all mixed sets;
the collision fixture demonstrates why a known expected hash is useful.
Eight original MIT projects cover shuffled scene/batch success and six deliberate
failures. Native references cover 108 grids across nine sampled versions spanning
1–40 and all correction levels; 2/16-part limits and split UTF-8 are checked.
**147 focused checks** and **16 installed Debug/Lean outcomes** matched.
Per-member tickets/grids, source-space graphics, selection, global candidate and
pixel/byte guards, native exception recovery, parallel instances, XML/reopen and
inspector dropdowns are qualified. App reinstall, six packaged source modules,
original assets/regeneration, strict JSON, links and text contracts were checked.
Two factual block-layout tables retain Project Nayuki's full MIT notice and pinned
source; no new decoder dependency was introduced. At this completion core was
**148/175 (84.6%)**, overall **211/470 (44.9%)**; no broad suite repeated. See
[contracts](qr-structured-append.md) and
[eight projects](<../examples/opencv tools/qr_structured_append/README.md>).
**ArUco-backed QR reading (S1-161)** extends `inspect.read_qr` with a
standard/ArUco backend dropdown, preserving the existing default and output ports.
All seven QR finder controls and 35 native ArUco detector fields have explicit
defaults, tooltips, numeric validation and a corner-refinement dropdown. Single
and multi modes work on both backends; curved remains standard-only. Per-call
native instances support previews and independent parallel branches. Pixel/window
guards, explicit uint8 working depth, mask/region semantics, error containment,
camera/module grids and ticket retention are documented.
Six original MIT scenes compare clutter, rotation, two-pixel modules, three labels,
a selected label and no-code behavior. Known rendering transforms bound corner
errors below 2.5 pixels; single rectified patches exactly match original modules.
The standard backend's small-label miss remains an explicit inspection failure,
while ArUco decodes it on this build; this does not rank arbitrary-image robustness.
**116 focused checks** and **12 installed Debug/Lean outcomes** matched, including
that deliberately failed comparison branch. App reinstall, three packaged source
modules, deterministic regeneration, original asset hashes, strict JSON, local
links, generated counts/parameters and text contracts were checked. At this
completion core was **147/175 (84.0%)**, overall **210/470 (44.7%)**; no broad suite repeated.
See [contracts](qr-backends.md) and
[six projects](<../examples/opencv tools/qr_backends/README.md>).
**Temporal pixel statistics (S1-160)** adds `sequence.pixel_moments` with selected
gray/RGB(A) channels or an optional paired image. Centered float64 cumulative and
exponentially weighted moments produce mean, variance, standard deviation,
covariance, correlation, count and effective-support maps. Sample/population
normalization, per-pixel mask/region histories, freeze/sample caps and explicit
variance/correlation validity avoid treating excluded pixels as observations.
Ticket/grid policies separate metadata consistency and asserted pixel registration
from physical synchronization. Memory/workspace guards, context/reset/gap/stream
rules, private worker copies, native exception containment and optional heatmaps
reuse the shared runtime. Seven original MIT replays verify sample deviations of
2/4 units, channel/source covariance, exponential weights, ROI support and
deliberate ticket/memory failures. Independent batch references and a Decimal
oracle qualify small variations around 1e12; ±1e100 samples remain finite.
**94 focused checks** and **14 installed Debug/Lean replay outcomes** passed.
App reinstall, four packaged source modules, generated references/counts,
strict JSON, original asset hashes/regeneration, links and text contracts were
verified; no broad suite was repeated. At this completion core was **146/175
(83.4%)**, overall **209/470 (44.5%)**. Counts/weights do not prove independent observations; scene
movement or lighting drift remains part of measured variation. See
[contracts](temporal-pixel-statistics.md) and
[seven projects](<../examples/opencv tools/temporal_pixel_statistics/README.md>).
**On-demand TIFF page access (S1-158)** adds `input.tiff_page` with a wired or
configured zero-based index, native one-page range decoding and an LRU bounded
by raw page count and bytes. An immutable snapshot and inspected directories
support the existing lossless gray/RGB(A), integer/HDR float TIFF subset, up to
1024 pages and 128 MiB encoded bytes. Refresh policies distinguish file-observation
checks, per-request content verification and explicit pin/reset. Directory-based
pixel/workspace guards run before native pixel decoding. Current acquisition
tickets remain fresh on hits; page grids bind content hash and index. Output
conversion cannot mutate raw cache entries. Copyable bounded state survives editor
worker handoffs, while JSON/XML reopening starts empty. A zero-step Counter input
now supplies a reusable fixed integer selector without resetting the TIFF block.
Five original MIT projects use a 257-page stack with exact uint8/uint16/float32/
float64 reference hashes and deliberate range/workspace failures. The fixed
11-request order makes nine native one-page calls and two hits, retaining at most
196,608 raw bytes versus 13,393,920 raw bytes for all pages. These are residency
and call-count observations, not an application speedup claim. **151 targeted
checks** and **10 installed Debug/Lean outcomes** passed, including existing TIFF,
typed buffers, coordination and reference behavior. App reinstall, exact wheel
source bytes, generated references/counts, strict JSON, regeneration, staged asset
hash, links and original text contracts were verified; no broad suite was repeated.
At that completion core was **145/175 (82.9%)**, overall **208/470 (44.3%)**. ImageCollection is not
exposed by this Python wheel; the qualified fallback still traverses intermediate
directories on a miss. File observations do not detect every same-size/time
rewrite, and native allocations/runtime copies are outside the stated cache
budget. See [contracts](tiff-page-access.md) and
[five projects](<../examples/opencv tools/tiff_page_access/README.md>).
The subsequent S1-160 completion is recorded above.

**Bounded encoded-video decoding (S1-157)** adds `cv.decode_video`, using native
FFmpeg IStreamReader callbacks directly on immutable clip bytes without temporary
video files. Byte/uint8-vector inputs or bounded local snapshots produce ordered
frame groups, an image preview, fresh per-frame tickets/grids and presentation
times. Selection scans decoded ordinals with explicit start/stride/count limits.
Controls expose output formats, native threads/timeouts, orientation, timestamp
and short-clip policies, encoded/decoded budgets and cumulative read/seek limits.
Callbacks contain sticky failures; native capture and reader resources close on
success, invalid input, cancellation and exceptions without partial publication.
Six original MIT FFV1 projects verify exact PNG samples and 10-FPS times, 257-byte
reads, grouped per-frame contour inspection and deliberate read/workspace/invalid
data failures. **156 targeted checks** passed, covering existing replay/control
flow/reference behavior as well as 1/257/4096-byte callbacks, formats, payloads,
XML/Save As, offscreen editor controls and native teardown. **12 installed
Debug/Lean outcomes** matched. App reinstall, exact wheel source bytes, generated
references/counts, strict JSON, deterministic regeneration, ten asset hashes,
links and text contracts were verified; no broad suite was repeated.
At that completion core was **144/175 (82.3%)**, overall **207/470 (44.0%)**. Native false reads do not
distinguish every corrupt clip from clean EOF. Growing/live streams, other
backends and hard native allocation/deadline bounds remain unqualified. See
[contracts](video-buffer.md) and [six projects](<../examples/opencv tools/video_buffer/README.md>).
The subsequent S1-158 completion is recorded above.

**Seeded MIL tracking (S1-156)** adds `sequence.track_mil` with all seven native
parameters, explicit uint8 canonical-grayscale processing, seed/region precedence,
warmup and loss/expiry/reseed, fixed-size box/pose results and source ticket/spatial
metadata. Region centers can gate updates; masks do not mask native MIL training.
Preflight work/state estimates reject excessive configurations before native
allocation. Native exceptions are contained, state survives editor worker handoffs
and saved projects/XML reopen without an opaque native model. Drawable results
honor graphics settings; raster overlays are opt-in. MIL supplies no native
confidence, scale change or part orientation. Five original MIT recorded projects
cover annotated translation, region teaching, template reacquisition and deliberate
region/work failures. Search radius 8 is explicit for their two-pixel motion:
the native default radius 25 showed up to 22.81 pixels of drift in eight local
replays; the tuned radius 8 observed up to 5 pixels. Tests retain an 8-pixel
accepted-center bound after warmup, not a real-scene reliability claim.
**83 targeted checks** and **10 installed Debug/Lean outcomes** passed;
expected failure observations remain failures. App reinstall, wheel source bytes,
generated references/counts, strict JSON, asset regeneration/hashes, links and text
contracts were verified. No broad suite was repeated. Core is **143/175 (81.7%)**,
overall **206/470 (43.8%)** at that completion. See [MIL contracts](mil-tracking.md) and
[five projects](<../examples/opencv tools/mil_box_tracking/README.md>).

**Precomputed optical-flow pyramids (S1-155)** remain Planned. The OpenCV 5.0.0
Python builder returns usable arrays, with optional derivative layers, but both
LK consumer APIs reject all 12 tested tuple/list/UMat variants. No builder-only
block is counted as a reusable cache. A native bridge or tracker-owned cache
still needs ownership/invalidation limits, parity and measured benefit. See the
[reproducible binding audit](optical-flow-pyramid-audit.md). The subsequent
S1-157 completion is recorded above; S1-155 remains open.

**Optical-flow file round trips (S1-154)** complete the existing loader with
`output.flow_file`, native read/write parity, bounded Unicode-safe snapshots,
explicit float64 narrowing/error, invalid-vector policies, optional previews and
exclusive UUID/fixed file publication. A SHA-bound finite JSON companion records
declared direction and optional image-pair pixel/type/encoding fingerprints plus
spatial/pyramid geometry. Reload can verify both anchors and bind their current
index grid without restoring old tickets or rescaling vectors. Warp/refinement
also reject contradictory directions, modified fields or mismatched images.
Five original MIT projects cover analytic signed motion, native round trips,
measured Farneback motion, bare interoperability and deliberate wrong-pair/direction
rejections. On the controlled translation the computed interior median/p95 endpoint
errors are about 0.00018/0.00243 pixels; this is fixture evidence, not an accuracy
claim for general scenes. **134 targeted file/warp/refinement/editor/example checks**
passed. All **10 installed Debug/Lean CLI outcomes** matched, including two invalid
association cases. App reinstalled; wheel source bytes, generated references/coverage,
strict JSON, source hashes, links and text contracts verified. No broad suite was
repeated. At that completion core was **142/175 (81.1%)**, overall
**205/470 (43.6%)**. The subsequent S1-155 binding audit and S1-156 completion are
recorded above. See [file contracts](flow-files.md)
and [five projects](<../examples/opencv tools/flow_file_roundtrip/README.md>).

**Program lifecycle and self-starting ROS demos** add a root **Program → Settings
and scripts** dialog with initial JSON variables, Startup, Before run, After run
and Shutdown hooks, managed services, asset declarations and runtime status.
Hooks run in bounded application-Python processes. Services use explicit argument
lists, readiness waits and owned process trees; ROS services use the existing
separate Lyrical profile. Services persist across Run once, while GUI execution
copies borrow ownership. New/Open, session close and application shutdown close
block resources, run Shutdown and stop services, with asynchronous window cleanup
and a final application-exit cleanup path. Partial startup, cancellation, hook
errors and unexpected service exit have explicit cleanup/diagnostic behavior.
Validation and file opening do not start services. Save As copies declared assets;
JSON/XML, project comparison, exported guides and the assistant API retain settings.
Two self-starting localhost Fast DDS programs reuse the real, attributed VisA PCB
photo for image and data reception without a separately started publisher.
Verification passed **266 focused lifecycle/editor/API/queue/scheduler/persistence/
ROS compatibility checks** and **five opt-in native Lyrical/Fast DDS checks**,
including real image bytes, repeated-run service reuse, native node cleanup and
application-exit joins. All **four installed Debug/Lean CLI runs** reproduced both
examples across two cycles, with closed service status and shutdown activity in
the final receipts. The app was reinstalled; packaged source bytes, asset hashes,
strict JSON, links, text contracts and generated references were verified. No full
regression suite or simulator/robot operation was run. That infrastructure increment
left OpenCV coverage at **141/175 core (80.6%)**, **204/470 overall (43.4%)**;
the subsequent S1-154 completion is recorded above. See [program lifecycle](program-lifecycle.md) and
[the ROS programs](<../examples/industrial integration/ros2_program_lifecycle/README.md>).

An open production-runtime follow-up is explicit **block resource preparation**
on the actual execution instances, including nested ToolBlocks and editor worker
handoffs. It must be bounded/cancellable, retain the execution client's prepared
resources, report resource status separately and run no dummy image, inference,
ticket or inspection result. It must not run on open/validate. Startup scripts
are separate subprocesses and cannot prepare an in-process block worker. The
[GDRNPP preparation study](<../examples/Simple Plugins/gdrnpp/resource_prepare/README.md>)
provides supporting workflow evidence; it does not add this automatic SDK hook.
Track the shared lifecycle seam under [production runtime #9](https://github.com/votecoffee/OpenCVDataflowGUI/issues/9).

A separate plugin/viewer follow-up is **cached-result rendering** on the owning
execution worker. A live output viewer should be able to select a proposal or
change diagnostic image/cloud views from an already completed result, with no
new inference, acquisition ticket, inspection history entry or numeric-output
mutation. Use bounded requests, cancellation and invocation/project/selection
identity checks to discard stale replies; shutdown must dispose queued work.
Expose the capability through a reusable plugin API rather than requiring each
tool to build its own popup. This remains planned and adds no vision completion.

**Codec preflight (S1-153)** now provides `cv.probe_image_file` with native reader
signature checks, page/frame counts and encoder registry queries. File, Writer
and combined modes expose separate optional observations, requested count flags,
an encoder-format dropdown/custom suffix and inclusive page policies. A bounded,
hashed Python snapshot feeds both file probes and supports Unicode source paths;
temporary copies are removed and original bytes remain untouched. No source image
ticket or pixel output is created. Expected external-file/native failures emit
structured failed preflights; invalid configuration emits no result. Reader/count
success explicitly does not certify compressed pixel integrity, image-type/codec
option compatibility, destination permissions or a later independent file read.
Verification passed **59 focused codec/reference/editor/example checks** covering
stills, mixed TIFF pages, APNG/GIF/WebP frames, writer availability, native failures,
Unicode paths, changing snapshots, limits, ticket payloads, corrupt pixels behind
valid headers, XML, Save As and deterministic fixtures. All **10 installed
Debug/Lean CLI outcomes** matched five original MIT projects. The app was
reinstalled; generated references/coverage, strict JSON, text contracts, links and
wheel source bytes were checked. No broad suite was repeated. Core is now
**141/175 (80.6%)**, overall **204/470 (43.4%)** at that checkpoint. Optical-flow
file round trips are now qualified above (S1-154). See
[the preflight contract](codec-preflight.md) and
[five projects](<../examples/opencv tools/codec_probe/README.md>).

**Run-linked contours (S1-152)** now supplies `inspect.run_contours` for uint8
sparse masks, reusing the ordinary contour measurements/collection/selection
contract. A strip-based horizontal-run preflight bounds native input complexity;
contour and vertex budgets reject oversized output before measurement/JSON work.
Optional raster outputs default off. Offset, mask/region center-or-clip selection,
tickets, pyramid frames, drawable graphics and inspection-count judgments are
qualified. The installed API returns flat links and native run-endpoint polygons:
hole areas can differ from ordinary contours, so this tool explicitly publishes
List semantics without hole parents. Ordinary contour/tree defaults are preserved.
Verification passed **51 focused contour/reference/editor/example checks** and
**10 installed Debug/Lean CLI outcomes** across five original MIT projects,
including deliberate run-budget rejection and count failure. Independent rectangle
geometry, raw native endpoints, degenerate masks, hierarchy limitations, selection,
offset frames, guards, XML, Save As and deterministic regeneration were checked.
A reproducible Windows native-only benchmark on 8000-square sparse masks observed
61.21 MiB versus 0.72 MiB additional peak commit; native medians were 25.73 versus
3.33 ms, with a separate 16.26 ms run preflight. It excludes graph/JSON/raster
cost and is not a speed guarantee. Generated references, strict JSON, local links,
text contracts and wheel bytes were checked; the application was reinstalled.
Core is now **140/175 (80.0%)**, overall **203/470 (43.2%)**. This closes the
20-increment cycle; the 4,801-pass broader checkpoint above is complete. Codec
preflight is now qualified separately above. See [the explicit native limits](run-contours.md)
and [five projects](<../examples/opencv tools/run_contours/README.md>).

**Binary hit-or-miss (S1-151)** now provides `cv.hit_or_miss` with a visual
foreground/background/ignore kernel grid. Standard shape presets remain available;
ordinary Morphology keeps its binary editor. Strict uint8 binary validation,
explicit nonzero/polarity modes, anchors, repeated erosions and native border
controls have qualified behavior. Full-kernel support excludes padded evidence
by default and bypasses native work when no complete footprint fits. Mask/region
inputs filter centers while neighborhoods retain the source image. Complete masks
and total counts remain available when coordinate/graphic caps sample the results.
Optional count gates emit valid failed inspections and reasons; invalid inputs
and exceptions emit no processing result. Tickets and spatial/pyramid payloads
retain their source context.
Verification passed **87 targeted morphology/reference/editor/example checks**,
including an independent Boolean erosion oracle, asymmetric/even kernels, both
constraint polarities, selection/frame guards, bounded coordinates, graphics-off,
signed-cell editing/resizing, isolated last-input preview, XML, Save As and the
existing binary editor. All **12 installed Debug/Lean CLI outcomes** matched six
original MIT projects, including deliberate inspection failure and invalid input.
Generated references/coverage, deterministic fixtures, strict JSON, local links,
text contracts and wheel source bytes were checked; the app was reinstalled.
No broad suite was repeated. Core is now **139/175 (79.4%)**, overall
**202/470 (43.0%)** at that increment. Run-linked extraction is now qualified above. See
[the matching contract](hit-or-miss.md) and
[six projects](<../examples/opencv tools/hit_or_miss/README.md>).

**Multipage memory transport (S1-149)** now provides `cv.encode_pages` and
`cv.decode_pages` under OpenCV Image Codecs. Ordered groups retain mixed sizes,
uint8/16 or float32/64 samples, logical channel order and per-page tickets,
coordinate/pyramid metadata and labels through immutable typed packets. Raw
TIFF bytes are also available, with explicit fresh-page provenance on decode.
Selection/stride, preview, format/depth conversions, compression, prediction,
strip height and DPI have controls and constraints. Directory preflight bounds
declared pages, fields, offsets, pixels and conversion workspace before native
pixel decoding; encoder readback verifies exact samples before publication.
Verification passed **96 targeted codec/reference/editor/example checks**, including
classic/BigTIFF and both byte orders, mixed depths/alpha, independent integer
TIFF/DPI readback, payloads, selection guards, contained failures, XML, Save As
and deterministic original fixtures. All **10 installed Debug/Lean CLI outcomes**
matched five projects, including deliberate budget and predictor rejections.
Generated references, strict JSON, text contracts, local links and wheel contents
were checked; the app was reinstalled. No broad suite was repeated. Core is now
**138/175 (78.9%)**, overall **201/470 (42.8%)** at that increment. Hit-or-miss
morphology is now qualified separately above. Native EXR still requires a
codec-capable build (S1-125); float64 multichannel TIFF cross-reader metadata
qualification remains open. See [the buffer contract](multipage-memory.md)
and [five projects](<../examples/opencv tools/multipage_memory/README.md>).

**Multiband mosaics (S1-117)** now extend registered image pairs, ordered scans
and bounded tile rendering. A mask-aware Float64 Gaussian/Laplacian path preserves
high-depth/HDR/alpha signal through composition; the explicit native CPU detail
backend offers float32/int16 weights for uint8 gray/RGB/BGR. Source-only samples
are restored after warping/correction, original coverage remains authoritative,
and globally aligned filter halos keep tiled output consistent with dense output.
Preflight workspace estimates include pyramid padding and peak buffers. Native
integer precision differs from Float64; the installed build's OpenCL path was
not qualified and is bypassed with caller settings restored.
Verification passed **116 targeted stitching/multiband/reference/editor checks**,
including analytic means/constants and controlled intensity steps, direct CPU
native references, HDR, odd grids, masks, dtype/encoding, tiled coordinates,
workspace guards, failure containment, controls and XML. All **14 installed
Debug/Lean CLI outcomes** matched seven new real-PCB-photo projects, including
the intentional native-depth rejection. Source/derivative attribution and hashes,
deterministic regeneration, strict JSON, text contracts, local links, generated
references and wheel contents were checked; the app was reinstalled. No broad
suite was repeated. Core is now **137/175 (78.3%)**, overall **200/470 (42.6%)**.
In-memory multipage transport is now qualified separately above. Native EXR qualification
(S1-125) still needs a codec-capable build. See [the blending contract](multiband-mosaics.md)
and [seven runnable projects](<../examples/opencv tools/multiband_mosaic/README.md>).

**Connection health monitors** now provide ROS 2, MQTT, EtherNet/IP, ControlLogix
and FANUC status without image inputs or required graph connections. Protocol
failures alarm red; responding endpoints with failing configured resources warn
amber. Critical ROS node/service absence also alarms. Optional read-only empty
GetParameters calls distinguish advertised from responsive ROS services; other
service types use discovery only. Thumbnails show the latest observation and
turn amber when stale, including in Lean mode. Hidden optional outputs, explicit
fail-on-alarm, interval caching and grouped endpoint logs support supervision
without creating image tickets. See [contracts and limitations](connection-health.md)
and [eight disabled/offline projects](<../examples/industrial integration/connection_health/README.md>).
Verification passed **164 focused protocol/runtime/scheduling/diagnostic/editor
checks** and **four native localhost Lyrical/Fast DDS checks**. Real local Paho
handshakes distinguish rejected subscriptions, missing/retained messages and
missing CONNACK. Native ROS checks cover receipt freshness, QoS/interface faults,
read-only service responses, an advertised but unresponsive service, worker
failure/recovery and cleanup. CIP/FANUC read paths use controlled replies; no
physical PLC/robot acceptance is claimed. All **16 installed Debug/Lean CLI
outcomes** matched (two activations each), including intentional offline alarms.
Deterministic project regeneration, strict JSON, text contracts, local links and
wheel contents were checked; the app was reinstalled. No broad suite was repeated
and no OpenCV coverage count changed. A supervisor independent of blocked graph
iterations, additional service-specific read-only probes and shared transport
pooling remain future work; current monitoring is explicitly scheduled by graph
activations.

**Camera health monitoring** (`inspect.camera_health`) now compares each image
with an explicitly trained, immutable reference/history. Background tracking
rejects coherent moving foreground; alternatively, one or more trained static
fiducials supply position evidence. Named lighting/focus regions follow the
measured image-plane transform. Separate warning/alarm limits cover position,
rotation, scale, channel-mean change and sharpness loss. Missing or insufficient
evidence alarms with reasons. The modeless trainer captures completed input
publications or imported images, uses the existing region editor, trains private
copies in a worker, and stages a pinned asset for explicit inspector Apply.
See [the training and measurement contracts](camera-health.md) and
[ten real-photo replay projects](<../examples/custom tools/camera_health/README.md>).
Verification passed **83 focused monitor/focus/plugin/editor checks**. All
**20 installed Debug/Lean CLI outcomes** matched, including controlled warning,
alarm and unavailable-evidence cases. Known translations, rotations, scales,
lighting gains and blur, multiple regions/history, grayscale/color/high-depth
formats, tickets, graphics-off, exception containment, sealed assets and XML
were checked. Deterministic image/model/project regeneration, source attribution,
strict JSON, text contracts, local links and wheel contents were checked; the app
was reinstalled. These image-based health indicators are not physical camera
calibration. This composite workflow reuses existing native capabilities and does
not change OpenCV coverage counts. No broad suite was repeated.

**Differential exposure comparison (S1-116)** now provides robust local/global
gain/offset fitting for already aligned reference/candidate images. Separate
fitting and inspection selections, effective support, gain/contrast/saturation
guards and explicit coverage limits keep unsupported pixels out of judgments.
Raw and adjusted residuals precede output rounding/clipping; per-channel reports,
drawable defect graphics and ticket failure reasons retain inspection evidence.
This complements flat-field and single-image illumination tools. Stable fit areas
and defect-scale validation are still necessary: broad defects can be learned
as lighting. See [the fitting contract](differential-exposure.md).
Verification passed **122 focused photometric/reference/editor checks**, including
independent local least-squares fits, known gains/offsets, a specified 20-pixel
defect, raw/adjusted metrics, native depths/alpha, masks, support and frame failures,
roundoff limits, XML and controls. All **eight installed Debug/Lean outcomes**
matched four new original examples, including intentional defect/support failures.
Deterministic PNG/project regeneration, strict JSON, generated coverage/reference
checks, source text contracts, links and wheel contents were checked; the app was
reinstalled. No broad suite was repeated. The multiband extension (S1-117) is now qualified above; native EXR still needs
a capable build.

**Coverage-aware mask resizing (S1-118)** now extends the existing resize/region
baseline with `cv.resample_mask`. Whole-grid pixel-cell area integration returns
binary acceptance, fractional coverage, resize geometry and source provenance.
Threshold one requires complete validity, zero accepts any positive area, and
intermediate limits are inclusive. Known target-frame mismatches are rejected.
Three original synthetic examples compare thresholds, aligned fractional chains
and a contained crop/resize mismatch. See [mask coverage contracts](regions.md#resample-validity-mask-coverage).
Verification passed **111 focused mask/region/native-reference/editor checks**.
All **six installed Debug/Lean outcomes** matched, including the deliberate
target-frame rejection. Exact rational cell intersections and native area
downsampling checks cover arbitrary ratios, holes, thin stripes, fractional
inputs and coverage conservation. Regeneration, JSON, source text contracts,
generated references and packaged module/registry contents were checked; the
app was reinstalled. No broad suite was repeated. Core coverage reached
**135/175 (77.1%)**, overall **198/470 (42.1%)** at that checkpoint. The paired
exposure follow-up is described above; native EXR qualification remains open.

ROS **ROI/binning-aware intensity rectification** extends the existing block
without adding another type. Raw K and rectified P are adjusted independently.
A dropdown selects the CameraInfo ROI, full sensor canvas or custom rectified
ROI, followed by optional output resizing. `do_rectify=true` requires an explicit
canvas choice; no tight ROI is estimated. Exact declared software crop/binning
transforms are accepted, unrelated pixel warps rejected, partial bins reported
and uncaptured output pixels remain invalid in the coverage mask.
Verification passed **161 focused native-OpenCV/offline/editor checks**, including
independent full-sensor projection references, pinhole/fisheye maps and pixels,
nonuniform binning, ROI/range failures, software spatial transforms, XML and enums.
All **six installed Debug/Lean outcomes** matched the three rectification examples.
The new real-PCB fixture explicitly labels its synthetic calibration and software
acquisition geometry. There are now nineteen ROS examples; live endpoints stay
disabled. The app was reinstalled and the wheel checked. No broad suite or
robot/simulator operation was performed. Automatic tight ROI estimation, stereo
baseline handling and physical depth reprojection remain open. See
[rectification contracts](ros2-rectification.md).

ROS **full-grid CameraInfo intensity rectification** now remaps pinhole/fisheye
images with explicit K/D/R/P, including nonidentity R. It returns a new optical
and pixel frame, current-grid camera/CameraInfo, raw-to-rectified rotation,
source-center validity mask and reusable coordinate maps. Ambiguous scalar data
requires an Intensity declaration; physical depth, already-rectified models
and stereo P translation are rejected. The first increment rejected cropped/binned
grids; the follow-up above adds explicit ROI/binning handling. Calibration source
headers and static bindings remain in Info rather than being silently rewritten.
Two new examples use a real PCB photo with clearly synthetic calibration and a
disabled static/raw-image subscriber. See [rectification contracts](ros2-rectification.md).
Verification passed **137 focused native-OpenCV/offline/editor checks**, including
independent point projection, pinhole/fisheye pixel parity, formats, border modes,
map reuse, endian normalization, ticket/run/recipe isolation, XML and typed controls.
All **four installed Debug/Lean outcomes** matched the new rectification examples.
The separately published bin-to-grid app/source packages were integrated through
`af16ddd` and its manifest fix `78c8cec`. Four pinned manifests (124 files) and two
Debug/Lean recorded-review replays passed; those replays checked saved evidence
and did not rerun simulator motion. The app was
reinstalled; no broad suite or robot/simulator operation was performed here.
Stereo baseline, automatic tight rectified ROI estimation and physical depth rectification
remain future work.
These native calls already exist in the coverage checklist; no OpenCV completion
row or denominator change is claimed for this ROS workflow.

ROS **ordered image timestamp pairing** now buffers two streams with positive
integer acquisition headers. Earliest feasible FIFO heads pair within an explicit
tolerance; older unmatched heads and receipt-time expirations are dropped.
Queues are bounded, duplicates do not replay, and backward stamps hold until
reset/new run context. Common pair tickets retain both original headers/tickets
and optional acquisition provenance. No clock alignment, spatial registration or
physical synchronization is inferred. Pair before static CameraInfo binding for
registered RGB-D. A real PCB fixture explicitly uses artificial times; a disabled
RGB-D template shows the complete chain. See [pairing contracts](ros2-time-pairing.md).
Verification passed **176 focused offline/editor/coordination checks** and one
native localhost DDS check: an out-of-window image was discarded, a later pair
matched at 1 ms, and static calibration produced the expected colored meter cloud.
Independent FIFO references exercise varied arrival interleavings; integer-ns,
limits, expiry, holds, run/recipe isolation, XML and editor controls were checked.
All **four installed Debug/Lean outcomes** matched the two new examples. The app
was reinstalled, wheel source contents and cleanup checked, and the catalog
imports without ROS extensions. No broad suite, simulator or robot operation
was performed.
This integration does not change OpenCV coverage counts. General out-of-order
matching, multi-stream assignment and clock alignment remain future milestones.

ROS **static CameraInfo reuse** now applies a cached calibration to successive
ticketed images without rewriting its original acquisition header. Explicit
bindings record calibration revision, original provenance, image ticket/grid,
receipt age and optional run context. Calibration updates alone emit nothing;
early images are discarded, expiry/reset invalidates the cache, and stale run
identities are rejected. The RGB-D validator's new Bound static mode verifies
these bindings while Per acquisition remains the default. A real T-LESS example
and disabled retained-calibration subscriber bring the ROS catalog to fourteen.
Verification passed **209 focused offline/editor/coordination checks** and one
native localhost DDS check: calibration was published once before subscription,
then reused for two independent image tickets and their meter point clouds.
All **eight installed Debug/Lean CLI outcomes** matched the four new/updated
calibration graphs; repeated offline runs issue fresh processing tickets for the
same recorded image. The app was reinstalled and packaged sources checked.
No broad suite, simulator or robot operation was performed; this integration
increment does not change the OpenCV coverage denominator.
See [static calibration contracts](ros2-static-camera.md). Approximate-time image
pairing now has a two-stream ordered implementation; broader timing and
stereo/rectification support remain separate work.

ROS **strict poses, acquisition-time TF and run/reset identity** now provide six
additional blocks. Stamped pose/transform conversion names parent/child direction,
meters/millimeters and quaternion policy. A persistent optional TF2 worker queries
positive acquisition time with bounded waits, age limits and cancellation; observed
clock resets invalidate its epoch. Static lookup timestamps are query times, not
source measurements. One run authority emits an unsaved session/generation and
opaque token; acquisition tags and guards reject stale local values/groups or
explicit external envelope tokens. These are data checks, not motion permission.
Three new examples cover numeric poses, a real PCB run-scope workflow and a
disabled TF acquisition. See [pose/TF contracts](ros2-poses-tf.md).
Verification passed **221 focused offline/editor/coordination checks**, **five
native TF checks** and three existing native DDS spot checks. Exact interpolation,
inverse direction, static-history recovery, age/future limits, backward clock
jumps, stale contexts and cancelled-worker cleanup were checked on localhost.
All **24 installed Debug/Lean CLI runs** matched the twelve ROS examples. The
separately integrated planning package passed its 22 core checks and fourteen
pinned file hashes. The app was reinstalled and its wheel contents checked;
Python 3.12 worker syntax remains compatible with the isolated Lyrical runtime.
No OpenCV coverage row was added, broad suite repeated or robot/simulator operated.

ROS **CameraInfo conversion and per-acquisition registered RGB-D validation** now
reuse the existing camera model and ticket coordinator. Three additional blocks
support raw K/D or conservative rectified P intrinsics, ROI/binning, canonical
CameraInfo encoding and strict ticket/frame/image-size/integer-stamp checks.
Depth scale is explicit and the output is physical meters. Two new examples
cover a real T-LESS recorded acquisition and a disabled three-topic subscriber.
The recorded acquisition reports its missing ROS timestamp explicitly.
Verification passed **183 focused offline/editor/coordination checks** and **two
native DDS checks**, including staggered color/depth/CameraInfo arrivals through
the real coordinator into calibrated 3D points. All **18 installed Debug/Lean ROS
example runs** matched. Nine existing hand–eye tests and the ChArUco example also
passed on the current OpenCV 5 runtime: hand–eye uses the repo's Park-style NumPy
solver with Rodrigues, not an unavailable `cv2.calibrateHandEye` binding.
The separately authored FoundationPose file-review examples were integrated by
fast-forward and replayed without inference; their rejected evaluations remain
unchanged. No broad suite, simulator or robot operation was performed.

Optional **ROS 2 Lyrical / Fast DDS** now provides image, point-cloud and structured-data
readers, gated publishers, offline converters and registered RGB-D validation under
**Industrial integration → ROS 2**. A separate ROS Python worker keeps native
ROS extensions independent of the Python 3.14/OpenCV 5 GUI. Sources preserve
headers and acquire tickets only on accepted messages; bounded queues and
explicit QoS/timeout controls expose waiting and drop behavior. Ambiguous writes
hold until reset. Nineteen [public examples](<../examples/industrial integration/ros2_fastdds/README.md>)
include real VisA PCB/T-LESS RGB-D converter graphs and disabled live templates.
See [runtime setup](ros2.md) and [issue #20](https://github.com/votecoffee/OpenCVDataflowGUI/issues/20).
The next milestones are:

1. General out-of-order/multi-stream image matching and clock alignment;
   stereo-baseline reference handling and automatic tight rectified ROI estimation;
   extended cloud-field editing and
   validated remerging after point edits. Per-acquisition CameraInfo/RGB/depth
   association, explicit static CameraInfo reuse and PointCloud2 conversion/read/write
   and two-stream ordered timestamp pairing are implemented. Monocular
   nonidentity-R intensity rectification, raw ROI/binning and explicit output grids
   have their own image/frame block.
2. Integrate local run tokens with external capture/service envelopes, request maps,
   server/robot/scene/frame revisions and held-out timing checks. Strict stamped
   poses/TF and local run/reset guards are implemented; rosbag2 and source TF
   sample-age diagnostics remain open. Standard headers do not carry run identity.
3. GUI services/actions with identities, cancellation, uncertain-outcome handling
   and trajectory/scene review. The optional source-only
   [plan-only ROS package](../integrations/ros2/README.md) is integrated separately;
   it does not by itself add GUI action controls or execution authority.
4. Coordinated Gazebo/ros_gz/MoveIt camera/robot examples and measured image/cloud
   throughput. Shared simulation dependencies and jobs require owner coordination.
   Reuse ChArUco → PnP → hand–eye tools and generic MarkerArray data publication;
   keep result-to-marker adapters in the tool plugins. ROS owns scene/TF/camera
   publishers and SimpleTools owns target detection/estimation. Core ROS message
   blocks do not authorize motion. The separately packaged
   [bin-to-grid simulation app](<../examples/Simple Plugins/bin_picking/bin_to_grid/README.md>)
   includes explicit simulation-only actions and recorded native acceptance;
   hardware motion and eye-in-hand timing/calibration remain separate qualifications.

This is an industrial integration track; it adds no OpenCV coverage completion
and native localhost DDS tests do not establish simulator or robot acceptance.

The optional **SimpleITK** integration has its own
[medical imaging roadmap](simpleitk-roadmap.md) and
[tracking issue #19](https://github.com/votecoffee/OpenCVDataflowGUI/issues/19).
Its foundation provides twelve blocks, typed 2D/3D medical images, explicit
OpenCV converters, remergeable metadata and physical grid reconstruction, native
processing/overlap measurements and nine public MRI/expert-label examples under
`examples/SimpleITK Tools`. Separate toolbox roots are **SimpleITK**,
**scikit-image** and **Simple Plugins**. Next medical milestones cover DICOM
series, bound metadata sidecars, reference-grid resampling, volume review,
segmentation/measurements and registration. This track does not change the
OpenCV coverage denominator or displace its existing prioritized checklist.

Assistant application development now has two portable skills, a documentation
map and planning template, and a versioned JSON/Python saved-project API. The API
discovers block metadata, applies validated edit batches, protects known file
revisions, runs offline projects and exports review previews/HTML/XML. This is
saved-file co-work; it does not expose unsaved GUI drafts or a remote service.
See [assistant development](assistant-development.md). This application capability
does not add an OpenCV coverage completion.

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
S1-125 remains partial until native EXR runs on a capable build.
Coverage-aware mask resizing (S1-118) now qualifies explicit pixel-cell area,
thresholds and provenance alongside the existing resize and transformed-region
tools. Differential exposure comparison (S1-116) now adds robust paired models,
raw/adjusted defect residuals and coverage qualification. Multiband composition
(S1-117) now covers high-depth masks and bounded, phase-aligned tile rendering.
[Large-radius stack smoothing](<../examples/opencv tools/stack_blur/README.md>)
(S1-159) now has bounded independent kernels, selected writes, high-depth-safe
float32 work and original-depth output. Five projects cover Gaussian comparison,
selection, uint16 processing, unsafe native rejection and explicit clamping.
Independent triangular references and matched-variance Gaussian measurements
record noise, edge position/width and local native-call time. The 127x127 fixture
reduces flat-area RMS error from 18.044 to 0.171 DN while broadening its edge to
70.71 pixels; it is not an edge-preserving filter. Native integer artifacts on
this build are guarded rather than emitted as valid results.
[Animated image codecs](<../examples/opencv tools/animation_codecs/README.md>)
(S1-111) now qualify APNG/GIF/WebP file and buffer paths, per-frame timing and
contexts, explicit GIF quantization, protected output and native readback checks.
Six original projects cover exact storage, per-frame inspection, file output
and an intentional timing failure. Targeted checks cover pixels, alpha, logical
color order, provenance, limits, enums, XML and contained failures. Both installed
CLI profiles reproduce all six expected outcomes. AVIF and native RGB APNG
posters remain unqualified; those inputs fail clearly. This adds batch codec
processing; GUI playback pacing remains separate.
[Native QR label generation](<../examples/opencv tools/qr_encoding/README.md>)
(S1-112) now exposes all native encoder controls, with checked text modes,
UTF-8/Shift-JIS handling, explicit quiet zones, integer module scaling and bounded
contact sheets/FlowGroup outputs. Generated grids retain source tickets without
reusing camera coordinates. Four original projects cover ordinary/Unicode native
readback, grouped structured-append generation and an intentional capacity failure.
Validation: 57 targeted checks including existing code readers; native matrices,
independent rendering, optional ZXing Kanji recovery, XML and graphics-off are
covered. Installed Debug/Lean CLI checks exercise all four projects. Structured
generation does not close S1-162 recovery or qualify print grading. The completed
offscreen checkpoint is recorded under [Validation cadence](#validation-cadence).
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
| 1 | [OpenCV core and baseline 2D](https://github.com/votecoffee/OpenCVDataflowGUI/issues/17) | Close common gaps in image processing, geometry, replay, and measurement. Cross-tile box fusion (S2-27), appearance tracking (S2-28), prompted segmentation (S2-29) and tiled segmentation (S2-30) are qualified; pinned text detection (S2-32), batched OCR (S2-33), constrained alphabet/lexicon decoding (S2-34), pinned learned single-image super-resolution (S2-35) and same-grid learned restoration (S2-36) and learned monocular depth (S2-37) and learned stereo disparity (S2-38) and learned optical flow (S2-39) and learned video tracking (S2-40) and image-descriptor PCA (S2-41) and texture/color mixtures (S2-42) and image-feature outlier detection (S2-43) and visual-word image retrieval (S2-44) and object landmark skeletons (S2-45) are qualified; HOG sliding-window object detection (S2-46) is qualified; cascade-based image detection (S2-47) is qualified; rotated crop classification (S2-48) is qualified; image-based regression (S2-49) is qualified; local anomaly maps (S2-50) and input-shift evaluation (S2-51) are qualified; next qualify learned instance tracking masks (S2-52); pinned SSD/YOLOX exports (S2-25/S2-26) are qualified; classifier/detector evaluation (S2-23) is qualified on bounded controlled images. Retain real SuperPoint extraction as open; random forest classification (S2-22) is qualified on controlled images; K-nearest classification (S2-21) is qualified on controlled images; reusable color/descriptor clusters (S2-19) are qualified on controlled images; named inputs, image batches, model receipts, detection/NMS, rotated detection, semantic/instance segmentation and ordered keypoint/heatmap decoding baselines are qualified. S1-121 now qualifies seven controlled curved-label projects, source/module provenance and explicit native-quadrilateral limits. S1-115 now qualifies shared camera/recorded setting negotiation, actual-frame/readback reports, explicit failed-image gating and five hardware-free native projects; physical camera acceptance remains device-specific. S1-84 now qualifies conditional line-PSF search on isotropic texture, disjoint tile validation, noise/ambiguity/boundary gates, current-pixel provenance and gated Wiener restoration with eleven independently generated projects. S1-148 now qualifies shared-root neighbor transfer, nearest/mean review fills, unchanged measured support, overlap/change/distance/selection gates, explicit incomplete corners and seven known-jitter/native-fit projects. S1-147 now qualifies explicit native estimation and none/ray/reprojection adjustment, original-grid initial/fixed/refined intrinsics, reserved-keypoint validation excluded from native fitting, native-call instrumentation and nine known-camera projects. S1-104 now qualifies unordered native panorama fitting, independent overlap/model checks, explicit partial policies, high-depth supported rendering and nonlinear source-point conversion with eight known-camera projects. S1-150 now qualifies response-aware HDR consistency, clipped-code contradictions, explicit measured/review masks, per-source native alignment support and six exact timed controls. S1-61 now qualifies projected template-appearance support with separate coverage, retained detector scores, selection/transform/ticket contracts and four native detector-to-pick examples. S1-69 now carries whole selections through nonlinear correction with exact raster regions, independent coverage/support, analytic references and five saved projects. S1-123 now qualifies bounded live timing diagnostics with raw/effective clocks, valid VFR/strict cadence/reference policies, session counters, plotting and six independently authored native Matroska controls across eight pipelines. S1-110 now qualifies independent requested-time sampling with native time seeks, bounded timestamp brackets/fallback, actual time/error and index provenance, fresh sample tickets/grids, exact lossless/compressed references and five annotated projects. S1-12 now qualifies owned-process FFmpeg network acquisition with cancellable response deadlines, bounded reconnects, stream resets and native loopback MJPEG projects. S1-11 now qualifies an explicit media-clock source, labelled timing policies, cooperative Stop/Resume, stride/speed, loop resets and three native lossless projects with fake-clock and offscreen integration evidence. S1-173 now qualifies exact common pixel/physical profile grids, rate-aware antialiasing, null/boundary support and source-coordinate provenance with three high-depth examples; the native resampleSignal binding remains unqualified. S1-172 now qualifies reusable detector cadence, bounded local LK similarity evidence, explicit detector age and provenance, native/controlled drift references, entering/lost parts and three independently counted budget/freshness replay projects. S1-54/S1-171 now qualify center/log-size/angle prediction with gated global assignment, Joseph covariance, wrapped angles, bounded uncertainty/expiry, current ticket/grid, analytic crossing/time references and two annotated native-blob replay projects. S1-170 now qualifies sealed clean-image capture/replay, declared camera/calibration and grid/recipe binding, first-current-image masks, explicit KNN freeze/guard workarounds and eight original projects. S1-169 now qualifies reference-based row motion, withheld-track validation and support/geometry guards, current-pixel video correction, explicit nonlinear source maps and six original projects. S1-168 now qualifies registered-neighbor detail restoration with explicit blur/sharpness/change gates, exact fallback/support and current-ticket contributor provenance, independent moving-part truth and six saved projects. S1-167 now qualifies content/timing-bound affine capture/persistence/replay, explicit validity and omissions, current-ticket coordinate provenance, safe JSON assets and reset/native-fit parity. S1-166 now qualifies supported relative gain/phase, shared selections/windows, numerical floors, exact independent blur/shift/brightness references, spectral-coordinate provenance and explicit judgments. S1-165 now qualifies independent timestamped replay groups, full/union/common-support grids, exact masks, depth-preserving and native factory paths, source-coordinate maps and per-ticket Unbatch inspections. S1-164 now qualifies stable source IDs, reordered/missing acquisitions, preserved blend priority and full-canvas coordinates, explicit incomplete/failure policies, strict grids, current tickets and sealed JSON layout persistence. S1-163 now qualifies center-based repeatability and thresholded matching quality under independently supplied transforms, maximum one-to-one assignments, shared support, descriptor families, explicit validity and diagnostic grids. S1-162 now qualifies ordered multipart QR recovery from scenes/groups, actual module-header/codeword checks, explicit rejection and parity-collision limits, per-member tickets and strict complete-payload emission. S1-161 now qualifies standard/ArUco QR reading, all finder controls, independent camera-corner and exact module references, explicit small-label backend differences and lifecycle/payload checks. S1-160 now qualifies centered per-pixel cumulative/weighted noise and paired covariance, explicit support/validity, ticket/grid contracts, independent numerical references and state/resource lifecycle checks. S1-158 now qualifies bounded indexed TIFF snapshots and raw-page LRU caching, exact mixed-depth page references, fresh acquisition tickets/content grids and explicit refresh/resource/lifecycle contracts. S1-157 now qualifies bounded native FFmpeg in-memory decoding, exact FFV1 frame/time references, explicit short-clip/timestamp policies, grouped frame tickets, resource guards and teardown. S1-155 remains Planned after both Python LK APIs rejected all 12 tested pyramid representations; a cache consumer still needs implementation and measured benefit. S1-156 now qualifies seeded MIL tracking with all native controls, explicit loss/reseed, tuned recorded examples and no confidence claim. S1-154 now qualifies bounded native .flo round trips, precision/invalid-vector policies and verified image-pair direction/grid metadata. S1-153 now qualifies native codec preflight with bounded snapshots, separate registry/header/count observations and explicit pixel-integrity limits. The latest broad checkpoint at 2a66fc8 passed 6,986 tests with 36 expected skips and unchanged tracked-file/package manifests; the twenty-increment cycle through S2-11 is closed and the twenty-increment cycle through S2-33 is closed by the broad audit and scoped repairs; current targeted cycle 19/20 after constrained OCR, learned super-resolution, restoration, monocular depth, optional-input plugin review, learned stereo, learned optical flow, learned tracking, descriptor PCA, image mixtures, image-feature outliers, visual-word retrieval, object landmark skeletons, HOG/cascade detection, pose-aligned classification, regression, local anomaly maps and input-shift evaluation, including classifier/detector evaluation (S2-23) and explicit engine-dispatch reliability. Concurrent untracked example work was preserved (program lifecycle infrastructure is outside vision coverage). S1-152 now qualifies bounded native run contours with explicit flat topology and different hole-polygon semantics. S1-151 now qualifies signed-kernel binary pattern matching with bounded coordinates and count judgments. S1-149 now qualifies ordered typed TIFF buffers, exact sample readback and bounded page decoding. S1-117 now qualifies native CPU/Float64 multiband composition with source-depth, mask, workspace and tiled-phase contracts. S1-116 now qualifies robust paired lighting models and raw/adjusted defect residuals with independent fit selections and support guards. S1-118 now qualifies area-coverage mask resizing with explicit thresholds and source/target coordinate checks. S1-159 now has bounded stack smoothing and measured Gaussian noise/edge/runtime comparisons. S1-111 now has verified APNG/GIF/WebP frame/timing transport across file and buffer APIs. S1-112 QR label generation now has checked native controls, text recovery and bounded grouped outputs. S1-07 metadata file/buffer round trips now have verified native chunk transport and explicit orientation semantics. S1-109 paired flow refinement now has independent motion/residual references and explicit support/failure evidence. S1-108 color-distribution tracking now has controlled translation/scale/loss evidence. S1-95 dense-flow warping and S1-88–S1-90 photo/color edits have controlled references and explicit support/failure evidence. Native EXR qualification for S1-125 remains open because this build has no EXR codec; then follow the remaining coverage checklist. Sequence and calibrated measurement follow-up is tracked in [#14](https://github.com/votecoffee/OpenCVDataflowGUI/issues/14). | Common inspection workflows run headlessly with examples and validated parameter, pixel-depth, selection and coordinate contracts. Use the binding inventory to evaluate candidate APIs. |
| 2 | [ML and DNN](https://github.com/votecoffee/OpenCVDataflowGUI/issues/15) | Cross-tile box fusion (S2-27), appearance tracking (S2-28), prompted segmentation (S2-29) and tiled segmentation (S2-30) are qualified; pinned text detection (S2-32), batched OCR (S2-33), constrained alphabet/lexicon decoding (S2-34), pinned learned single-image super-resolution (S2-35) and same-grid learned restoration (S2-36) and learned monocular depth (S2-37) and learned stereo disparity (S2-38) and learned optical flow (S2-39) and learned video tracking (S2-40) and image-descriptor PCA (S2-41) and texture/color mixtures (S2-42) and image-feature outlier detection (S2-43) and visual-word image retrieval (S2-44) and object landmark skeletons (S2-45) are qualified; next qualify HOG sliding-window object detection (S2-46); pinned SSD/YOLOX exports (S2-25/S2-26) are qualified, then remaining image-model adapters. Classifier/detector evaluation (S2-23) is qualified for the documented local axis-aligned baseline. Native random forests (S2-22) are qualified on controlled CPU images. Exact brute-force K-nearest classification (S2-21) is qualified on controlled CPU images. Reusable color/descriptor clusters (S2-19) are qualified on the stated controlled CPU fixtures. ALIKED extraction/matching (S2-17) is qualified on the stated CPU fixtures. DISK extraction/matching (S2-16) is qualified on the stated CPU fixtures. S2-15 has a qualified external tensor/matcher path but remains Partial pending permitted real SuperPoint extraction. Batched/dynamic ONNX, revisioned model receipts, semantic/instance masks and ordered landmarks are qualified; continue measured LightGlue adapter and K-nearest/random-forest work. [Named multi-input inference](multi-input-inference.md) (S2-03) is qualified with independent preparation and complete named bindings. The [object detection/NMS baseline](object-detection.md) (S2-08) is now qualified with explicit row/grid/source contracts and CPU fixtures. Native SVM training, prediction and model I/O are available in [the classical ML workflow](classical-ml.md) (S2-20). | Versioned model and preprocessing fixtures reproduce CPU results. Measure accelerator support on each backend used. |
| 3 | 3D and calibrated geometry | Finish calibration revision management and offline hand-eye work in [milestone 2](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/2). Extend stereo and prioritize cloud interoperability, cleanup, segmentation, comparison, mesh output and bounded reconstruction in [milestone 6](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/6). | Held-out targets meet declared geometric tolerances. Recorded sequences demonstrate frame, unit and memory bounds. Live robot pose exchange follows offline geometry and acquisition timestamp validation. |
| 4 | Contrib expansion | Select extra-module tools by workflow value, starting with segmentation, texture, restoration and specialized feature or motion methods from the [coverage backlog](opencv-coverage.md). | Each selected tool has a qualified Python path on a supported build and ships with a CPU fixture, example and documented limits. |
| 5 | Application-specific ToolBlocks | Package reusable tray orientation, conveyor pick selection, depth-assisted bin picking, surface and bead inspection, OCR/code grading, color checks and robot-frame measurements. | Each package has documented ports and calibration assumptions, known-good and known-bad validation sets, and repeatable ticket verdicts and reasons. Hardware-dependent acceptance is tracked separately. |

The optional Simple3DMeasure 0.1.0.dev2 plugin supplies observed line fitting,
angle, separation, probe distance and span measurements, with
[generated and recorded section examples](<../examples/Simple Plugins/3d_measure/lines/README.md>).
This is the partial baseline for S3-129. A native XYZ `fitLine` path with selectable
robust distances and retained-point identity qualification remains open. Recorded
sections use supplied fixture poses; automatic edge detection and independent
dimensional accuracy remain separate work.

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
- Paired selected input/output plugin captures from one completed execution, including private payloads and explicit retention guards.
- Separate modeless read-only plugin review popups in Results, without training or draft staging.
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
| [SimpleITK medical imaging](simpleitk-roadmap.md) | M1 foundation shipped in `dcc8c5a`: twelve blocks and nine labeled MRI examples. Continue DICOM/metadata persistence, volumetric processing/review and registration in [#19](https://github.com/votecoffee/OpenCVDataflowGUI/issues/19). | Physical geometry, tags and ticket provenance remain explicit; public ground truth and native/independent checks qualify each path. Separate from OpenCV coverage. |
| [Reliable acquisition and replay](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/3) | Independent camera workers, bounded queues, timestamp-window matching, backend enumeration and triggers, multi-source recordings and validation bundles. [#5](https://github.com/votecoffee/OpenCVDataflowGUI/issues/5), [#6](https://github.com/votecoffee/OpenCVDataflowGUI/issues/6) | A stalled source cannot stall another source or coordination deadlines. Replay reproduces ticket outcomes and failure reasons. Physical synchronization needs separate backend and trigger evidence. |
| [Editor and reusable jobs](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/4) | Undo/redo, autosave recovery, schema migrations, promoted parameters, dynamic bindings and versioned ToolBlock/plugin contracts. [#7](https://github.com/votecoffee/OpenCVDataflowGUI/issues/7), [#8](https://github.com/votecoffee/OpenCVDataflowGUI/issues/8). | Existing and nested jobs round-trip with their layout and interfaces. Interrupted saves can be recovered. Start alongside acquisition. |
| [Production runtime and validation](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/5) | Program lifecycle scripts and owned services now provide first-run startup and program/application-close cleanup. Continue operator view, deployable jobs, recipe and asset revisions, durable ticket history, result routing and retention. [#9](https://github.com/votecoffee/OpenCVDataflowGUI/issues/9), [#10](https://github.com/votecoffee/OpenCVDataflowGUI/issues/10), [#11](https://github.com/votecoffee/OpenCVDataflowGUI/issues/11) | Operator and headless runs agree on the same package. An unattended replay pilot survives restart and sink failures, preserves decision provenance, and reports latency, throughput and memory. Depends on replay and compatible job files. |
| [Measured execution and browser feasibility](https://github.com/votecoffee/OpenCVDataflowGUI/milestone/8) | Opt-in compiled ToolBlocks and a scoped OpenCV.js/WebAssembly viewer study. [#4](https://github.com/votecoffee/OpenCVDataflowGUI/issues/4), [#16](https://github.com/votecoffee/OpenCVDataflowGUI/issues/16) | Compilation preserves output semantics and selected diagnostics and shows a repeatable end-to-end benefit after amortization. The browser study ends in a go/no-go decision based on module coverage, transport, latency and memory. [Benchmark criteria](opencv-backend.md#compiled-toolblocks-future). |

Start with editor undo/recovery ([#7](https://github.com/votecoffee/OpenCVDataflowGUI/issues/7))
and the shared acquisition/replay contract ([#5](https://github.com/votecoffee/OpenCVDataflowGUI/issues/5)
and [#6](https://github.com/votecoffee/OpenCVDataflowGUI/issues/6)). Current
graph save/load, per-ticket diagnostics, local replay and the headless runner
provide the baseline for these milestones. Operator packaging and durable
ticket history follow the compatible project and replay contracts.

Plugin review follow-up: integrate and qualify each plugin's completed-model/fit
review factory through the delivered Results action and paired snapshot API.
Retain immutable completion-bound evidence without a configured-recipe reload
or repeated search. Plugin-specific acceptance remains with its own fixtures.
