# Native ONNX features and LightGlue

**OpenCV > Features Framework (features) > Feature matching** includes
**ALIKED ONNX features** (`ml.native_aliked_features`) and **LightGlue ONNX
feature matcher** (`ml.native_lightglue`). They run through OpenCV 5 without
PyTorch, Kornia or the Python LightGlue package. Existing PyTorch blocks and
projects keep their IDs and settings.

## Models and extraction

Choose an explicit local ONNX file, optionally pin its SHA-256, and configure a
model byte budget. Each activation reads and hashes a bounded private snapshot.
New native objects import that snapshot through a temporary file; changing the
original during import cannot replace the checked bytes. Native objects are
cached by content/settings, cleared on processing exceptions, reset and close,
and omitted from projects. Save As rebases ordinary model paths; external files
are not silently copied or downloaded. Models are trusted executable native
inputs; byte/row/work limits do not bound their internal allocations or isolate
native faults in a separate process.

The extractor requires the native ALIKED heads `keypoints` (normalized XY),
`descriptors` (128 columns) and `scores` (probabilities). Width/height select the
stretched RGB network grid and must agree with the export. It uses the New CPU
engine and native per-row L2 normalization by default. Gray, tagged RGB/BGR and
alpha images are interpreted correctly; uint16 uses full-scale conversion and
float32 samples must be in [0,1]. Only the working image is converted to uint8;
the original image, encoding, depth, ticket and spatial metadata are returned.

Native ALIKED ignores its mask argument, so these blocks explicitly filter
**source feature centers after inference**. Mask and Region intersect; alpha can
exclude fully transparent centers. The CNN still sees surrounding pixels.
Minimum score/count and raw-feature/pixel/network/model budgets are explicit;
rows are never silently truncated. Source row indices remain available. Turning
normalization off emits `native_aliked_raw_v1`, deliberately distinct from the
unit-descriptor matcher contract.

## Matching and geometry

Connect the current image, its Query features, and Reference features. Choose the
descriptor family explicitly: ALIKED/DISK (128) or external SuperPoint (256).
Features from existing compatible extractors/packers can be used. Family, shape,
finite unit descriptors, bounded coordinates, known ticket/recipe/frame context,
current image size and maximum feature/attention-pair budgets are checked before
native import. A reference can come from a different acquisition ticket. These
family interfaces do not qualify every export or a real SuperPoint extractor.

The ONNX contract is `kpts0`, `kpts1`, `desc0`, `desc1` to `matches0` (int64 index
pairs) and `mscores0` (float confidence). The native Query/Train order is the
application's Reference/Current query order. Every invocation sets fresh XY and
image-size context. OpenCV normalizes each axis as `2*xy/[width,height]-1`, which
must match the export's training convention. Context is cleared after matching;
native matchers are never shared between blocks or worker copies.

The native factory currently exposes no engine selector. This wrapper requests
the OpenCV CPU backend/target and reports the native automatic engine. Accelerator
options are intentionally absent because the installed New engine warns that
target selection is unsupported. No GPU execution or silent GPU fallback is
claimed. Upstream source contracts are linked in [SOURCES](<../examples/opencv tools/native_lightglue/SOURCES.md>).

Candidates returned at native threshold zero have scores `1-DMatch.distance`.
The block applies its inclusive score floor and retains rejected pairs/scores in
Report. Export-internal pruning and native-discarded invalid indices cannot be
recovered this way; Report is not an unpruned attention matrix. Indices and scores
must be finite, in range and one-to-one. Empty features produce a normal failed
inspection without loading a model. A bad input/import/native error emits no
current result and follows the common exception/diagnostic framework.

RANSAC/LMedS/RHO, reprojection tolerance, confidence, iterations, minimum matches,
inliers and ratio control the planar homography. A passing fit emits an
image-plane pose and Reference-to-Query transform. Failure withholds both, while
matches, scores and geometric inlier flags remain inspectable. Green/red markers
are drawable result graphics for downstream overlays/filmstrip; graphics-off
skips these diagnostics. A geometric inlier fraction is not calibrated confidence,
3D pose, occlusion measurement or proof of physical part identity.

## Qualification and examples

[Six self-contained projects](<../examples/opencv tools/native_lightglue/README.md>)
bundle original images and untrained arithmetic ONNX models. Their observations
depend on image pixels, descriptors and point context. An independently known
2.007843-pixel translation, direct native tensor outputs, score rejection, ROI
filtering, empty selections, RGB uint16 and explicit budget errors exercise the
contracts. These fixtures do not establish learned extraction quality.

The separate opt-in, local-only reference uses the exact ALIKED/LightGlue exports
and two images referenced by OpenCV's own tests. On Python 3.14.3/OpenCV 5.0.0
contrib 5.0.0.93 it yields **1,000 features per image, 211 matches and 207 geometric
inliers**. Independent direct ONNX preparation agrees within **1e-4 source
pixels**, **1e-5 descriptor units**, **1e-6 score units** and an identical
homography. The [receipt](<../examples/opencv tools/native_lightglue/qualification.json>)
pins all four external assets. This qualifies this CPU pair and API transport;
the numerical pose reference uses the same correspondences, not independent
scene ground truth. Other backbones, exports, accelerators and industrial accuracy
remain separate work. External model/image bytes are not redistributed.
