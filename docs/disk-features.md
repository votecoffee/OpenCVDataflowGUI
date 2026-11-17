# DISK features and LightGlue

**Custom > ML > Feature matching** includes **DISK learned features**
(`ml.disk_features`) and **LightGlue DISK feature matcher** (`ml.lightglue_disk`).
They run real local-weight extraction and matching, using the optional PyTorch,
Kornia and LightGlue environment described in [ML setup](ml.md).

Prepare the pinned depth, epipolar and matching checkpoints explicitly:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --disk
```

The script records source/size/SHA-256, checks the release digests and preserves
existing files. Weights stay in ignored `.local/models`; saved projects refer to
local paths. Opening or validating a project does not import ML runtimes, download
weights or run inference. The three models have separate hashes. The two detector
variants share the supported 128-descriptor architecture; choose their local file.

## Extraction and reference capture

Image supports uint8/uint16 or normalized float32 gray/color. A private normalized
RGB working copy feeds the network; the primary Image output preserves the
source's pixels, encoding and depth. Physical depth planes and float HDR values
outside 0..1 reject. Tagged RGB/BGR/RGBA/BGRA are interpreted by the shared image
converters. Alpha can exclude zero-alpha keypoint centers or be ignored.

**Resize long edge** is zero by default. A positive value only downsizes, using
the selected OpenCV interpolation and rounded dimensions. Cubic overshoot is
clipped to the normalized input range. **Pad if not divisible** lets Kornia
zero-pad the right/bottom to multiples of 16 and exclude padding from its output
maps. Padded network axes must be at least 32 pixels. Source and padded-network
pixel limits are checked before allocation/inference.

**Max features**, odd **NMS window size** and **Detection threshold** expose the
native detector controls. Thresholding is strictly greater on raw detector
scores; these scores are not probabilities and may exceed 1 or be negative.
The qualified backend is Kornia 0.9.0, including its native suppression/tie
behavior. Results are not promised identical across other backend revisions.

Mask, Region and alpha **filter centers after native whole-image top-N**. They
do not mask the CNN's receptive field or reduce its dense-image work. Excluded
features can consume top-N slots; raise the cap if appropriate. A connected
selection with no event gates execution. `raw_indices` retains each selected
row's identity in the native result.

Coordinates map back to integer-origin source pixel centers:
`source_xy = (network_xy + 0.5) / actual_scale - 0.5`. Descriptors are L2-normalized
128-value rows. `lightglue_disk_v1` contains points, raw scores, descriptors,
image size, row indices, source coordinate context and extractor digest/backend
version/device. `report` records source/network scaling, padding, raw/retained
counts and selection policy. `minimum_count` gives a normal pass/fail judgment;
zero permits an empty capture. `train(inputs)` captures portable reference
features using the same processing, without fine-tuning network weights. Supply
the captured image's actual metadata when its identity matters; training does
not inherit an earlier execution's ticket.

Feature markers are drawable result objects. Their optional count limit does
not truncate numeric results. Lean skips graphics. Image tickets, recipe,
spatial/pyramid and processing-start context survive; known selection conflicts
reject. Each instance owns a lazily loaded model cache, keyed by local file
identity, digest setting and device. Close/reset releases its model references.

## Matching, poses and results

Connect Query features, Reference features and the current query Image. The
shared [local-feature contract](local-features.md#matching-and-inspection)
checks query body/wire/image context, reference identity, source size, finite unit
descriptors and work budgets. DISK, SIFT and SuperPoint schemas/checkpoints remain
distinct even where descriptor dimensions coincide. A local matching checkpoint
is loaded with PyTorch's weights-only reader; an optional SHA-256 can pin it.

CPU/CUDA/Auto, learned layers, attention, mixed precision, adaptive stopping/
pruning and match confidence remain configurable. Homography method, threshold,
confidence/iterations and match/inlier criteria govern one planar registration.
Failed geometry/count criteria emit no pose or transform, while current match
diagnostics remain available. Pose origin is reference pixel (0,0); use ordinary
pose offsets and pose-relative regions for inspection.

The primary Image remains unchanged. Overlay is a review image; drawable match
markers color geometric inliers/outliers green/red and can feed the final
filmstrip image. Lean skips optional painting. Exceptions are contained by the
base executor with block diagnostics/timing; the next valid activation can
recover. Failed-input admission policies prevent work and preserve the last
processed image. Missing connected selections also prevent unmasked fallback.

## Qualified scope

[Ten projects](<../examples/custom tools/disk_matching/README.md>) cover known
translations, source resizing, ROI filtering, both detector variants, RGB16,
empty/count failures, local inspection regions and SIFT comparison. Two separate
attributed VisA PCB photographs provide independent approximate mounting-hole
checkpoints. Native Kornia parity, descriptor shapes/norms/raw scores, source
ownership, JSON/XML/Save As, exact regeneration, model integrity/cache/error
recovery and Debug/Lean filmstrip behavior are checked.

CPU qualification uses Python 3.14.3, OpenCV 5.0.0, Kornia 0.9.0 and
PyTorch 2.15.0+cpu. CUDA controls are exposed but accelerator accuracy/performance
remain unqualified. These image-pair controls qualify a planar baseline, not
repeated-part discovery, arbitrary model exports or metric/robot localization.

Upstream references: [DISK](https://github.com/cvlab-epfl/disk),
[Kornia DISK API](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.DISK),
[LightGlue](https://github.com/cvg/LightGlue).
Models/libraries retain their Apache-2.0 terms; see the example's attribution.
