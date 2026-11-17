# External SuperPoint features and LightGlue

**Custom > ML > Feature matching** includes **Pack SuperPoint feature tensors**
(`ml.pack_superpoint_features`) and **LightGlue SuperPoint feature matcher**
(`ml.lightglue_superpoint`). These accept locally exported feature heads and use
the matching learned LightGlue checkpoint. The existing SIFT blocks retain their
IDs, settings, RootSIFT descriptors and reference-image workflow.

This is a **partial SuperPoint integration**: external tensor conversion and
learned matching are qualified; a real SuperPoint detector/extractor fixture is
still open. No SuperPoint detector, detector weights or inference source is
bundled or downloaded. The
[upstream licensing note](https://github.com/cvg/LightGlue#license) distinguishes
Apache-2.0 LightGlue matcher weights from the separate restrictive SuperPoint
detector/code/weight terms. A local model's use and redistribution require the
appropriate rights. The bundled controls are original untrained arithmetic and
prescribed descriptors; they do not establish learned detection accuracy.

## Tensor packing

Connect an image and its matching inference tensors/preprocessing. Select:

- XY: `[K,2]`, optionally batch `[1,K,2]`.
- Descriptors: `[K,256]` or explicitly channels-first `[256,K]`, optionally batch 1.
- Scores: `[K]` or `[1,K]` probabilities. An explicitly blank output name uses 1
  for every score, rather than guessing a tensor.

All heads must be finite float32 `tensor_v1` records, with corresponding rows.
Zero-row heads are valid. The raw-row limit applies before selection, without
silently truncating. L2 normalization is on by default; zero descriptors reject.
With normalization off, supplied rows must already have unit norm within 0.001.

Declare network pixels or normalized fractions multiplied by the full network
width/height, and pixel centers or pixel edges. Center coordinates are converted
to edges by adding 0.5, mapped through the exact resize/letterbox inverse, then
converted to source centers by subtracting 0.5. The output uses integer-origin
source pixel centers. Tensor axes and coordinate conventions are never inferred
from model filenames.

Inclusive **Minimum score**, source bounds, Mask and Region filter corresponding
rows together. A connected missing selection prevents execution. The retained
`indices` and feature `raw_indices` refer to original raw rows. `minimum_count`
controls a normal inspection judgment, while empty/filtered records remain
available for diagnostics. `report.detector_executed` is always false.

`features` is `lightglue_superpoint_v1`, carrying `image_size`, points,
descriptors, scores, raw indices and `coordinate_context`. The primary image
output retains its depth, encoding and pixels. Ticket, recipe, spatial/pyramid
and processing-start metadata follow the current image; known context conflicts
reject. Optional feature markers are drawable objects, limited independently
from numeric output. Lean skips them.

## Matching and inspection

Connect packed Query features, Reference features and the current query Image.
Query body/wire metadata must agree with the current image. Reference features
may come from another acquisition ticket; their own body/wire context must agree.
Known frame/ticket/recipe conflicts reject before loading the matcher. The query
feature grid must match its image. A row budget bounds both descriptor sets before
learned attention; source-image pixels are separately limited. These checks bound
declared work, not process memory or hard real-time latency.

Choose a local `superpoint_lightglue.pth` and optionally its SHA-256. Loading uses
PyTorch's weights-only reader and all required matcher parameters; SIFT/ORB records
or incompatible checkpoints reject. File identity/settings changes invalidate
the cache. Reset/close releases the owned model references. Opening or validating
does not import optional ML runtimes, load weights or contact the network.

CPU/CUDA/Auto, layers, attention/mixed precision, adaptive stopping/pruning and
match threshold are exposed. CPU mixed precision rejects. Homography method,
reprojection threshold, confidence/iterations, minimum matches/inliers/inlier
fraction govern one planar registration. Failed geometry emits no current pose
or transform; matched-point diagnostics remain available. This is not repeated
part discovery, metric calibration or a physical-pick qualification.

The pose maps reference pixel (0,0) to the query and supports ordinary
pose-relative regions. Match markers show geometric inliers/outliers in green/red
and can be overlaid on the final filmstrip image. The primary `image` remains
unchanged; `overlay` is a review image. Lean skips optional painting for both the
new matcher and the existing SIFT matcher. The base executor contains exceptions,
records timing/diagnostics and allows the next valid activation to recover.

## Preparation and evidence

Use the optional runtime installation in [ML setup](ml.md), then explicitly run:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --superpoint-matcher
```

This downloads only the Apache-2.0 LightGlue matcher into ignored `.local/models`,
records source/size/SHA-256 and preserves existing files. It does not prepare a
SuperPoint detector. Local-export preparation remains outside graph execution.

The [seven controls](<../examples/custom tools/local_features/README.md>) use
the same +8/+6 textured image pair as the SIFT regression. Original prescribed
keypoints/descriptors qualify packing, matching and geometry without a detector.
The comparison project also runs genuine SIFT extraction on that pair. A real
permitted SuperPoint extractor, its export conventions and detector repeatability
still need independent qualification before S2-15 can become Complete.
