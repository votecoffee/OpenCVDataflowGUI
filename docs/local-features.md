# External SuperPoint features and LightGlue

**Custom > ML > Feature matching** includes **Pack SuperPoint feature tensors**
(`ml.pack_superpoint_features`) and **LightGlue SuperPoint feature matcher**
(`ml.lightglue_superpoint`). These accept locally exported feature heads and use
the matching learned LightGlue checkpoint. The existing SIFT blocks retain their
IDs, settings, RootSIFT descriptors and reference-image workflow.

**SuperPoint ONNX features** (`ml.superpoint_onnx_features`) runs trained
dense detector and descriptor heads locally using OpenCV 5. Both extraction
paths emit `lightglue_superpoint_v1` feature records. They can feed the same
matcher.

## Trained SuperPoint v6 extraction

Prepare the independently trained checkpoint by Rémi Pautrat and Paul-Edouard
Sarlin with Python that has `torch` and `onnx` installed. The
[upstream release](https://github.com/rpautrat/SuperPoint/blob/1411bbd68c50163555d39c1b26e9e046ebd48f27/README.md)
explicitly identifies this v6 checkpoint as MIT-licensed. Its
[MIT notice](third-party/superpoint-v6-MIT.txt) accompanies the export. The
[original Magic Leap detector release](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE)
has distinct terms. The example uses the independently trained MIT checkpoint
and a separately prepared [Apache-2.0 LightGlue matcher](https://github.com/cvg/LightGlue/blob/main/LICENSE).

```powershell
.\.venv\Scripts\python scripts/prepare_superpoint_v6.py
```

The script checks the source SHA-256, loads weights only, and exports a single
ONNX file with its license and receipt under ignored `.local/models`. It also
accepts `--checkpoint` for an existing copy of the pinned checkpoint. The
built-in extractor requires only OpenCV at run time. Graph execution reads
local files. The example graphs pin the exported SHA-256 and bound the byte
count before import. The application reuses the native network for the same
digest, including across input sizes.

The block expects one normalized grayscale `[1,1,H,W]` input and two named
float32 output heads. `logits` is `[1,65,H/8,W/8]` with 64 row-major offsets
and a no-keypoint channel. `descriptors` is `[1,256,H/8,W/8]`. The block
softmaxes over all 65 channels, reconstructs the pixel heatmap, suppresses
nearby peaks, and samples an L2-normalized descriptor field with bilinear
interpolation. The CPU path is qualified for the v6 head layout.

Gray or color source pixels may be uint8, uint16 or normalized float32. The
block converts to normalized grayscale and downsizes proportionally when
needed. Right/bottom padding brings each axis to a multiple of eight and at
least 32 pixels. Peak scores use a strict lower threshold. An optional border
is measured from the unpadded content. Coordinates map back to source pixel
centers before region, mask and alpha filtering. The CNN still sees the full
image. The strongest eligible points are retained up to the feature cap, with
row-major order used to break score ties. Raw indices identify flattened
padded-grid pixels. The output image retains its original pixels and format.

The [four saved v6 projects](<../examples/custom tools/superpoint_v6/README.md>)
qualify CPU extraction on original synthetic board images, a blank-image
count failure, source-coordinate filtering after resize and padding, and a
known planar shift through LightGlue. Dense heads, keypoint coordinates, and
descriptor samples were checked against the upstream PyTorch implementation.
Keypoint locations agreed exactly on all three images. The largest score
difference was 0.0000035 and the largest descriptor component difference
was 0.00000072. The controlled geometry test covers one board and lighting
change.

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

This prepares the Apache-2.0 LightGlue matcher in ignored `.local/models`,
records its source, size and SHA-256, and preserves existing files. Prepare
the trained v6 detector separately when using the ONNX extraction path.

The [seven controls](<../examples/custom tools/local_features/README.md>) use
the same +8/+6 textured image pair as the SIFT regression. Original prescribed
keypoints/descriptors qualify packing, matching and geometry without a detector.
The comparison project also runs SIFT extraction on that pair. The separate
trained v6 projects provide the learned detector evidence.
