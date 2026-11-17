# Pinned image-based text detection

Use `ml.dnn` followed by the existing `ml.text_regions`. The
[PP-OCRv3 examples](<../examples/opencv tools/pinned_text_detection/README.md>)
provide explicitly prepared, hash-pinned Apache-2.0 English weights and known
source polygons. Model loading, immutable snapshots, named inputs/outputs,
receipts and backend selection use the ordinary DNN infrastructure. The
probability map stays observable between inference and decoding.

## Decoder controls

Connect the same acquisition image to inference and text-region decoding, plus
inference's Tensors and Preprocessing outputs. The head must be a finite
float32 `tensor_v1` probability map in [0,1], with HW axes or leading singleton
axes. Multi-image batches, class channels, quantized tensors and logits require
explicit conversion or another decoder. Source/tensor/preprocessing tickets,
known frames/root dimensions and known recipe revisions must agree.

Pixels strictly above **Pixel threshold** start external contours. The mean
probability inside each filled contour must meet **Minimum confidence**.
**Minimum area** and **Minimum side** apply before expansion, in head pixels.
Choose the expansion explicitly:

| Expansion | Geometry in the probability map |
| --- | --- |
| `rectangle_scale` | Multiply both fitted rectangle dimensions by Padding ratio; retains the earlier default |
| `db_unclip` | Offset the fitted rectangle by `area * Unclip ratio / perimeter` on every side |

DB-style expansion is useful for a model's shrunk text maps. Our decoder fits
and expands in the head grid before restoring source coordinates. OpenCV's
native DB wrapper has different contour traversal, integer scaling and
postprocessing order; arbitrary images need not produce identical polygons.
The official [DB model API](https://docs.opencv.org/5.0/main_modules/classcv_1_1dnn_1_1TextDetectionModel__DB.html)
and [implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/model.cpp)
are reference points. A controlled identity-grid comparison verifies strong
polygon overlap with the native wrapper without claiming equivalence everywhere.

Contours are external-only: internal holes do not become text instances. A
bridge in the probability map can join two words; a gap can split a line.
There is no hidden word, line, language or reading-order inference. The limit
retains highest-confidence regions, then returns them ordered by center y,x.
This is a deterministic spatial order, not paragraph reading order.

## Source coordinates and selection

For head coordinate `u`, head width `W`, network width `N`, and inverse
preprocessing edge map with scale `a` and offset `t`, source pixel-center
coordinate is `a * ((u + 0.5) * N / W) + t - 0.5`; likewise for y. The inverse
maps must be finite, positive, mutually inverse resize/letterbox maps. This
corrects the earlier decoder's half-pixel offset when source/network scales
differed. Image crop/pose metadata remains on the source grid for later consumers.

Optional Mask and Region inputs intersect on the original image grid. A
detected region's center must lie within their selection; inference sees the
full image unless its own selection is separately configured. Border policy
either keeps expanded corners outside the image or discards that whole region.
It does not clamp corners or invent a clipped rotated rectangle.

The decoder returns the original image unchanged, `ocr_regions_v1` corner/
center/confidence records with image size and context, a region count, and a
report. Source color format, depth, ticket, frame and auxiliary payload are
preserved. Recognition remains a separate `ml.recognize_text` step; the detector
does not emit recognized characters or assess OCR correctness.

## Judgments, resource bounds and graphics

Minimum/Maximum count apply to retained regions; maximum zero disables that
limit. Minimum zero retains the earlier permissive empty-result behavior.
Failure reasons distinguish count failure and truncation. **Fail on truncation**
defaults on: an accepted region dropped by Maximum regions cannot silently
produce a passing inspection. If disabled, the report still records truncation
and the pre-limit count. Numerical/current outputs remain available on ordinary
failures. Malformed maps, incompatible metadata and resource violations use
the executor's normal contained error path, with no new output event.

Map samples, source pixels, contour candidates and returned regions are bounded
with explicit controls. Map axes are at most 4096; source axes at most 16384.
Too many contours rejects the invocation before contour scoring rather than
silently scanning a prefix. These guards bound declared data/work, not native
model workspace, scheduler copies or whole-process memory. Inputs are read-only
within the block; graphics allocate their own review image.

Draw overlay controls the optional raster. Global/per-block graphics settings
also apply, so Lean runs avoid raster and drawable generation. Debug outlines
use aggregate pass/fail color. For the final image, connect the original-image
output and hidden Result graphics to Overlay result graphics, then Filmstrip
image. Outline color represents detector judgment, not OCR validity.

## Qualification

The pinned network is PP-OCRv3 English from OpenCV Zoo revision
`48b10fa505aeb51fbd127fc60a5c660040e823b7`. Exact weights, input/output names,
normalization, padding and CPU settings are recorded in the examples' source
notice and model contract. Opening/validation never fetches weights. Preparation
checks bytes, preserves differing existing bundles and is idempotent offline.
Changing sealed preprocessing/execution settings requires an explicit new export.

The eight original MIT projects cover three labels, known 12-degree rotation,
ROI selection, joined/separate text, and blank/count/truncation failures. Known
glyph polygons are independent of the network. Both Classic and New CPU paths
are compared; the saved baseline is explicitly Classic. The unit fixtures also
cover exact source-grid equations, unequal resize/letterbox scales, DB distance,
threshold boundaries, contour bridges, metadata conflicts, budgets and recovery.
This qualifies the described software workflow. Natural-image accuracy,
application-specific word grouping, accelerator performance and physical
inspection/OCR acceptance remain separate work.
