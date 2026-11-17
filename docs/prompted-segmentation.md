# Prompted image segmentation

Three blocks under **OpenCV → Deep Neural Networks (dnn) → Prompted segmentation**
provide the qualified EfficientSAM-Ti April 2025 interface:

1. `ml.segmentation_prompts` binds source-image points and an optional box.
2. `ml.efficientsam` runs the pinned fused encoder/prompt/mask ONNX model.
3. `ml.decode_prompt_masks` selects, maps and judges a current candidate.

The model bundle pins inference settings/preprocessing. Decoder thresholds and
selection remain independently editable. There is no training, runtime download,
split encoder export or reusable image-embedding cache. Other SAM families and
accelerators need separate qualification.

## Prompts and native input

Point records use source pixel-center coordinates `[x,y]`, labels **1 foreground**
or **0 background**, and an optional `[x,y,width,height]` box in continuous source
pixel-edge coordinates. Points are distinct and inside `[0,width-1] × [0,height-1]`.
A positive box stays inside the image and consumes two of six total slots. At
least one foreground point or box is required. Connected point/label/box inputs
override their saved fields only on a current event. A box prompts the model;
it does not clip the returned object.

`segmentation_prompts_v1` retains image size and ticket/spatial/pyramid/recipe
context. Mismatched known frames, tickets, root dimensions or recipes reject.
Wired prompts retain their bound identity; static saved coordinates bind anew
to each image. Arbitrary user blocks can produce the same explicit record.

The qualified model consumes a uint8 image: canonical BGR/gray/RGB interpretation,
linear stretch resize **while uint8**, RGB float32 conversion and `/255`, then
NCHW `[1,3,1024,1024]`. Gray is explicitly expanded; alpha images require an
upstream explicit conversion. Prompts use direct `x*1024/source_width` and
`y*1024/source_height`, matching this export's convention, without an added half
pixel. Point labels and box corners 2/3 occupy `[1,1,6]`; coordinates are
`[1,1,6,2]`. Unused coordinates are zero with label **-1**, distinct from background.
All slots rebind every activation; stale points cannot survive a shorter request.

Outputs are three `[1024,1024]` float32 **logit images**, a three-value predicted
quality record, prompts and inference info. Planes travel as an ordered FlowGroup,
avoiding millions of Python scalar records. Their child frames use correct
pixel-center resize maps back to the original source; the outer group identifies
the source. Select group entry can display any plane without another model call.
Candidate buffers remain stable across later forwards. Source pixels/encoding
are unchanged.

## Source binding and decoding

Info binds the exact source pixel/format signature, prompt revision, model bytes
and inference configuration. Candidate child payloads and quality records carry
the matching request/inference revisions. Decoder checks these alongside current
ticket/frame/recipe metadata. Mixed requests, changed source pixels, invalid
shapes/nonfinite values or wrong grids reject with no current output event.
These are pipeline correlation checks, not signed attestations from arbitrary
Python plugins or proof of physical camera synchronization.

Candidates are ranked by descending raw predicted quality, with native index as
tie-breaker. Index mode chooses an explicit candidate. Point membership is tested
on each full source-grid mask at nearest source centers before output selection:

- **Report only:** retain quality/index selection and report contradictions.
- **Prefer consistent:** prefer the highest-quality candidate satisfying points;
  fall back to the highest-quality candidate if none does.
- **Require consistent:** apply the same preference and fail if the selected
  candidate contradicts any foreground/background point.

Predicted quality is a model IoU estimate, **not measured IoU or a calibrated
probability**. It is retained raw; the block-color score is clamped to `[0,1]`.
Quality, point consistency and selected pixel/fraction limits give distinct
ordinary failure reasons while retaining current geometry. A high quality can
still accompany a failed prompt constraint.

Default resampling thresholds native logits with `>= 0`, then nearest-resizes the
binary mask. Linear-logit mode resizes raw logits before thresholding; boundaries
can differ. Optional source-grid logits always use linear interpolation and can
therefore differ from a mask-nearest boundary. Mask/region inputs restrict the
**output** after full-image inference; they do not crop inference or alter the
raw prompt-membership checks. Fractions use the selected support as denominator.

Mask output is uint8. Optional Region is an exact scanline-run region bound to
the source frame. Disable Region/Logits outputs to avoid their optional work and
events. Drawable mask contours respect graphics/Lean settings and independent
contour/vertex limits; limiting graphics never changes numerical masks. Filmstrip
overlays can combine decoded outlines with foreground/background/box graphics.

## Bounds, persistence and evidence

Source axes are limited to 16,384 with a configurable pixel budget (default 4M).
Native output budget must fit three 1024² planes and three qualities. Region-run
overflow rejects, rather than truncating. Model workspace and graph snapshots
are additional memory; these bounds do not guarantee native peak RSS or latency.
The fused model re-encodes each image/request. No interactive cache speedup is
claimed. Shared ONNX loading, immutable model snapshots, receipts, cache/reset,
target selection and exception handling remain in use.

[Eight runnable examples](<../examples/opencv tools/prompted_segmentation/README.md>)
use the pinned Apache 2.0 OpenCV Zoo model, an unchanged CC0 photograph and original
shape fixtures. Independent native preprocessing/prompt/mask references, exact
rectangle truth, normal quality/prompt/selection failures, changed-request
rejection, Debug/Lean, frame/pixel guards, numeric budgets, recovery, warm rebinding,
XML, portable bundles, offscreen controls and a cold GUI worker qualify this CPU
baseline. Rectangle Dice is 0.977 for the box/nearest case and 1.0 for the
point/linear case on this fixture, without general image/industrial accuracy claims.

Primary references: [pinned model/license](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/image_segmentation_efficientsam),
[native preprocessing/postprocessing](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/image_segmentation_efficientsam/efficientSAM.py).
The app passes label-0 background points into the model; it does not reproduce
the demo's separate background-only post-filter.
