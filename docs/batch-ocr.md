# Ordered OCR over source regions

`ml.recognize_text_batch` rectifies multiple source quadrilaterals, assembles
bounded native NCHW batches, and decodes one explicit CTC output. It reuses
ordinary batched DNN preparation, shape guards, immutable model snapshots,
backend controls and receipts. The shared batch extension normalizes native
output batch axes before creating rows; other image-inference blocks retain
their original axis-zero behavior. The existing `ml.recognize_text` remains
available and now uses the same engine-preserving ONNX snapshot loader.

Try the [eight saved projects](<../examples/opencv tools/ocr_batch/README.md>).
Seven use original MIT arithmetic controls without downloads; one uses an
explicitly prepared unchanged Apache-2.0 CRNN English model on printed labels.

## Source regions and image contract

Connect an image and optional `ocr_regions_v1` data from Text regions from
probability map. Known wire/embedded tickets, frames, root dimensions and recipe
revisions must agree with the source. Region image size, when supplied, must
match. Unannotated regions are explicitly interpreted in the connected image's
pixel coordinates. Records preserve their supplied order, IDs and detector
fields; OCR confidence is a separate field.

Without Regions, the Quadrilaterals parameter supplies ordered source quads.
An empty list uses the whole image or bounding rectangle of the source selection.
Quad corners are cyclic, with the first edge becoming the crop's horizontal
reading direction. The default detector already emits this orientation.
The block does not guess reading direction, rotate upside-down text, or infer
paragraph order. Each crop is warped to the configured Width/Height before
ordinary color/depth normalization. Pixel-center homographies, source ticket,
recipe and auxiliary payload are retained on its internal image packet.

Mask and Region intersect on the original source grid. Their selection filters
quad centers and replaces excluded pixels with Background before rectification.
Centers outside the source/selection become skipped records, preserving their
indices. Partial quads can sample the configured border background; this does
not establish that the whole text was visible. Background uses source intensity
units and must fit integer source depth. The original image, encoding, depth,
ticket and frame are returned unchanged. The eight fixture projects use one
source acquisition each; this block is not a multi-ticket image collector.

## Model, batching and vocabulary

Declare exact native Input shape `[N,C,H,W]`, using `-1` only for dimensions the
export genuinely supports. Batch size controls real rows per call. Variable,
pad-repeat and reject tails retain the normal batch-inference semantics.
Repeated padding rows never become reads. Batch-coupled models can still be
affected by padding; that policy needs model-specific qualification. Empty
selection invokes neither native inference nor model/vocabulary file reads.

Choose TNC or NTC for the native output's time/batch/classes axes. Exactly one
output is selected, normalized to NTC internally and split back into original
region order. Shape/layout mismatches are errors rather than guessed reshapes.
The original arithmetic fixtures execute N=3 and qualified N=1/N=3 tails. The
pinned CRNN export has fixed N=1 and native output [24,1,37]; larger N is rejected
before inference when its declared shape is respected. There is no hidden
serial fallback or claim of measured batching speedup.

Vocabulary excludes blank and lists all other classes in native class order.
Blank index is explicit. Unique multi-character tokens are permitted. Inline
tokens are included in model settings; a UTF-8 token-line file requires an exact
SHA256, is read as one bounded activation snapshot and is declared for Save As.
Output class count must be vocabulary length plus one. These are configured
tokens, not a language model, lexicon or automatic alphabet discovery.

Use Softmax for logits or Probability for already normalized scores. Native
model contracts seal preparation, vocabulary/settings and execution configuration.
Receipts retain the submitted native shapes, artifact identity and import path;
per-read call/row indices are available when receipts are enabled. Budgets cover
regions, source/crop pixels, prepared/retained tensor elements and time steps.
They do not bound process RSS, native model workspace or scheduler snapshots.

## CTC scores and judgments

Greedy decoding selects each time step's highest-scoring class, merges adjacent
equal classes, then removes blank runs. Equal classes separated by blank remain
separate tokens. Ties choose the lowest class index. Character evidence records
each emitted token, class ID, half-open step span and peak probability in its run.

Read confidence is the geometric mean of those emitted run peaks. All-blank
paths have no character confidence. This score is neither the summed CTC string
posterior nor a calibrated probability that the text is correct. It is not the
detector's region confidence. Scores need qualification against actual labeled
application images before serving as an acceptance limit.

Every original region has a `reads` row with index, source region, raw text,
confidence, character evidence and `read`, `blank`, `low_confidence` or `skipped`
status. Blank/low-confidence rows have explicit reason codes. Count and Text
include only accepted reads. Minimum reads sets the required accepted count.
Aggregate All additionally requires every selected row to pass; Minimum reads
allows failed siblings deliberately. Skipped rows do not fail by themselves.

Ordinary inspection failures still emit current reads/report/image and reasons.
Invalid settings, metadata conflicts, unsupported shapes and native exceptions
use the executor's contained error path, producing no new output event. A valid
next invocation can recover. Common failed-input policies, timing, logs and
ticket lifecycle remain owned by the base executor. The single source ticket
gets the aggregate result; rows expose their local verdicts explicitly.

Optional drawable source quads and recognized labels use per-read pass/fail
colors. Global/per-block graphics settings suppress them in Lean without
changing numbers. Connect Image and Result graphics to Overlay result graphics
and then Filmstrip image for the final visualization.

## Qualification and limits

Independent scalar path/confidence references cover repeats, blanks, ties,
nonzero blank IDs, multi-character tokens and extreme logits. Actual native
TNC/NTC models exercise dynamic batches, padding, order, source homographies,
metadata, current failures, limits and recovery. RGB/uint16 source pixels and
format survive grayscale inference. The pinned CRNN reads three independent
glyph labels and agrees with the separate single-region recognizer; its weights
are unchanged and its fixed-batch limitation is explicit.

Alphabet/lexicon-constrained prefix decoding is available in the separate
[constrained OCR block](constrained-ocr.md). Training and OCR accuracy over
natural industrial image sets remain separate roadmap work. The controlled examples
qualify this software/data contract, not optical or machine acceptance.
