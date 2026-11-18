# Tracking instance masks

`sequence.track_instances` consumes the existing decoded instance-mask collection
and the current image. It assigns temporal IDs using motion, mask overlap and
optional masked appearance. Find it under **OpenCV → Deep Neural Networks → Object
tracking**. Try the [eight replay examples](<../examples/opencv tools/instance_tracking/README.md>).

## Inputs and model identity

Connect Image and Collection from the current segmentation workflow, and expose
the upstream ONNX inference Receipt. Default identity binding requires one current
ONNX call. Artifact/contract/preprocessing identity forms part of the sequence
context. A model/preprocessing change therefore resets or separates streams according
to the ordinary context policy. Receipts associate supplied metadata; this block
does not independently authenticate mask pixels against the original tensor.

Custom mask producers can explicitly choose Configured identity and provide a
model revision. This is a caller assertion. Reset when changing decoder settings
or semantics not covered by that identity. Known image/collection/receipt/timestamp
tickets, recipes and coordinate frames must agree. Collections need zero origin,
the current image dimensions and individual `image_region_v1` masks. Native proposal
row IDs are retained as Source instance ID; they do not drive temporal matching.

An optional mask/rectangle/circle/polygon region intersects observed instance pixels.
This changes the measured visible area and centroid; excluded objects become
unobserved. Original source pixels, encoding and depth are returned unchanged.
High-depth images use a full-scale uint8 working copy for optional histograms.

## Association and uncertainty

The tracker predicts each center with constant velocity. It translates the last
observed mask to that center using nearest-pixel translation, exclusively for
association. This does not estimate mask deformation, rotation, amodal extent or
hidden surface pixels. Current masks come from segmentation.

Candidate gates cover residual center distance, predicted-mask IoU, optional
class equality and appearance distance. Expose the controls needed by the job:
mask/center cost weighting, gray or joint HSV histogram bins, appearance weighting,
Hellinger distance gate and update rate, and velocity update rate. Assignment uses
the shared maximum-cardinality/minimum-cost solver. Scores and association costs
are not calibrated identity confidence.

Withhold is the default ambiguity policy: close local row/column cost alternatives
are withheld, and those observations do not spawn duplicate tracks. The report
lists ambiguous source instance IDs. Best is an explicit alternative that accepts
the solver's choice; it cannot establish the physical identity of indistinguishable
overlapping parts. Local cost margins are a conservative heuristic, not a global
identity probability.

Minimum observations controls confirmation. Missing tracks retain predicted centers,
velocity, age, missed count and last-observed ticket in Report. They expire after
the configured missed frames or acquisition-time deadline. A long acquisition
interval expires existing tracks. IDs increase within one context; generation plus
ID is the run-scoped Track key. Reset/new contexts may restart numeric IDs.

## Outputs and lifecycle

Tracks is an ordinary sortable/filterable box collection whose items retain
independent overlapping masks, current scores and source geometry. Its active
geometry is the current mask's tight bounds; mask centroids/areas are measured
from selected current pixels. Existing **Select instance mask**, configured by ID,
can select a temporal track. The Labels image resolves overlapping current masks
by score and then lower track ID; Overlap count and the collection preserve overlap.

**Only confirmed observed masks are emitted.** Unobserved or ambiguous tracks have
no current output mask. Reports explicitly distinguish observed and predicted
states. Do not treat a predicted center as a current image measurement.

Drawable mask contours and track labels can be aggregated onto the original image.
An optional diagnostic image colors current track IDs. Graphics limits may omit
outlines; the numeric masks/labels and associations remain unchanged. Off/Lean
avoids these diagnostic graphics.

This serial sequence block uses the shared readiness, per-stream context, memory,
reset and state-copy contracts. Save/reopen excludes live tracks. Runtime copies
receive independent state. Failed-input rejection emits nothing and preserves the
last processed image/state. Invalid temporal steps clear the affected stream and
recover with a new generation; early association validation can reject before
changing state. Choose nominal, explicit monotonic milliseconds or video timestamps.
Video tracking requires one consistent grid/clock and a reset at a new sequence.

Budgets cover image pixels, instances/tracks, decoded full-grid masks, mask-IoU
pixel work, region runs and retained sequence state. Capacity/work overflow is an
execution error, never silent observation truncation. No final ticket judgment or
aggregate confidence is assigned by tracking itself.

## Controlled qualification

Four original lossless FFV1 videos contain 38 frames. A 671-byte original ONNX
model extracts current red/blue channel masks; boxes cover the source frame.
Both parts share a class, so temporal association cannot rely on class identity.
The input row order/IDs are independently varied in unit controls.

Crossing, short occlusion, gray appearance and uint16 replays have zero ID switches
and exact visible masks. Expiry intentionally creates one ID switch on re-entry.
Confirmation withholds two initial visible masks; identical transparent-overlap
observations are explicitly withheld, producing two visible-mask misses. No masks
are fabricated through those gaps. The budget example deliberately emits errors.

Independent visible-mask arithmetic checks pixel IoU, current mask counts and ID
histories. Direct native ONNX masks match all 38 frames exactly. These are controlled
CPU software examples, not pretrained video-segmentation accuracy, physical object
identity proof, general occlusion recovery or production acceptance.
