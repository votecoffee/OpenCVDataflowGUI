# Learned box tracking

`sequence.track_vit` wraps native OpenCV `TrackerVit` using the existing seeded
tracker/sequence lifecycle. It accepts a local ONNX or a sealed model contract.
The [nine projects](<../examples/opencv tools/learned_tracking/README.md>) include
the unchanged small Apache ViTTrack model, source/license records and known-motion
annotations. They run offline without preparing another dependency.

## Initialization, updates and loss

Provide an initial `[x,y,width,height]`, a wired current-grid Seed, or a mask/region
whose bounding rectangle supplies the seed. Wired Seed takes precedence when a
model is needed. A seed does not replace an active model every frame. Initialization
captures the native template, emits status and withholds current box/pose. There
is no valid native score until an update has run. Warmup counts eligible learning
activations; the examples use three, so the first two activations withhold primary
geometry even when the first update is accepted.

Native score threshold, contrast, center jump, image bounds and optional selection
center gates decide acceptance. A rejected update emits no current Bounds, Points,
Pose, Tracked region or Overlay. Current Image, Status, Ready and any newly observed
Tracking score remain available. Loss is an ordinary failed result, distinct from
an execution exception. After Max misses, the native model is discarded. Expired
tracks remain failed while waiting for a usable new seed; configured initial bounds
are not silently reused as reacquisition. A current wired detector seed or Reset
can initialize a new track. Track IDs distinguish reacquisitions.

The score is the native window-weighted response exposed by `getTrackingScore`,
not calibrated correctness, visibility or probability. It can remain high when
a different guard rejects the box. Nonfinite/out-of-range native scores are
contained exceptions, not clipped into valid measurements. Native internal
hypotheses can change before a later contrast/jump/selection guard rejects them;
the wrapper retains only the last accepted bounds in Status and does not claim
to roll back native search state. Expiry/reseeding gives a deliberate recovery path.

## Inputs, geometry and configuration

Accepted images are uint8 gray, BGR/RGB or alpha formats. Native tracking receives
canonical BGR; the current original image is returned unchanged. All native Params
controls have equivalents: Model path, Mean, Std, Tracking score threshold,
backend and target. A preloaded network also gives explicit Classic/New import and
fusion settings. Mean/std defaults mirror the installed native tracker, which owns
template/search preparation and decoding. Fixed 128x128 template, 256x256 search and
native crop factors are described in the preparation record rather than exposed as
editable settings that the API does not support.

Mask and region define seed rectangle support and, in Center mode, the allowed
accepted center. They do not remove pixels from the learned template. Seed only
mode uses them for initialization without gating later centers. Known seed/selection
frame conflicts reject processing. Connected mask/region inputs must provide a
value. Bounds and tracked regions use the current source grid; the region also
stores its coordinate context. Pose origin is the box center, with image-axis
orientation zero; the tracker does not measure part rotation. Ticket, recipe,
spatial and pyramid context follow the current image, not the initialization image.

The existing sequence framework owns stream separation, warm-up, input-gap resets,
settings/grid/recipe resets, bounded stream counts and overflow policy. Changing
configuration restarts the relevant context. Run-profile changes alone preserve
tracking state. Independent graph copies and reopened projects start fresh;
opaque native handles are not serialized or deep-copied. The editor explicitly
shares its idle execution/display state through the established sequence API.

## Model identity, resource guards and diagnostics

Every eligible activation freezes and hashes bounded model bytes before using the
sequence context. A changed model identity resets the context; an invalid changed
model cannot silently continue with the old tracker. Sealed contracts additionally
verify exact configuration and declared native preparation. Receipts record model
hash/revision, environment, import transport, requested/selected backend, explicit
fallback, current update count, score and whether a native model is still active.
An expired receipt describes the executed model without claiming it is active.
Summary/Off are available; native internal tensors are not exposed.

Classic uses the shared immutable temporary-file snapshot loader to preserve its
engine selection. CPU avoids GPU target enumeration, including on a background
worker. Experimental non-CPU selection must be explicit; unavailable-target CPU
fallback occurs only when configured. Native exceptions do not trigger silent
import retries. The examples qualify both Classic and New on CPU; they do not
qualify accelerators or prove per-layer placement.

Source pixels, native search crop size and conservative model/workspace allowances
are bounded. Sequence byte limits are estimates, not a native allocator or hard
process-memory cap. Masks may reduce accepted geometry without reducing model
work. Exceptions are handled by the base executor and reset sequence state without
a new output event. Close/reset releases native handles; runtime state snapshots
exclude those handles.

Drawable status/box graphics follow the current image. Raster Overlay is hidden
initially and computed only when exposed and graphics are enabled. Lean or graphics
off skips result drawings/rasters without changing tracking acceptance. Last accepted
bounds remain explicitly historical in Status during loss; they are not emitted
as a current pose or box.

## Evidence and limits

The native model is included unchanged: 714,726 bytes, with exact publisher hash
and Apache notice. A public CC BY 4.0 T-LESS image is resized and artificially
translated by (2*i,i). The independently declared 43x43 ROI is evaluated against
known translated boxes. Direct native CPU tracking has minimum IoU **0.93284** and
maximum center error **1.11804 pixels** on six updates. A blank frame has native
score **0.09838**, below the configured 0.2 threshold. Initialization/update parity,
loss/expiry, explicit detector reseeding, selection, score/jump/resource alarms,
color formats, model integrity, stream/reset/copy behavior, Debug/Lean, portable
Save As/XML and cold Run All teardown are checked.

These are artificial annotated clips, not recorded part motion, object segmentation,
physical dimensions or general occlusion qualification. Other scenes, models,
lighting and accelerators require their own evaluation. See the
[source record](<../examples/opencv tools/learned_tracking/SOURCES.md>) and independent
`benchmark.py` / `benchmark.json` for the exact model/image identities and results.
