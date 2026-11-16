# Seeded MIL box tracking

`sequence.track_mil` wraps OpenCV 5 `TrackerMIL` in the shared stateful sequence
framework. Find **Track object box (MIL)** under **OpenCV → Video Analysis (video) → Object tracking**.
Use it when a detector or a user can supply an initial rectangle and a fixed-size
local tracker is appropriate. See the [recorded examples](<../examples/opencv tools/mil_box_tracking/README.md>).

## Inputs and results

Image accepts uint8 gray, RGB/BGR or RGBA/BGRA, at most 2.1 million pixels and
32766 pixels per axis. Processing uses canonical grayscale; original image data
is not changed. OpenCV's implementation uses one integral-image channel, so
converting explicitly avoids accidentally tracking only blue on a color image.

Seed is `[x,y,width,height]` or `{"bounds":[...]}` in image pixel coordinates.
The rectangle must fit inside the image and be at least 8×8. Seed priority is:
wired seed, configured **Initial bounds**, then nonempty mask/region bounds.
A seed from a different declared coordinate frame is rejected. Wired seeds are
used only when a model needs initialization, not on every accepted update.

Mask and editable rectangle/circle/polygon regions can supply seed bounds.
**Selection mode = center** also rejects updates whose box center falls outside
the selection; **seed_only** permits subsequent movement outside it. These are
wrapper gates, not a native MIL training mask. Training/search still sees the
full grayscale frame.

Results include `bounds`, center `points`, `pose`, `tracked_region`, `ready`,
activation/learning counts and `status`. Pose angle zero follows image axes;
MIL does not measure part rotation. The box size stays fixed. MIL returns no
confidence or probability: `score` stays unset and
`status.native_confidence_available` is false. Contrast is a diagnostic, not a
confidence estimate. Ticket, arbitrary payload and spatial metadata follow the
existing sequence framework.

## Readiness, loss and memory

Initialization reports **initialized / awaiting_update** and withholds primary
results. A successful update can emit only after **Warmup** learning activations
(default 1). The examples use 3 to withhold an observed inaccurate first native
update; warmup is not an accuracy guarantee. `status.last_accepted_bounds` is a
diagnostic seed/last accepted rectangle and can be stale while primary bounds
and pose are withheld.

Native failure, invalid/out-of-image bounds, low contrast, excessive center jump
or a disallowed center report **lost**, fail judgment and a structured `mil_…`
reason. That activation emits status/readiness, but no primary box or pose.
After **Max misses**, the model expires and requires a new wired seed or reset;
the configured initial rectangle is not silently reused. A new model receives
a new track ID. A low-contrast initial seed reports **seed_required** without
creating a model.

Contexts, reset, gap expiry, stream limits and state budgets use
[sequence memory](sequence-memory.md). Native models survive editor worker
handoffs and are released on reset/close. They are not serialized in JSON/XML;
reopening a saved project starts fresh. Parameter, image geometry, encoding or
spatial-context changes start an appropriate new context. Native exceptions are
contained by block execution, clear the affected stream and produce diagnostic
errors without crashing the graph.

## Parameters and resource guards

All seven fields exposed by `TrackerMIL.Params` are available:

| Block field | OpenCV field | Default |
| --- | --- | ---: |
| init_positive_radius | samplerInitInRadius | 3 |
| init_negative_samples | samplerInitMaxNegNum | 65 |
| search_radius | samplerSearchWinSize | 25 |
| track_positive_radius | samplerTrackInRadius | 4 |
| track_positive_samples | samplerTrackMaxPosNum | 100000 |
| track_negative_samples | samplerTrackMaxNegNum | 65 |
| features | featureSetNumFeatures | 250 |

Native SDK policy determines their effect; this OpenCV implementation uses its
initialization sampling modes during updates. Forwarding a control does not
mean each control independently changes every update. Search radius must exceed
both positive radii. Fields have range validation/tooltips, and selection modes
use dropdowns. See the [parameter reference](block-parameters.md).

**Maximum work** bounds a conservative feature/candidate product before native
model creation: features times search-window area, positive-radius area and
negative sample caps. **Max bytes MB** reserves estimated integral-image,
response and feature-model storage before allocation. These are preflight
estimates, not a hard native allocator, RSS limit or execution deadline. Large
images/features/search windows remain expensive; track a bounded field of view
and measure the actual workload.

## Graphics and evidence

Drawable box/text results participate in normal result overlays and filmstrip
aggregation. Lean/off can suppress unrequested drawings. Raster **Overlay** is
hidden by default and generated only when explicitly exposed; it preserves the
source's color view. Turning **Draw overlay** off disables both forms.

Focused tests cover native translation in gray and RGB, all parameter forwarding,
validation, loss/expiry/reseed, memory/work guards, native exceptions, region and
ticket contracts, XML/reopen, worker handoff and offscreen editor dropdowns.
Five original MIT saved projects replay seven annotated frames, template
reacquisition and deliberate wrong-region/work-limit failures in Debug and Lean.
The examples reduce search radius to 8 for their two-pixel motion, and accepted
centers after warmup are checked within 8 pixels. An eight-replay local probe
with the native default radius 25 observed up to 22.81 pixels of drift; radius 8
observed up to 5 pixels on the same fixture. This is fixture-specific tuning,
not a general recommendation to reduce the search window. Native MIL is
approximate and stochastic; these software checks do not
qualify occlusion, scale change, scene identity or robot-picking reliability.

Upstream: [video tracking API](https://docs.opencv.org/5.0/main_modules/video_track.html),
[MIL implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/video/src/tracking/tracker_mil.cpp).
