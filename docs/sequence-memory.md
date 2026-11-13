# Sequence memory

Sequence blocks retain bounded, private state between accepted image activations.
Use them for temporal operations or learning a reference image. Native flow and
background models are under **OpenCV > Video Analysis (video)**. Built-in
sequence workflows and reference memory are under **Custom > Sequence**.

Open [sequence memory](<../examples/custom tools/sequence_memory/sequence_memory.json>).
Run continuously, or press F5 several times. The source initially holds a textured
patch still and then moves it. The example compares background subtraction,
Farneback flow, point tracking and a learned reference. No camera or download is needed.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/sequence_memory/sequence_memory.json" --repeat 8
```

## Readiness and events

Causal sequence blocks expose `ready`, `activations` and `status`. The status includes eligible
learning activations, required count, stream/context/generation, current and previous
tickets, and algorithm diagnostics. Before readiness, primary image/measurement
outputs emit **no events**; control/status outputs still emit. Nodes show Waiting
and an activation message; diagnostic graphics may be viewed during warmup.

The warmup count is a minimum, not proof of model quality. Background subtraction
can additionally require consecutive stable frames. Tracking requires enough tracks
of sufficient age, and can become unready again when features disappear. Readiness
is separate from inspection pass/fail and never assigns a final ticket judgment.

Causal primary outputs carry the current input ticket and `sequence` provenance;
older frames are model history, not new events for their old tickets. A frozen reference
records its learning ticket in status rather than replacing the current image ticket.
Existing input `reject_failed` policies run before learning: rejected inputs neither
advance state nor replace retained previews. To make final decisions, gate on readiness
or consume primary outputs; a status event during warmup does not mean inspection finished.

## Continuous centered stabilization

**Stabilize continuous stream** accepts one ordered camera stream and delays
each image by up to Radius frames to smooth with past and future observations.
Before that look-ahead is available, image outputs wait; Ready, Buffered and
Status report the current state. At most 2 × radius + 1 source images are
retained, subject to the stored-state budget. Native work buffers, result
snapshots and GUI copies use additional memory. Latency is measured in future
frame arrivals; a paused live source needs a Flush pulse to release its tail.

Each delayed image, coverage mask, transform and report keeps the original
ticket, timestamp and upstream outcome. The report identifies its smoothing
window, segment, observed/smoothed pose and coverage. A coverage rejection
emits the affected ticket's failed report and withholds that image. It does not
assign the failure to the newer frame that triggered processing.

Finite Run all/CLI playback flushes the final tail automatically, including
chained buffered blocks, ToolBlocks and background acquisition queues.
Place the background acquisition queue before buffered inspection.
Flush also works as an explicit boolean pulse for live sources. Reset is
different: it discards pending frames and history. Settings, grid, source,
recipe, recording-loop and configured gap changes start a new segment after
flushing the old tail. Rejected tracking starts a separate coordinate segment,
so scene cuts never share a smoothing window.

The corrected image uses a segment reference frame. Transform records retain
the original input frame and both correction directions. Use the transform
to map between raw and corrected coordinates; coverage excludes unsupported
border samples. The block preserves gray/RGB/BGR/alpha encoding and uint8,
uint16, float32 or float64 samples; motion fitting uses an 8-bit working image.

Try the [144-frame example](<../examples/custom tools/continuous_camera_stabilization/README.md>).
One block handles one ordered stream; use separate blocks for independent
cameras.

### Buffered plugin lifecycle

Plugins with delayed output can subclass `BufferedStreamBlock` from
`opencv_dataflow.buffered_stream`. Implement the abstract `pending_output`
and `pending_tickets` properties, private bounded runtime snapshots, and `process`. Declare
`stream_inputs` for the primary image/set input names. These inputs should be
optional with `required_when_connected=True` to allow an empty drain call.

The scheduler sets `end_of_input` only on the final call when image ancestors
have ended. If pending output remains, finite playback continues with
`process({})` until it is drained. ToolBlocks forward the signal and preserve
per-packet emissions. Independent scalar controls do not keep an image stream
open. Preserve original packet payloads in `Result.emissions`, including a
valid per-packet outcome when one activation emits different judgments.
Pending ticket identities keep production completion and end-to-end timing
open until delayed work is emitted.
The base reports emitted tickets rather than attributing a delayed judgment
to the current trigger. Reset/close and `copy_runtime_from` remain explicit
plugin responsibilities.

## Included blocks

| Block | State and outputs | Readiness / controls |
| --- | --- | --- |
| Running background subtraction | Running mean/variance; foreground mask, background, masked foreground | Warmup, optional stability; absolute/variance threshold, rate, deviations, variance floor, foreground-learning policy, learn input |
| Debounce foreground components | Current confirmed/rejected masks, current-grid blob collection, count and observation diagnostics | Separate global warmup and consecutive component hits; threshold, connectivity, area, displacement, area-ratio, optional box IoU and component limit |
| Farneback optical flow | Previous frame and optional previous flow; dense float32 XY vectors, magnitude, angle, visualization | Needs previous frame and warmup; scale, levels, window, iterations, polynomial neighborhood/sigma, Gaussian window, warm-start flow, visualization scale |
| DIS optical flow | Same dense flow, magnitude, angle and visualization outputs as Farneback | Needs previous frame and warmup. Select Ultrafast, Fast or Medium, then tune patch search, pyramid scales, propagation, refinement and warm start. |
| Dense-grid Lucas–Kanade optical flow | Tracks a regular grid and interpolates dense vectors with a support mask | Needs previous frame and warmup. Tune grid spacing, LK search and termination, trackable texture threshold, and minimum interpolation support. |
| Track points (Lucas-Kanade) | Previous grayscale image, feature positions, IDs and ages; track collection and overlay | Warmup plus minimum mature tracks; corner detector controls, LK window/levels/termination, patch error or minimum eigenvalue, forward/backward tolerance, reseeding and age limits |
| Stabilize camera motion | Previous grayscale frame, background feature tracks and a smoothed global trajectory. Corrected image, coverage mask and forward/inverse transform. | Needs a usable fit after the anchor frame. Translation or similarity, smoothing, motion and coverage gates, re-anchoring on loss of fit. |
| Image memory / trained reference | First/latest/mean image; reference, absolute difference, current image | Eligible learning count or validated trained seed; learn input, freeze after warmup, strict/shape-only reference context |

Flow vectors are displacements from the previous image grid to the current image in
local pixels. Angles are in degrees. Point records include previous/current positions,
displacement, persistent generation-scoped ID, age and error; their coordinates are
local to the current input grid. Spatial/pyramid payloads retain the mapping to the
root image. These are point tracks, not object identities or trained object trackers.
The dense-flow value has a compact report description and a `vectors` NumPy array for
custom consumers. See the [OpenCV tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
for the native Farneback and Lucas-Kanade parameter semantics.

DIS uses OpenCV's Dense Inverse Search. Level zero is the full-resolution grid.
Automatic coarsest scale is bounded by the image and patch size. The preset's
finest scale can reduce for a smaller input. An explicit incompatible scale
or patch stride is reported as invalid. Masks and regions zero the reported
vectors outside the selection, while warm start uses the full preceding field.
See the [OpenCV 5 DIS API](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html)
and the [PCB photograph replay](<../examples/opencv tools/pcb_motion/README.md>).

**Dense-grid Lucas–Kanade optical flow** interpolates tracked grid nodes in
the previous image. Only valid tracks contribute displacement. The **Support
mask** indicates where their combined bilinear weight meets the configured
minimum. Pixels below that minimum have zero vectors. Use the support mask
when zero motion and missing estimates need different treatment.

Farneback, DIS and dense-grid Lucas–Kanade can compute a second flow in the
reverse direction. Set **Consistency tolerance** above zero, then expose
**Consistency mask** and **Consistency error** in the sidebar as needed.
The check samples reverse flow at the forward endpoint and measures the
return error in pixels. The error is NaN for samples outside the selection,
outside the destination, or without grid support. The mask marks evaluable
pixels whose error meets the tolerance. Status reports selected, evaluable,
and consistent counts. The check adds a reverse solve and keeps the forward
vectors unchanged. Agreement alone does not establish accuracy in ambiguous
texture or occluded areas.

All blocks accept optional circle/rectangle/polygon regions or masks. Background and
flow retain full-image neighborhood/learning context and restrict reported foreground
or vectors to the selection. Reference learning stores the full image and masks its
difference output. Point tracking seeds and retains points inside the selection.
Background supports uint8/uint16/float32; flow and point tracking require uint8
(color converts to grayscale). Reference memory preserves supported image depth and encoding.

The [camera stabilization example](<../examples/custom tools/camera_stabilization/README.md>)
replays measured shifts and rotations. The background selection controls
feature centers. The correction applies to the whole frame. Corrected pixels
and their coverage mask share a new coordinate frame for each anchor.
The transform reports the mapping to and from the current raw input.
For sequence plugins that change the grid, a sequence step can provide an
output frame per primary port. The port retains its current ticket and sequence
status and uses the supplied frame in place of the input spatial and pyramid grid.

## Foreground component debounce

**Custom > Sequence > Foreground masks > Debounce foreground components** consumes
single-channel uint8, uint16, float32 or float64 foreground images. Samples
strictly greater than **Foreground threshold** become foreground. For raw
MOG2/KNN masks, threshold 127 excludes 127-valued shadow labels. Threshold units
are the input's raw samples; normalized floating masks typically use 0.5.
Connect the existing background block's binary mask with threshold zero instead
when its shadow policy has already been applied.

A component must meet the pixel-area limits after mask/region selection. It
inherits confirmation only if it and exactly one previous component are each
other's sole candidates under the centroid-distance, smaller/larger area-ratio
and optional bounding-box IoU gates. Zero IoU disables overlap gating so a
moving part can cross disjoint pixel locations. An empty frame, absent match
or ambiguous association restarts that component's hit count. There is no
prediction, gap bridging or emission of historical pixels. Crowding, touching
parts, merges/splits and motion beyond the gates can prevent confirmation;
this is conditional temporal filtering rather than general identity tracking.

The **Mask** contains only currently confirmed pixels. **Rejected mask** contains
current thresholded pixels removed by area or temporal checks. **Components**
is a normal blob collection with shape metrics, current-grid geometry and
confirmation metadata. **Observations** reports eligible current candidates,
hit counts, conditional generation-scoped association IDs and gate outcomes.
Numbers are current-grid pixels; spatial/pyramid payloads retain rescaling
context without silently multiplying coordinates. Current ticket, acquisition
start time and companion payload data remain attached. Old tickets are model
provenance, never new output identities.

Default global warmup is three eligible activations and the default component
requirement is three consecutive hits. Warmup emits status but no primary data.
Hits count consecutive eligible activations in the same stream/context. Inputs
rejected by the common failure policy or skipped upstream do not advance or
clear history. Ticket values are not interpreted as frame counters. Use the
reset input or a time-gap limit when an upstream discontinuity must break evidence.
After warmup, an empty mask/count zero is an ordinary successful filtering result,
not an inspection failure. Green result-graphics objects mark confirmed boxes;
amber marks pending boxes, annotated with capped hit counts. Global/per-block
graphics-off settings skip creating these objects. No graphics are emitted
during warmup, when primary observation data is withheld.

Only current component summaries and capped streak counters are retained, with
configurable stream, component and byte limits. Exceeding a bound fails safely
and clears runtime evidence; native labeling/intermediate images need extra
memory beyond the retained-state budget. With the default reset policy, changing
the rasterized selection starts a fresh context, as do grid, recipe, parameter
and time-gap boundaries. The separate-context policy can resume a cached context,
as described below. Worker/editor copies own independent summary data. Runtime
memory is transient; saved JSON/XML stores controls and layout.

Try the [original nine-frame replay](<../examples/custom tools/foreground_debounce/README.md>),
which checks exact expected masks for disjoint motion, glints, shadow labels,
disappearance and reconfirmation. Native measurements use
[OpenCV connected components and shape descriptors](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html).

## Streams, resets and execution

By default memory is keyed by image ticket source, or `default` for un-ticketed images.
Supply a stable string/integer `stream` to choose another key. Ticket numbers alone
do not reset memory. Different image dimensions/depth/encoding, spatial/pyramid grid,
recipe revision, effective parameters or trained reference start a new context.
`context_change=reset` discards the old source context; `separate` caches contexts.
Configured stream-count and stored-byte bounds either error or evict the oldest context.
Byte limits cover stored arrays/metadata, not temporary calculations or UI snapshots.

An optional maximum ingress gap resets stale context (default 1000 ms for flow/tracking;
zero disables it). This measures local monotonic processing time, not camera timestamps.
Set it to zero when stepping slowly with F5. A boolean `reset=true` clears that stream
before processing the incoming frame. **Reset sequence memory** in the inspector clears
the selected block; **File > Reset flow state** clears all runtime memory. Errors clear
memory to prevent reuse of partially modified state.

State survives F5 worker handoff and Stop/Resume through independent copies. Layout-only
movement preserves it. Structural/configuration changes reset runtime; save/open starts
fresh. Live histories are never serialized in projects. Place sequence processing after
an acquisition queue so only accepted FIFO frames advance it. Sequence blocks are rejected
in the queue's acquisition prefix and inside fresh parallel per-part jobs. Ordinary
nested ToolBlocks retain memory; each instance owns its own streams.

## Trained references and recipes

Run an image into Image memory, then use **Capture reference from last input** in its
inspector. This captures the input pixels, not the accumulated mean. Save the project
to retain it, or use the reference JSON import/export buttons. Runtime reset preserves
the trained asset; **Clear trained reference** removes it. A validated seed is ready
on the first activation, with learning warmup bypassed. Shape, depth and encoding must
match; strict mode also checks stored grid/recipe context. `shape_only` explicitly
permits reuse across that context boundary.

Assets are lossless NPY pixel data inside checksummed JSON, never pickle. They are
limited to 8 MiB of pixels and retain only spatial/pyramid/recipe provenance, without
old tickets or outcomes. The existing recipe size/revision limits also apply.

In **Recipes > Manage recipes…**, copy the selected Image memory's trained asset into a
recipe model slot (or import its reference JSON). Use **Recipe field** with
`path=models.reference` and connect its value to Image memory's `reference` input.
Feed the corresponding recipe-tagged image to its image input. An explicit recipe-field
assignment binds the asset to that selected recipe revision while retaining its grid;
changing the asset/revision resets temporal context. The wired reference takes precedence
over a stored reference. This supports swapping trained images along with pattern models.

## OpenCV MOG2 and KNN foreground masks

**MOG2 background subtraction** and **KNN background subtraction** maintain
independent models per stream. During warmup the primary outputs wait while
status reports learning activations. Once ready, the blocks emit the raw
mask, binary foreground and shadow masks, a background estimate and an image
containing the visible foreground. The main mask excludes shadows by default.

Use a mask or region to restrict the reported foreground and shadow pixels.
Learning still uses the full frame. Wire a separate `known_foreground` mask
to mark pixels OpenCV should treat as foreground and skip while learning.
MOG2 accepts uint8 or float32 intensity values in [0, 255]. KNN accepts uint8.
Set MOG2's Learn input to false to freeze adaptation. KNN uses automatic or
positive learning rates because the pinned native build still adapted at zero.

The editor retains each live model across Run Once and Stop/Resume. Reset
flow state, a stream or context reset, or reopening a project starts a
fresh model. Independent execution graphs also start fresh. The sequence
budget covers retained native CPU model arrays. Transient images and
previews use additional memory.

Run the [native background example](<../examples/opencv tools/native_background/native_background.json>)
with `opencv-dataflow-run "examples/opencv tools/native_background/native_background.json" --repeat 25`.
The synthetic target stays still during warmup, then moves.
See the [OpenCV background subtraction tutorial](https://docs.opencv.org/5.0/tutorials/others/background_subtraction.html).

## OpenCV GSoC and LSBP foreground masks

Find both sample-based models under **OpenCV → Background and Foreground
Segmentation (bgsegm)**. GSoC adapts a bank of pixel colors and can suppress
repeated foreground toggles. LSBP also matches a local binary texture
descriptor. Each stream keeps its own model. Start with an empty scene and
let the configured warmup finish before using the binary **Mask**,
**Foreground**, and **Background** outputs.

Both accept uint8 or normalized Float32 images. The foreground keeps the
input format. The native background estimate is uint8 gray or BGR.
Mask and Region restrict the foreground outputs while the full frame
updates the model. GSoC's color learning rate controls only its matched
color and error averages. LSBP adapts through its own update parameters.

The default camera-motion mode assumes a fixed camera. The Lucas–Kanade
option needs sufficiently textured frames with successful feature tracking
and at least 25 pixels on each axis. Model memory grows with the number of
pixels and samples. OpenCV retains two sample buffers for each stream.
Reset flow state to replay a training sequence from the first frame.

[OpenCV bgsegm reference](https://docs.opencv.org/5.0/extra_modules/bgsegm.html)

## Track a seeded object box

**Track object box (CSRT)** and **Track object box (KCF)** track one target
per stream in a sequence of uint8 images. Provide a wired `seed` box, a
configured initial rectangle, or a mask or region from which to take the
bounding rectangle. The box is `[x, y, width, height]` in the input image
grid. A wired object containing `bounds`, such as the location output
from **Match template**, also works. CSRT can use the selected pixels as
its initial segmentation mask. These inputs initialize the tracker and
its later search follows the native tracker.

The first successful update emits the box, center, a center pose and a
rectangular region. The pose angle is zero. Status identifies the track
and reports initialization, tracking, a missed update or the need for a
seed. An optional overlay shows the current box. Contrast and center-jump
limits can reject implausible updates. Misses withhold the box, pose and
region. After the configured number of consecutive misses, use a new
wired seed or reset the sequence to start a new track.

Live models survive Run Once and Stop/Resume in the editor. Reopening the
project initializes a fresh tracker. The state budget uses a conservative
estimate from the search patch and algorithm settings.

Run the [native box tracking example](<../examples/opencv tools/native_box_tracking/native_box_tracking.json>)
with `opencv-dataflow-run "examples/opencv tools/native_box_tracking/native_box_tracking.json" --repeat 8`.
Both blocks follow the same moving synthetic target.
See the [OpenCV tracking tutorial](https://docs.opencv.org/5.0/tutorials_contrib/tracking/tutorial_introduction_to_tracker.html).

## Associate conveyor parts across frames

**Track conveyor parts** associates template or pattern matches, measured
object collections, or pose lists. A distance gate limits associations.
The assignment first maximizes the number of accepted pairs and then
minimizes their total center distance. Each part keeps a track ID across
frames, and a missed detection can remain as a bounded prediction.

Set motion to **encoder** and provide an absolute encoder value sampled
at acquisition, over the encoder port or as an `encoder` field in the
image payload. Ticketed inputs must share the image ticket. Set the signed
image-space displacement per count, optional counter modulus and maximum
step. The rollover mode uses the shortest signed difference and rejects
half-turn ambiguity. Missing values, ticket mismatches and implausible
steps withhold track outputs for that activation and preserve the last
accepted positions and encoder value.

The `tracks` output marks observed, predicted and offscreen positions.
The `poses` output contains only current observations inside the image for
downstream image inspection. The missed-frame limit and offscreen margin
bound retention. Encoder prediction assumes constant displacement per count
in the working image grid. Physical encoder capture, timestamp pairing and
world-space calibration are separate workflow steps.

Run the [conveyor tracking example](../examples/applications/conveyor_tracking/conveyor_tracking.json)
with `opencv-dataflow-run examples/applications/conveyor_tracking/conveyor_tracking.json --repeat 15`.
The synthetic source emits two parts and an encoder value on the same
ticket. One detection drops out briefly and one part leaves the camera view.

## Custom sequence blocks and future work

Subclass `SequenceBlock`, declare `data_outputs`, defaults and choices, and implement
`step(inputs, state, mask)` returning `SequenceStep`. Always declare every primary
output, and use `ready`/`learned` to control readiness/counts. State must support deepcopy
and fit the memory budget; override `estimate_state_bytes` for large models. Do not
retain native handles, external resources or workers in this state. Register it through
the existing trusted Python plugin system.

Hardware encoder acquisition, timestamp-aware resampling and explicit
learning of accumulated references remain future work.

## Temporal median background from a batch

Connect **Batch serial entries** to **Temporal median background** and collect
two to 64 aligned views from a fixed camera. The block calculates the median
independently at each pixel and color channel. An even batch averages the
middle pair and rounds integer outputs to the nearest sample. The result
carries the batch ticket and the pixel grid of the earlier middle frame.

The block accepts gray, RGB or BGR images in uint8, uint16 or float32. It
keeps alpha from the earlier middle frame and uses that frame's pixels outside
an optional mask or region. Use a batch with at most 128 MiB of source pixels.
A clear view of a pixel in more than half of the frames can remove a moving
occlusion from the estimate. Persistent objects and camera motion can affect
the result.
