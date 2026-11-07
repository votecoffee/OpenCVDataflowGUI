# Sequence memory

Named projects in this guide refer to optional local examples. See the
[public example gallery](../examples/public/README.md) for bundled projects.

Sequence blocks retain bounded, private state between accepted image activations.
Use them for temporal operations or learning a reference image. They appear under
**Sequence > Temporal processing / Image memory** in the toolbox.

Open sequence memory.
Run continuously, or press F5 several times. The source initially holds a textured
patch still and then moves it. The example compares background subtraction,
Farneback flow, point tracking and a learned reference. No camera or download is needed.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/flow tools/sequence_memory/sequence_memory.json" --repeat 8
```

## Readiness and events

Each block exposes `ready`, `activations` and `status`. The status includes eligible
learning activations, required count, stream/context/generation, current and previous
tickets, and algorithm diagnostics. Before readiness, primary image/measurement
outputs emit **no events**; control/status outputs still emit. Nodes show Waiting
and an activation message; diagnostic graphics may be viewed during warmup.

The warmup count is a minimum, not proof of model quality. Background subtraction
can additionally require consecutive stable frames. Tracking requires enough tracks
of sufficient age, and can become unready again when features disappear. Readiness
is separate from inspection pass/fail and never assigns a final ticket judgment.

Downstream primary outputs carry the current input ticket and `sequence` provenance;
older frames are model history, not new events for their old tickets. A frozen reference
records its learning ticket in status rather than replacing the current image ticket.
Existing input `reject_failed` policies run before learning: rejected inputs neither
advance state nor replace retained previews. To make final decisions, gate on readiness
or consume primary outputs; a status event during warmup does not mean inspection finished.

## Included blocks

| Block | State and outputs | Readiness / controls |
| --- | --- | --- |
| Running background subtraction | Running mean/variance; foreground mask, background, masked foreground | Warmup, optional stability; absolute/variance threshold, rate, deviations, variance floor, foreground-learning policy, learn input |
| Farneback optical flow | Previous frame and optional previous flow; dense float32 XY vectors, magnitude, angle, visualization | Needs previous frame and warmup; scale, levels, window, iterations, polynomial neighborhood/sigma, Gaussian window, warm-start flow, visualization scale |
| DIS optical flow | Same dense flow, magnitude, angle and visualization outputs as Farneback | Needs previous frame and warmup. Select Ultrafast, Fast or Medium, then tune patch search, pyramid scales, propagation, refinement and warm start. |
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
and the [public PCB replay](../examples/public/pcb_motion/README.md).

All blocks accept optional circle/rectangle/polygon regions or masks. Background and
flow retain full-image neighborhood/learning context and restrict reported foreground
or vectors to the selection. Reference learning stores the full image and masks its
difference output. Point tracking seeds and retains points inside the selection.
Background supports uint8/uint16/float32; flow and point tracking require uint8
(color converts to grayscale). Reference memory preserves supported image depth and encoding.

The camera stabilization example
replays measured shifts and rotations. The background selection controls
feature centers. The correction applies to the whole frame. Corrected pixels
and their coverage mask share a new coordinate frame for each anchor.
The transform reports the mapping to and from the current raw input.
For sequence plugins that change the grid, a sequence step can provide an
output frame per primary port. The port retains its current ticket and sequence
status and uses the supplied frame in place of the input spatial and pyramid grid.

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

Run the native background example
with `opencv-dataflow-run "examples/opencv tools/native_background/native_background.json" --repeat 25`.
The synthetic target stays still during warmup, then moves.
See the [OpenCV background subtraction tutorial](https://docs.opencv.org/5.0/tutorials/others/background_subtraction.html).

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

Run the native box tracking example
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

Run the conveyor tracking example
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
