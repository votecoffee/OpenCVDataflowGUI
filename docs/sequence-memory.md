# Sequence memory

Sequence blocks retain bounded, private state between accepted image activations.
Use them for temporal operations or learning a reference image. They appear under
**Sequence > Temporal processing / Image memory** in the toolbox.

Open **File > Load sequence memory example** (`examples/sequence_memory.json`).
Run continuously, or press F5 several times. The source initially holds a textured
patch still and then moves it. The example compares background subtraction,
Farneback flow, point tracking and a learned reference. No camera or download is needed.

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/sequence_memory.json --repeat 8
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
| Track points (Lucas-Kanade) | Previous grayscale image, feature positions, IDs and ages; track collection and overlay | Warmup plus minimum mature tracks; corner detector controls, LK window/levels/termination, patch error or minimum eigenvalue, forward/backward tolerance, reseeding and age limits |
| Image memory / trained reference | First/latest/mean image; reference, absolute difference, current image | Eligible learning count or validated trained seed; learn input, freeze after warmup, strict/shape-only reference context |

Flow vectors are displacements from the previous image grid to the current image in
local pixels. Angles are in degrees. Point records include previous/current positions,
displacement, persistent generation-scoped ID, age and error; their coordinates are
local to the current input grid. Spatial/pyramid payloads retain the mapping to the
root image. These are point tracks, not object identities or trained object trackers.
The dense-flow value has a compact report description and a `vectors` NumPy array for
custom consumers. See the [OpenCV tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
for the native Farneback and Lucas-Kanade parameter semantics.

All blocks accept optional circle/rectangle/polygon regions or masks. Background and
flow retain full-image neighborhood/learning context and restrict reported foreground
or vectors to the selection. Reference learning stores the full image and masks its
difference output. Point tracking seeds and retains points inside the selection.
Background supports uint8/uint16/float32; flow and point tracking require uint8
(color converts to grayscale). Reference memory preserves supported image depth and encoding.

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

## Custom sequence blocks and future work

Subclass `SequenceBlock`, declare `data_outputs`, defaults and choices, and implement
`step(inputs, state, mask)` returning `SequenceStep`. Always declare every primary
output, and use `ready`/`learned` to control readiness/counts. State must support deepcopy
and fit the memory budget; override `estimate_state_bytes` for large models. Do not
retain native handles, external resources or workers in this state. Register it through
the existing trusted Python plugin system.

MOG2/KNN native subtractors, CSRT/KCF object trackers, explicit learning of accumulated
references and timestamp-aware sequence resampling remain future work. These need a
deliberate ownership/snapshot contract before introducing opaque native state.
