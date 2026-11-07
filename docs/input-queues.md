# Queued input images

Named projects in this guide refer to optional local examples. See the
[public example gallery](../examples/public/README.md) for bundled projects.

Place **Control flow > Acquisition queues > Queue input images** between the image
source and the rest of the job. Connect its `image` output to downstream image
processing, matching and per-part inspection. Normal graphs already avoid
overlapping whole-graph runs; this block additionally retains waiting images and
supports acquisition while the previous frame is being inspected.

Each queue releases **one frame per consumer iteration**, FIFO by enqueue order.
The frame stays in flight until that graph iteration finishes; another frame is
not released concurrently. Parallel branches or per-pose inspections within the
current frame are still allowed. Downstream buffered operators can retain work
across iterations, so a completed queue iteration is not proof that every later
batch/join has drained.

## Modes

| Mode | Behavior |
| --- | --- |
| buffer_only (default) | Capture during ordinary runs, retain bursts, and release one image per Run once/iteration |
| background_acquisition | During Continuous or the headless runner, acquire upstream on a dedicated thread while downstream processes one frame at a time |

Buffer-only mode supports multiple queues and queues inside reusable ToolBlocks.
It can drain a burst when the upstream branch subsequently emits nothing. Waiting
captures drain before another acquisition when their upstream snapshots are
available. The image's ticket, custom payload, encoding and spatial/pyramid
provenance are preserved. Pyramid inputs use the inspector's image-level selector,
largest by default.

Background acquisition currently supports **one queue in the root graph**. Other
queues must use buffer_only. Place it before buffered flow operators or parallel
inspection blocks; those are not supported in the acquisition prefix. Stateless
preprocessing and acquisition ToolBlocks can be upstream. All ancestors required
to produce the queue's image run on the producer side. Camera resources are opened,
read and closed on their owning acquisition thread.

The consumer uses captured upstream results for the selected frame, including
ancillary outputs such as source information and reference images. This keeps
metadata paired with the image being inspected while acquisition advances.
Upstream node previews show the capture being processed, not a live view of the
newest waiting frame. Nested upstream ToolBlock previews are also restored from
the captured snapshot. Sources/branches outside that acquisition prefix still
execute in the consumer graph.

Route per-frame image processing through the queue. Bypassing it with another
direct source-image wire bypasses the gate. For burst sources, each image needs
its own ticket to correlate per-frame upstream emissions; incompatible bursts
are rejected. Changing ancillary streams should be carried with their frame, not
broadcast independently into a queued inspection.

## Capacity and overflow

Configure `capacity` (default 32 waiting frames) and `max_bytes_mb` (default 128
MiB). Both limits apply. Byte accounting includes captured upstream inputs,
outputs and graphics, so the effective frame count can be lower than capacity.
The in-flight frame, an acquisition already being made, graph intermediates and
UI snapshots require additional memory; these are queue-data limits, not hard
process-heap limits. A capture larger than the entire byte limit is an error.

| Overflow | Behavior |
| --- | --- |
| block (default) | Pause background acquisition when full; wait for consumer space without dropping accepted frames |
| drop_oldest | Discard the oldest waiting frame; the in-flight frame is protected |
| drop_newest | Reject the incoming frame and retain existing waiting frames |
| error | Stop acquisition with an explicit queue-full error; accepted waiting frames remain available |

Synchronous execution cannot wait for its own downstream consumer. In buffer_only
mode, block/error reject an incoming burst that would exceed storage after one
release, before mutating the previously accepted queue. The rejected burst is not
automatically retried. Drop policies are explicit and counted; there is no implicit
"latest image wins" behavior. Drops can create numeric ticket gaps, so use a
post-queue manifest if another FIFO reorder stage needs explicit expectations.

`capture_interval_ms` optionally paces background acquisition. Zero uses source
pacing, such as a blocking camera read. Application backpressure cannot guarantee
that a camera backend or external hardware retains physical frames it never
delivers to this queue. Trigger/PLC handshaking remains the acquisition source's
responsibility.

## Status, stop and lifecycle

Outputs include `queued`, `busy`, `dropped` and a `stats` record with accepted,
released, completed, failed, rejected, byte usage and high-water counters. `busy`
is one while the consumer iteration owns an image, then zero after it finishes.
`waiting` means no image was released; downstream image branches remain inactive.
An ordinary negative inspection contributes to the queue's failed-iteration count.
An erroneous or cancelled in-flight iteration is not automatically replayed,
because some downstream operations may already have performed side effects.

When acquisition fails, accepted frames can drain before the error is reported.
Stop requests cancel cooperatively. An ongoing native read/inspection must return
before its thread is joined; no managed acquisition thread is left running after
Stop finishes. An acquisition that finishes after Stop may not enter the queue.
Only accepted waiting captures are retained.

The GUI preserves waiting captures across Stop and later Run once/Continuous.
Run once can drain an existing captured frame without reacquiring its source.
Pending images and counters are runtime state and are **not saved in project
JSON**. Opening a project starts empty. Applying configuration, rewiring, Reset
flow state or closing the job clears queued work. Save image data separately when
it must survive application exit. Layout-only moves preserve the queue.

The headless runner processes the requested number of consumer iterations;
background mode is enabled by the block configuration. Prefetch can leave waiting
frames after the requested count; the receipt reports that backlog, and process
exit releases the remaining in-memory queue. Exit status indicates processing
outcomes, not that all captured frames have been inspected. Place result writers
downstream of the queue when only inspected frames should cause writes.

## Example

Open queued input and start
**Continuous**. The synthetic source fills a four-frame queue while one frame's
25 matched parts are inspected in parallel. Watch the queue counters, then Stop.
Run once resumes the oldest accepted waiting capture. No camera or output writer
is required. Queueing, metadata coherence, Stop/Resume and overflow behavior are
covered by automated tests.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/flow tools/queued_input/queued_input.json" --repeat 3
```
