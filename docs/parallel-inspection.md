# Parallel inspection of detected parts

**Control flow > Parallel inspection > Inspect each pose in parallel** runs a
reusable inspection ToolBlock independently for each detected part. Connect the
source `image` and the pattern matcher's `matches` to `features`. A measured
blob/line/contour collection, a list of `image_pose_v1` poses, or a single pose is
also accepted. Pattern poses retain matcher offsets and full geometric transforms.
For measured collections, `feature_angle` selects fixed axes or the measured axis.

Each invocation accepts up to 64 parts and uses 1–32 workers (default 4). Results
are emitted once after the part jobs settle, in the original feature order,
regardless of completion order. This adds concurrency across parts; Split/Join
remain useful for independent branches within a job.

## Define the per-part inspection

Select the block and click **Edit inspection graph**, or double-click its node.
The default ToolBlock receives public `image` and `pose` inputs. It creates a
radius-5 local circle and checks for exactly one foreground blob inside it. Edit
the region with **Draw / edit region…** or replace the steps with your own tools.
The default graph uses ordinary OpenCV processing and exposes data, count and
overlay outputs. Use **Up one level** to return to the parent job.

The terminal names are configurable through `image_terminal`, `pose_terminal` and
`context_terminal`. Optional `context` data can provide a shared recipe or other
per-invocation settings. All required public inputs must map to these terminals.
Use **Load inspection ToolBlock…** to replace the inspection with an existing
definition, or **Export inspection ToolBlock…** to reuse it elsewhere. Imports are
validated before replacing the current definition. Load plugins explicitly before
opening definitions that use custom types; projects do not execute plugin imports.

Each part owns a fresh graph, block instances and private input snapshots. Buffered
flow state does not carry from one part to another. Waiting/unfinished inspection
graphs produce an error outcome rather than a passing part. Inner Split workers
execute serially inside each job so the outer worker count controls concurrency;
nested Inspect-each blocks inside an inspection are rejected. Custom plugin module
globals remain shared Python state and need appropriate synchronization.

Jobs retain the source coordinate frame and parent acquisition ticket. Each part
also gets a distinct scoped child ticket, with its original index and parent
ticket in the `inspection` payload. When both image and feature tickets are known,
they must agree; poses must also map into the image's coordinate frame. A bad pose,
incompatible template or excessive collection size rejects the invocation before
starting any part jobs.

## Aggregation and early stopping

`stop_after_passes` and `stop_after_failures` default to zero (disabled). A positive
value stops launching new parts once that many passing or unsuccessful inspections
have completed. Inspection errors count toward the failure stop threshold, while
remaining separately identifiable in `error_count`.

Thresholds use completion order. At most `workers` jobs are submitted at once.
Already-running jobs are asked to stop between steps. A native OpenCV call or
custom function already running can finish; resulting pass/fail counts may exceed
the threshold by up to the remaining in-flight jobs. The operator waits for these
jobs to settle and releases their managed resources before returning the aggregate.
No managed inspection continues after the output is emitted. Use one worker for
exact N-result stopping. External writes already performed are not rolled back;
place aggregate result writers after the block when a single final write is wanted.

Every original part has a record with one of `pass`, `fail`, `error`, `cancelled`
or `not_started`. Uninspected parts are never labeled passing. `complete` means
every part returned a pass/fail/error outcome; check `error_count` separately for
valid execution. `stop_reason` identifies completed work, pass/failure limit,
cancellation, output budget or an empty collection.

Choose the aggregate acceptance rule:

| Policy | Passing condition |
| --- | --- |
| all_pass | All parts inspected and passing |
| any_pass | At least one passing part; partial coverage is allowed |
| min_passes | At least `required_passes`; partial coverage is allowed |
| max_failures | Complete coverage and no more than `allowed_failures` |

An error outcome fails every aggregate policy. External cancellation also fails
the aggregate. Empty input follows `empty_policy` (fail by default). To accept
after the first three passing inspections, set `stop_after_passes=3`,
`result_policy=min_passes`, and `required_passes=3`. To reject quickly on the first
failure, use `stop_after_failures=1` with the default all-pass policy.

## Results and retained images

The block exposes:

- `results`: JSON `parallel_inspection_v1`, containing every part's pose, feature
  ID, status, measurements, score, timing and error, plus aggregate counts and coverage.
- `group`: completed pass/fail/error records as a flow group, with each part's ticket.
- `images`: a flow group of each completed inspection's configured `image_output`
  (default `overlay`), when present. Use Unbatch/Select group entry to consume them.
- `pass_count`, `fail_count`, `error_count`, `complete`, and a full-image outcome overlay.

The aggregate score is passing parts divided by the original total, so early
acceptance can have a low score despite meeting its configured pass threshold.
Image outputs are described by shape/depth in JSON records and receipts; raw image
buffers travel through `images`. Set `image_output` empty to disable image retention.
A missing named image output simply produces no collected image for that part.

`preview_index` chooses whose internal graph snapshots are retained for inspection
and region editing. If that part did not complete, a completed part is used instead;
the actual index is recorded in results and headless receipts. This preview is a
captured frame, not shared live worker objects. The embedded inspection definition
and layout persist in the project, including inside other ToolBlocks; runtime
snapshots and outcomes do not enter project JSON.

`max_input_mb` bounds source image bytes times simultaneous workers (default 256
MiB), and `max_result_mb` bounds retained result data/images (default 128 MiB).
These are data budgets, not hard process-heap limits; graph intermediates, private
snapshots and a retained graph preview use additional memory. Input-budget excess
fails before work starts. Result-budget excess records an error and stops further
submission. Reduce workers/image size, disable per-part image retention or increase
the configured budget as appropriate.

## Try 25 matched parts

Open [parallel part inspection](<../examples/flow tools/parallel_parts/parallel_parts.json>), then press F5. The graph trains on fixed geometry,
locates 25 repetitions, and inspects the small central spot independently at each
matched pose. Expected result: 25 completed inspections, 25 passes and no failures.
The geometric matcher now supports up to 256 matches; this operator accepts up to
64 parts per invocation.

Set **25 synthetic parts > fail_every** to 5 and apply to remove every fifth spot.
The next run still finds all 25 geometries, then reports 20 passing and 5 failing
inspections. Restore zero for the all-pass sample. No camera, downloaded asset or
output writer is required. These behaviors are also covered by automated tests.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/flow tools/parallel_parts/parallel_parts.json" --repeat 3
```
