# Block activity events and timelines

Debug and Balanced run profiles capture complete activity spans. Lean omits routine
successful spans while retaining warning/error events and failed finishes. The view
identifies the current capture profile; see [run profiles](run-profiles.md).

Every running job has a common `graph.activity` collector. Root blocks, nested ToolBlocks,
camera acquisition workers and parallel per-part jobs publish to the same thread-safe,
bounded log. It is independent of ticket pass/fail history: an input arrival or exception
is an activity event, not an image emission or a final inspection judgment.

## Timeline tools

Run any example, then open:

- **Results > Ticket activity timeline…** for a Gantt chart of a selected ticket's work.
  It includes source work, nested tools and child inspections associated with the parent
  ticket. Parent tickets are preferred when opening a completed parallel inspection.
- **Results > Block concurrency…** for one row per retained logical block and a rolling
  time axis. Set the window between 0.1 and 600 seconds. Empty rows identify registered
  blocks that have no work in the current window.

Green is pass, red fail, purple error, amber waiting/warmup, blue active, and gray inactive
or skipped. Colors show each tool's own result; they do not rewrite upstream failures or
assign a final ticket judgment. Nested ToolBlock spans include their children and overlap
them. Parallel part instances share their logical block's row, with up to eight stacked
lanes; further overlaps share the last lane. Full runtime instance paths remain in details.

Diamond flags identify exceptions, blocked inputs, rejected failed inputs, queue activity,
dropped frames and plugin diagnostics. Selecting a bar/flag shows the invocation, source,
thread, ticket identities, timestamps, result and event data. Input arrival during a busy
queue is distinct from a capacity wait: a busy queue can still accept another frame.

Filter by ticket or block name/path, zoom, drag to pan, or pause the view to inspect a
frozen snapshot. **Export events…** writes the retained event log as JSON and preserves
existing files. Headless `--report` receipts also contain an `activity` snapshot. No camera
is required: the synthetic examples work, and the parallel part inspection example makes
the concurrency visible. Run continuously to see the rolling window advance.

## Event contract

Events use `block_event_v1` and include:

| Field | Meaning |
| --- | --- |
| `kind` | `input`, `start`, `finish`, `exception`, diagnostic condition or plugin-defined kind |
| `invocation` | Correlation ID for one call, including a blocked arrival before it starts |
| `source` | Block ID/name/type, full runtime `path`, stable `logical_path` for row aggregation |
| `tickets`, `parent_tickets` | Typed ticket identities; numeric/string/timestamp ticket values retain their types |
| `time` | UTC timestamp, process clock identity and monotonic nanoseconds |
| `thread` | Reporting thread identity |
| `data` | Compact arbitrary JSON diagnostic data |
| `started`, `duration_ms`, `judgment`, `result` | Finish-event timing and pass/fail/score/final-decision summary |

Each accepted invocation reports start and finish, including waiting/warmup and error
results. Failed-input rejection reports `rejected` without a fake work bar or timing
sample. Input sources reserve their ticket before acquisition/processing, so their live
start events already identify the work. A failed acquisition therefore consumes an auto
ticket and remains traceable. Invalid source configuration may fail before a ticket can
be assigned; those events still identify the block/invocation. `ticket_assigned` records
the assignment separately. A queue/collection tool can discover additional tickets during
processing; its finish event supplies the complete observed set for the reconstructed bar.

Concurrent calls to the same block serialize through a per-instance execution lock.
The second call reports `blocked`, then `unblocked` when it can start. Cancelling a blocked
call reports `cancelled_wait` without changing the active call's inputs/results. Recursive
execution of the same instance is rejected to prevent deadlock. Different block instances
and independent per-part jobs can still run concurrently.

Queues report `queue_busy` for arrivals during downstream work, `queue_wait`/`queue_resume`
for capacity gates, and `dropped` with the overflow policy and affected ticket. A capacity
gate before requesting a camera frame has no frame ticket yet. These markers report
observed software conditions, not hardware trigger collisions or camera-driver internals.

## Custom block diagnostics

Every block also provides `block.logger` with severity, structured parameter snapshots
and exception tracebacks. **Results > Block diagnostics** opens the selection-aware
bottom dock with verbosity filters. See [block diagnostics](diagnostics.md).

Inside `process`, use the common reporting method:

```python
self.report_event("measurement", {"score": 0.93, "part": "left"})
```

It automatically uses the invocation's tickets. Outside an invocation, provide values:
`block.report_event("trigger", {"channel": 2}, values={"image": packet})`.
Data must be a JSON object of at most 8 KiB; never attach images or native objects.
Standalone block users can attach an `ActivityLog` through `Graph.bind_activity(log)`;
custom consumers can read private `log.snapshot()` copies and use `activity.spans()` to
reconstruct completed/running intervals. `ActivityLog.emit()` validates the low-level
schema for adapters that publish events directly.

## Retention and clock boundaries

The default log retains at most 20,000 events and approximately 16 MiB of serialized event
data. Individual events are capped at 256 KiB; ticket and parent-ticket lists each retain
up to 64 identities and report omissions. Finish records include their start timestamp so
completed bars remain reconstructible after the original start record is evicted. An active
start that is evicted cannot be reconstructed until its finish arrives. The views report
evictions and display limits (400 rows, 4,000 bars, 2,000 flags) instead of implying complete
history. Retained history is runtime-only; F5 and Stop/Resume preserve it, while opening a
new project creates a new log. Export before losing history that matters.

Only readings from one process clock are plotted together; use the clock selector when
viewing mixed-clock records. UTC is for correlation, not duration calculations. Bars use
the [block work timing boundaries](timing.md), excluding a wait before the invocation can
start. Arrival/blocked flags expose that additional delay. No automatic scheduler critical
path, cross-machine clock synchronization or hardware real-time guarantees are implied.
