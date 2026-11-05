# Execution and ticket timing

For start/finish events, ticket Gantt charts and rolling block concurrency views, see
[block activity events and timelines](activity-timelines.md).

Every block has runtime `block.timing` statistics, also included in headless JSON
reports and the inspector:

- **Current**: live elapsed time while an invocation is executing; blank when idle.
- **Last**: duration of the most recently handled invocation, in milliseconds.
- **Average**: cumulative arithmetic mean, with an activation count.

Timing starts when an invocation is accepted, before validation, input snapshots and
processing. It ends after output validation, payload/outcome stamping and retained-result
preparation, before publishing the ticket-history event. It includes Python/OpenCV work,
internal waits and nested processing, but excludes upstream queue waits and GUI frame
acknowledgment. This is elapsed wall duration, not CPU time. Nested durations overlap;
summing them double-counts work.

Failures and normal waiting/polling/warmup invocations count. Failed-input rejections,
inactive branches and scheduler-skipped blocks retain their timings without adding samples.
Parallel part inspections aggregate child tools' timings across all executed parts,
separately from the selected image preview. Child processing state remains independent.

Statistics survive F5, Stop/Resume, layout moves, configuration resets and queued worker
handoff. Acquisition statistics include all handled captures, even captures later dropped
by a queue. They belong to the current runtime: save/open starts fresh statistics rather
than serializing histories. `elapsed_ms` remains the last duration for existing plugins;
`block.timing.snapshot()` supplies current/last/average/count.

## Ticketed connections

Ticketed outputs carry a `processing_time` payload with model `processing_time_v1`:

| Field | Meaning |
| --- | --- |
| `started_at` | UTC ISO 8601 timestamp when the source started handling the frame |
| `finished_at` | UTC timestamp when this tool finished preparing validated outputs |
| `elapsed_ms` | Source start to this output, including queue/scheduling waits |
| `started_ns`, `finished_ns` | Local monotonic readings used to calculate duration |
| `clock_id`, `finished_clock_id` | Process identities for those clock readings |
| `inferred_start` | True if a legacy/external ticket lacked timing and measurement began at the first observed tool |

The start follows images/data through ordinary tools, pyramids, batches, branches,
queued frames and ToolBlocks. Each output advances its own finish without mutating sibling
branches. A new input source starts a new acquisition measurement even when reusing a
PLC-supplied identity. The ticket value and identity remain unchanged; measurement works
for numeric, string and timestamp tickets. A driver's actual exposure timestamp is not
inferred from the source invocation timestamp.

Durations use a monotonic clock so wall-clock corrections do not distort them. Readings
from different processes/machines cannot be subtracted: UTC timestamps remain visible,
but `elapsed_ms` is null until a suitable synchronization contract exists.

## Ticket results

**Results > Ticket results…** shows Start UTC, Last finish UTC and elapsed milliseconds,
plus individual tools' Work ms. The ticket row's `processing_time` spans the earliest
observed start to latest observed finish for that identity. Out-of-order parallel events
cannot shorten this local span. It survives truncation of older event rows. Errors and
rejections remain visible in the timeline; errors include their work duration.

The latest finish can advance while other branches or queued work remain outstanding.
It is an observation of completed handling, not a guarantee that every branch finished.
Use Join/parallel aggregation followed by **Final ticket judgment** to designate the
decision boundary; `finalized_at` records the first explicit decision time. Later diagnostics
can still advance the observed finish. Reusing a fixed/external ticket identity combines
its observations, so use distinct IDs per workpiece.

Output timestamps precede stamping/copying the retained output snapshot; Work ms includes
that preparation. Their measurements can therefore differ slightly at a terminal block.
Both exclude subsequent ticket-history publication.
