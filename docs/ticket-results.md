# Ticket outcomes, input gates and final judgments

Final judgments support stable fail codes/messages, and the status bar reports cumulative
ticket counts and end-to-end timing. See [production counts and validation](production-validation.md)
for reason propagation, counting boundaries and scored validation image sets.

Ticketed connections carry source-start/output-finish UTC timestamps and monotonic
elapsed time. The results window shows these spans and per-tool durations; each block
retains current/last/average statistics. See [execution and ticket timing](timing.md)
for queue waits, parallel spans and explicit decision boundaries.

Tools processing ticketed data publish a compact result event to the job's shared
`graph.ticket_results` consumer. All emitted connections carry a `tool_outcome_v1`
record in their payload's `outcome` field. This works for image, depth, pyramid,
number and data ports; groups retain outcomes on their individual member packets.
Unticketed data remains usable and has no automatic ticket result event.

## Connection status and input policies

The connection outcome contains:

- `passed`: this tool's result, separate from earlier tools.
- `chain_passed`: this result and the accumulated status of data used to produce it.
- `source`: block ID, name, type and a nested graph/part path.
- `score`, `message` and `data`: the optional score and a compact arbitrary JSON record.

A later passing operation does not erase an accumulated failure. For example,
a failing contour inspection followed by successful grayscale conversion produces
`passed: true, chain_passed: false` on the grayscale output. The tracker retains
the original inspection failure and the later successful conversion separately.
Ordinary tools conservatively combine the inputs they consume. Routing, queues,
batch/unbatch and ToolBlock boundaries preserve per-ticket status rather than
applying one burst's aggregate node color to every ticket. A group input is failed
when any of its members failed. Selecting one member retains that member's status.
Neighborhood operations still use their documented mask/region semantics.

Each input has an inspector **failure policy** dropdown:

| Policy | Behavior |
| --- | --- |
| `continue` (default) | Process passing or failed inputs; preserve accumulated failure on outputs |
| `reject_failed` | Reject the invocation when that provided input carries a failed accumulated status |

Rejection happens before processing and input capture. It publishes an
`input_rejected` tracking event but **no data output events**, including scalar
outputs. The last processed input image, payload, result, output images and
graphics remain available. The block is `inactive` and the inspector explains the
rejection; retained previews belong to the earlier processed ticket. Headless
receipts use current emissions, never those retained previews, to report counts.
An input with no outcome is accepted for backward compatibility. Unticketed
upstream errors still skip dependent tools using the existing error behavior.

For an ordinary invocation with several inputs, rejecting any configured input
rejects the whole invocation; it is not silently omitted as an optional argument.
For bursts, the gate applies to each invocation independently and later accepted
entries still run. Buffered operators gate each arriving entry before buffering.
Queues also filter before accepting captures, including background acquisition.
Already accepted waiting frames can still drain when a newer arrival is rejected.
Gate related branches consistently: dropping only one input to a Join can leave an
unmatched ticket until reset or the configured buffer limit is reached.

Policies are saved in project JSON as `input_failure_policies`; omitted policies
mean `continue`. Reusable ToolBlock exports retain public and internal policies,
including public-terminal renames. Applying configuration/reset clears histories
and flow buffers as before. Layout-only moves keep them.

## Final judgment

Ordinary pass/fail is an intermediate measurement; it does not finalize a ticket.
Add **Inspection > Ticket results > Final ticket judgment** at an explicit decision
point. Connect a ticketed `value`, including a data result or image. Its modes are:

- `upstream` (default): finalize using the accumulated connection/group outcome.
- `boolean_input`: use an actual boolean on the optional `passed` data input.
- `pass` / `fail`: explicitly assign that final judgment.

The block forwards `value`, emits a `judgment` record and publishes a distinct
`final_judgment` event. Its `data` parameter adds a compact JSON record such as a
station or recipe identifier. Keep its value input's failure policy at `continue`
when it must finalize failing tickets. A final block behind a rejecting gate cannot
judge suppressed events. Connect it independently to the inspection result when
both finalization and failure-gated diagnostic branches are needed.

Final judgments are explicit decisions, not automatic end-of-graph detection.
Overrides can declare final pass despite an earlier failure; earlier failures and
`chain_passed: false` remain visible. Conflicting final decisions produce
`conflicting_final`, not last-writer-wins. Late ordinary events do not overwrite an
explicit final decision. A final judgment on a group applies that decision to all
its member tickets. Per-part final decisions can live inside a parallel inspection
ToolBlock; aggregate final decisions belong after its aggregate result.

## Common consumer and lookup window

Open **Results > Ticket results…** during or after execution. Search ticket value,
kind, session, source or current status. Select a row for its event timeline,
source paths, timestamps, scores, custom data and parent-part ticket relationship.
The modeless window updates while the job runs and preserves selection. It shows
the live acquisition/inspection consumer without reading mutable worker images.

The consumer is shared by the root graph, nested ToolBlocks, independent branch
workers, parallel per-part jobs and background acquisition. Captured upstream
snapshots do not replay source events when a queued frame is consumed. Events are
serialized by the consumer and have unique IDs, sequence numbers and UTC times;
worker completion order is not a FIFO claim. Processing exceptions publish `error`
when a consumed ticket is known. Acquisition errors before an image/ticket exists
cannot be assigned to a ticket automatically. Cancellation and inactive branches
do not fabricate passing results or final decisions.

| Status | Meaning |
| --- | --- |
| `in_progress` | Intermediate events exist; no final judgment or known failure |
| `intermediate_fail` | A tool/chain failed; no final judgment |
| `error` | A processing exception was reported; no final judgment |
| `final_pass` / `final_fail` | An explicit final judgment exists |
| `conflicting_final` | Both final pass and final fail were assigned |

`had_failure`, `had_error` and aggregate judgment flags survive event trimming for
retained tickets. No implicit timeout, expected-tool manifest, persistent database
or PLC handshake is provided. A ticket can remain in progress indefinitely if no
final block receives it. The headless runner's job `passed` field continues to
describe tool execution/inspection outcomes; final ticket decisions are separately
reported under `ticket_results` in its JSON receipt.

Ticket identity follows Join/FIFO: `(session, kind, value)`, intentionally grouping
the same external/PLC ticket across camera sources. Numeric `1` and string `"1"`
are different tickets. Use unique external values for distinct workpieces or
provide a new session in the external ticket object. Reusing an external value in
the same session appends to the existing work record, including its final flags.

History is bounded: defaults are 500 tickets, 128 recent events per ticket and
16 MiB of serialized retained events, plus Python/Qt overhead. Events are limited
to 16 KiB; plugin `event_data` is limited to an 8 KiB JSON object. Old least recently
updated tickets are evicted to enforce count/byte bounds; counters disclose ticket
evictions and per-ticket trimmed events. Images are not stored in the tracker.
These limits are not process-heap limits or a production audit retention policy.

History survives GUI Run once/Continuous and Stop/Resume. Configuration and flow
reset keep recorded results, while starting/opening another project begins a new
consumer. Ticket events are runtime state and are not saved in project JSON.
For a durable receipt, use the headless `--report` option. A camera or database is
not required for the example or automated validation.

## Example

Open [ticket results](<../examples/control flow tools/ticket_results/ticket_results.json>), run once, then open **Results > Ticket
results…**. Two synthetic parts produce a `final_pass` ticket. The grayscale
diagnostic branch accepts only passing inspection images. Change **Find contours >
expected_count** from 2 to 3, apply, and run again: the new ticket is `final_fail`,
and that branch emits nothing. Configuration application intentionally clears old
previews; runtime failure rejection itself does not overwrite prior previews.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/control flow tools/ticket_results/ticket_results.json" --repeat 3 --report .local/ticket-receipt.json
```

## Custom blocks

Return `Result(..., passed=False, score=0.3, event_data={"reason": "size"})` for an
ordinary failed measurement. An authorized decision tool can additionally return
`final_judgment=False`; use a real boolean, or `None` for intermediate results.
The executor validates, stamps and publishes these results automatically. Do not
put image arrays or unbounded logs in `event_data`.

`graph.ticket_results.snapshot()` returns a private JSON-compatible history;
`find(ticket)` looks up a typed ticket object. `Graph.bind_ticket_results(consumer)`
shares a compatible consumer with a host integration. Direct standalone
`Block.execute()` calls carry outcomes but publish centrally only when the block
is attached to a graph's consumer (via binding/running). Composite plugins that
emit independent ticket outcomes can opt into `outcome_passthrough` and
`per_packet_outcomes`; ordinary single-result tools should leave these disabled.
