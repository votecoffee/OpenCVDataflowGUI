# Tickets and control flow

Ticketed connections also carry tool outcomes. Input failure policies can suppress
failed entries before buffering, and a shared collector tracks intermediate events
and explicit final judgments. See [ticket results](ticket-results.md) and
**Results > Ticket results…**. A rejection emits no downstream data; configure
matching branch gates consistently to avoid leaving a Join without one side.

Input sources now attach a ticket to each acquisition. Tickets travel in wire
payloads alongside primary images/results; ordinary processing preserves the
primary input's ticket. This is separate from spatial transforms and pyramid
provenance. A reference/template or mask is not automatically synchronized with
the primary image; use an explicit join when multiple results must agree.

## Acquisition identities

Built-in `input.*` blocks expose optional `ticket` input and `ticket` data output.
Select a source to configure **Ticket mode**:

| Mode | Meaning |
| --- | --- |
| auto_number | Increasing acquisition number within the graph's runtime session |
| auto_timestamp | UTC ISO8601 timestamp with microseconds |
| external | Use the wired ticket input; a missing ticket is an error |
| fixed | Use the inspector's Fixed ticket JSON value |

Numbers and strings are accepted directly. For explicit timestamp semantics use
`{"timestamp":"2026-11-04T15:30:00Z"}` with a timezone, or pass a complete
`flow_ticket_v1` record. Equivalent timestamp offsets are normalized for matching.
Booleans, empty text and nonfinite numbers are rejected. Text is limited to 1024
characters. Plain timestamp-looking strings are still valid string tickets.

Ticket records contain `value`, `kind`, `session`, acquisition `sequence` and
`source` ID. Auto numbers are global to acquisitions/group creation in that graph,
not guaranteed consecutive for one source. Branch copies retain the full identity.
External/fixed raw tickets share an external namespace so separate cameras can
match the same PLC job ID. Complete records can supply their own session namespace.
Acquisition sources bind the record's source ID to the acquiring block. Joins match
the part identity across sources; manifest FIFO additionally distinguishes source
IDs, allowing individual camera images to share one PLC part ticket.
Use unique external identities for active/recent work; resetting state clears the
duplicate history. A fixed constant is useful for one-shot configuration, not a
replacement for a PLC counter that advances per part.

**Assign ticket** can replace the ticket on an existing value while retaining its
payload and acquisition source. Use this for externally triggered jobs or plugin
sources. A PLC driver can output a number/string on a data port and wire it to
`ticket`; no PLC network driver or PLC writes are added by these blocks. Custom
sources with type IDs beginning `input.` participate in automatic acquisition
ticketing; `ticket` is a runtime-managed reserved terminal name.

## Blocks

[Coordinate inputs](input-coordination.md) adds ticket-matched deadlines, greedy
latest snapshots, trigger-driven latest snapshots and first-arrival races. It is
an explicit alternative to the indefinite Join; ordinary OpenCV inputs keep their
existing scheduling behavior.

For one inspection repeated across many detected parts, use
[Inspect each pose in parallel](parallel-inspection.md). It has an editable
per-part ToolBlock, bounded workers, aggregate results and optional early stopping.

For acquisition buffering and one-frame-at-a-time processing, insert
[Queue input images](input-queues.md) after the source. Optional background capture
can continue while the current frame is inspected.

All new operators appear under **Control flow** in the toolbox.

| Block | Behavior |
| --- | --- |
| Assign ticket | Auto, timestamp, external or fixed identification on any primary value |
| Split parallel branches | Duplicate a ticketed value into 2–4 isolated branches |
| Join parallel branches | Match 2–4 branch results by typed ticket identity, independent of arrival position |
| Batch serial entries | Collect 1–64 successive entries, optionally from distinct acquisition sources |
| Unbatch serial entries | Emit every group member as a separate serial event with its original ticket/payload |
| Select group entry | Extract a zero-based group entry without serial expansion |
| Case select | Route to the first matching case output, or default; unselected outputs emit nothing |
| FIFO reorder by ticket | Hold completed results until the expected head ticket arrives |

Split has four output sockets; `branches` enables the first 2–4. Its `workers`
setting (1–8, default 2) enables concurrent execution of independent ready nodes in
that graph. Graphs without Split execute serially. Acquisition sources remain on
the owning execution thread, as do ToolBlock boundaries; a ToolBlock can enable
parallelism inside its own graph. Each node processes its own serial entries in order;
this is graph-level branch concurrency, not concurrent calls into one block or an
independent camera-trigger service. Python plugins must be safe to run on worker
threads, or use `workers=1`. Source/branch values are private snapshots so sibling
mutations do not change each other. OpenCV may also use its own internal threads.

Join buffers unmatched tickets and reports a group plus individual `branch_1`…
outputs when all configured branches agree. Duplicate arrivals on one branch are
errors. A missing branch never causes a result to be paired with the next ticket.
Join emits complete groups as available; FIFO is a separate policy.

Batch accepts up to four input streams, consumed in deterministic connection order
per available serial position. For example, three camera image ports can connect
to `item_1`…`item_3` with `size=3` and `distinct_sources=true`. Every group member
retains its own ticket and coordinate payload; the group gets a separate ticket.
Different images are not relabeled into one spatial frame. To require the same
part ID across cameras, use Join rather than positional batching.

Unbatch emits each original member; downstream blocks execute once per emitted
entry, including output writers. `order=stored` preserves the group's order;
sequence sorting is explicit. Use Select group entry when only one item is needed.
Groups use `any` ports and must be unbatched/selected before ordinary image/data
consumers. Receipts describe group images by shape/depth, never raw pixel arrays.

Case select has 1–4 configured cases and a default output. Choose equals,
less-than, greater-than or inclusive range comparison. An optional numeric modulo
supports even/odd or round-robin case selection. The selector can come from its
data input or a dotted path such as `ticket.value` or `value.score`. Cases are
evaluated in order. Chain case blocks to build larger decision trees. Inactive
branches do not execute downstream writers or reuse old images.

## FIFO and bounded waiting

**Manifest mode** supports strings, timestamps and arbitrary numbers. Wire the
pre-branch stream to `expected` and completed branch results to `value_1`…`value_4`.
The expected stream defines FIFO order; results can arrive first or out of order.
There is no lexical sorting or guess about missing string/timestamp identities.

**Number mode** expects integer ticket values starting at `start`, incrementing by
positive `step`. It rejects late/off-sequence tickets and holds gaps. Do not use
this mode if other acquisitions/groups consume numbers in the same clock; use the
manifest instead. Neither mode silently skips a missing expected ticket.

Join, Batch and Reorder expose `max_pending` and `max_bytes_mb` limits (default 64
entries/128 MiB). Overflow or duplicate synchronization input fails the block and
clears its transient buffer. Join/Reorder remember the last 4096 completed keys to
detect replay. These Join/Batch/Reorder operators have no automatic timeout or drop policy. Resolve missing
results or explicitly use **File > Reset flow state**. Per-node input/output event
expansion is limited to 1024 per graph iteration.

`waiting` means a block needs more entries; `inactive` means its input branch
emitted nothing. These are normal states, not inspection failures. Runtime errors
still skip dependent nodes, and negative inspections still supply their results.
The headless runner accepts waiting/inactive states; a successful run does not
prove that every buffer drained. Inspect `queued` outputs for that requirement.

Buffers and acquisition counters survive continuous and repeated GUI Run once
iterations, including nested ToolBlocks. They are not serialized in project JSON.
Loading a project starts a fresh session. Applying configuration, rewiring or
Reset flow state clears queued work; moving nodes preserves runtime state. The GUI
retains its last buffered snapshot when stopped so another Run once can continue.
Closing the graph releases buffers and camera resources. Cancellation is
cooperative between block invocations; an in-flight native operation can finish.

## Example

Open **File > Load ticketed control flow example** or `examples/control_flow.json`.
Three synthetic camera sources are grouped, then deliberately unbatched in reverse
order. Two parallel image-processing branches join by ticket, case selection routes
even/odd tickets, and a manifest restores the original acquisition order before a
histogram. Run once: the final histogram receives three serial image events, FIFO
has `queued=0`, and all nodes pass. Continuous runs need no camera or file writes.

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/control_flow.json --repeat 3
```
