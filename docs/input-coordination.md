# Coordinating multiple inputs

Use **Control flow > Synchronization > Coordinate inputs** for explicit arrival
policies. Connect two to four streams to `input_1` … `input_4`. This keeps ordinary
OpenCV node scheduling predictable and makes intentional reuse or missing inputs
visible in the graph.

## Choose a policy

| Mode | Behavior | Suitable use |
| --- | --- | --- |
| `ticket_all` (default) | Wait for one arrival on every configured input with the same ticket; optionally expire incomplete sets | Several cameras/inspection branches for the same workpiece |
| `latest` | After warmup, emit a snapshot on every new input event, reusing fresh cached inputs | Live previews and exploratory fusion |
| `trigger_latest` | Only the designated primary input triggers a snapshot using fresh cached supporting inputs | Image inspection with a cached reference/configuration |
| `first_arrival` | Emit the first arrival for each ticket, then ignore later arrivals for that recently closed ticket | Redundant alternatives where one winner is sufficient |

For inspection across cameras, prefer **ticket_all with a common PLC/workpiece
ticket**. Matching uses `(session, kind, value)` and allows different acquisition
source IDs. Independent auto-numbered camera acquisitions are not automatically
the same workpiece. The existing Join remains an indefinite strict barrier;
Batch is positional grouping, not a proof that members belong to one workpiece.

First-arrival means one winner **per ticket**, not a single global winner. The
first event delivered to the block wins. The current scheduler delivers ties in
deterministic stream/connection order; this is not a claim about physical capture
order. A new unique ticket starts another race. These blocks do not start parallel
work themselves; use Split or parallel inspection where appropriate.

## Deadlines and late arrivals

`max_wait_ms` defaults to 1000 in ticket_all mode. Its deadline starts when the
coordinator accepts the first member for a ticket. Other members and replacement
duplicates do not extend it. An arrival at or after expiry is late. Zero explicitly
disables expiry. The timeout choices are:

- `discard` (default): close the incomplete ticket, publish a failed timeout event,
  and emit no data for it.
- `emit_partial`: emit available members with `complete: false`, an explicit
  missing-input list and failed coordination status on the group/value outputs.
- `error`: report the timed-out ticket and fail the block; transient flow buffers
  clear under the existing error/reset contract.

Closed tickets suppress late arrivals by default, including duplicates after
completion and alternatives after a first-arrival winner. `late=error` is an
explicit alternative. Suppressed arrivals emit **no data events**, including no
status-output event. While a ticket is pending, `duplicates=error` is the default;
`replace` replaces that input's pending member without restarting its deadline.

The ticket window records timeouts, ignored late arrivals, and trigger inputs that
arrived without supporting data. These are intermediate events, not automatic
final judgments. Route a completed/partial group to Final ticket judgment when an
explicit final decision is desired. Discarded timeouts have no downstream data
event, so a downstream final block cannot finalize them automatically.

## Freshness and warmup

Latest modes intentionally allow different tickets in one snapshot. They preserve
every original member ticket, set `mixed_tickets: true` when needed, and give mixed
groups a separate runtime ticket. They do not relabel cached images as the new
workpiece. Use these modes deliberately; finalizing a mixed group applies that
decision to its members, including reused tickets, and can create conflicting
final judgments. Prefer ticket_all for actual multi-camera workpiece decisions.
Managed recipe revisions are also reported as `mixed_recipes`. Strict ticket_all
and Join reject differing managed revisions; ordinary processing requires a
consistent recipe context. See [recipes](recipes.md).

`max_age_ms` defaults to 1000 and bounds cache age at coordinator ingress. Zero
explicitly allows unlimited reuse. Expired entries are removed on subsequent
arrivals/polls. This is receipt age, not camera exposure time or an external clock.
`warmup=all` requires every configured input to be present and fresh before an
emission. `warmup=any` explicitly permits incomplete snapshots, flagged failed
with their missing inputs. In trigger_latest mode, supporting updates only refresh
the cache; a trigger without required support is not queued for later replay.
Prime the supporting inputs before the trigger, or wait for the next trigger.

"Any input changes" means **a new event**, not pixel-by-pixel equality detection.
Within a graph iteration, latest can emit after each arrival, including an
intermediate snapshot mixing new and previously cached data. Inspect the reported
member tickets and ages; no atomic camera snapshot is implied.

## Outputs and failure status

- `group`: one labeled FlowGroup event for the coordinated snapshot. Complete
  groups use configured input order; partial groups contain only present labels.
- `value`: the configured primary input, or the triggering/first present input if
  the primary is missing. Its original ticket and spatial/image metadata remain.
- `info`: completeness, reason, present/missing labels, trigger/primary input,
  member tickets, mixed-ticket flag and ingress ages. This context is also carried
  in the outputs' `coordination` payload and recorded in ticket events.

All three streams describe the same emissions in order. Waiting/ignored events do
not emit counters downstream; the node message shows waiting/cached/emitted,
timeout and late-arrival counts. A first-arrival result intentionally needs only
one member and is not failed simply because other inputs are absent. A complete
group still carries member tool failures; judge the **group** to inspect all member
outcomes. The primary value's status alone does not assess other cameras.

Use Select group entry for complete groups, Unbatch for individual entries, or a
group-aware ToolBlock/plugin for partial snapshots. Partial indices can shift;
inspect labels and `info.missing`. Do not split several partial groups into ports
and rely on ordinary positional multi-input scheduling to realign them. A failure
gate on the group/value can reject partial snapshots. Rejecting a failed arrival
before coordination does not replace an older latest-cache entry; the freshness
limit still governs that cached entry.

## Timing and lifecycle boundaries

Deadlines are **nonblocking and cooperative**: the scheduler polls this block
after available input events and on idle graph iterations, including inside
ToolBlocks. No sleep blocks the graph waiting for another input. A one-shot run
can return waiting; it does not stay alive to fire a timer. Continuous execution
checks again on later iterations. Unavailable upstream branches do not prevent
the coordinator from checking its existing pending tickets.

Sources currently execute serially. The coordinator cannot receive an event or
service a timer while a source/native call is blocking the execution thread, and
it cannot recover the time a frame waited upstream before ingress. Therefore these
timeouts are not hard real-time deadlines and do not guarantee simultaneous camera
exposures. Use suitable hardware triggers/PLC/device coordination for that.
Independent source workers and timestamp-window matching are future work in
[issue 5](https://github.com/votecoffee/OpenCVDataflowGUI/issues/5). Timestamp
tolerance, clock-domain alignment and event-time watermarks need explicit designs;
timestamp-looking ticket text alone does not implement them.

`max_pending` and `max_bytes_mb` bound stored pending/cache data. Closed-key history
is separately bounded by `history_size` (default 4096); sufficiently old closed
keys can be accepted again after eviction. Counted buffer bytes are not a hard
process-heap limit. Buffer overflow/configuration errors fail and clear transient
state. Existing input failure policies run before buffering and preserve rejected
inputs' prior previews.

Runtime buffers, freshness timestamps and closed keys survive GUI Run once and
Stop/Resume but are not saved in project JSON. Monotonic deadlines include time
spent stopped; expired pending sets are handled on resume. Configuration, rewiring
or Reset flow state clears coordination state. Layout-only moves preserve it.
Background-acquisition prefixes cannot contain buffered coordination operators;
the existing single root background queue is not an independent multi-camera
acquisition service.

## Example

Open [input coordination](<../examples/control flow tools/input_coordination/input_coordination.json>). A **Counter input** stands in for
a PLC part ID and supplies each synthetic camera's external ticket input. Run once
or continuously, inspect `info`, and open **Results > Ticket results…**. Both cameras
agree on one part ticket and the group receives an explicit final judgment. No
camera or external writes are needed.

Counter input emits exact integer values on its data `value` port, configured by
`start` and nonzero `step`, with runtime progress retained between runs. Its separate
automatic `ticket` output identifies the counter acquisition; use **value** when
feeding the cameras' ticket inputs. It is a demonstration source, not a PLC driver.
Production part identities must remain unique across workpieces and resets.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/control flow tools/input_coordination/input_coordination.json" --repeat 3
```
