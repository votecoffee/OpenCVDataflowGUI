# Production counts, fail reasons and validation image sets

The bottom status bar continuously shows **Total, Pass, Fail, pass rate and average
end-to-end processing time** for the current job. These are ticket counts, not block
activation counts. Totals survive F5, Stop/Resume and eviction of detailed ticket events.
**Results > Reset running counts** resets the counters while stopped, retaining diagnostics
and ticket history. Opening a different project starts fresh runtime counts.

## Counting and timing

An explicit **Final ticket judgment** supplies the decision when present. Repeated events
for a ticket update its result without counting another image; conflicting final decisions
count as a failure. Per-part inspection child tickets do not inflate parent-image totals.

For programs without a final-decision block, a root graph iteration with no outstanding
waiting blocks records completed tickets using their accumulated outcomes/errors. This
completion measurement does not assign an explicit final judgment to the connections or
rewrite the ticket's judgment history. A source-only acquisition worker does not perform
this fallback completion. Add an explicit final judgment when the program's acceptance
boundary differs from ordinary graph completion.

Pass rate is `pass / total * 100`. Average time covers the source's start through final
decision/completed root iteration, including input queues and processing branches; a later
completion observation extends the measured span. Timings from incompatible process clocks
are excluded from the average and the timed-ticket count is available in the statistics.
Neither UI frame acknowledgment nor validation runs add to production processing totals.

The aggregate counters are cumulative; identity corrections/deduplication retain the most
recent 8,192 completed ticket identities. A repeated identity after leaving that window is
counted again. Use distinct numeric/string/timestamp IDs per workpiece, as required for
ticket coordination. Counters and their recent ledger are runtime data, not saved program
configuration. See [timing](timing.md) and [ticket results](ticket-results.md) for clock and
final-decision semantics.

## Explain a failed judgment

Final ticket judgment now offers:

- `fail_reason`: a stable code, such as `part_count` or `surface_scratch`.
- `reason_message`: its readable explanation.
- Optional `reasons` data input: one reason or a list of codes/objects.

A reason object is `{"code": "surface_scratch", "message": "Scratch in inspection region"}`.
Custom blocks can return `Result(..., passed=False, fail_reasons=[...])`. Ordinary failing
tools without an explicit reason get a code based on their block type and their result
message. Reasons follow ticketed connections through later tools, groups and final decisions;
the final judgment output includes a `fail_reasons` list for output/reporting blocks.
Ticket results display final reason codes and retain the complete explanations in events.

Configured final reasons precede inherited reasons. Explicit final PASS emits no final
failure reasons, while earlier failures remain visible in history and accumulated outcomes.
The reason contract allows up to 32 unique code/message pairs and 4 KiB total; codes are
1–128 characters and messages up to 512 characters. Merged lists retain reasons within
that budget. Prefer stable codes for automation and validation; wording can change independently.

## Build a validation set

Open **Results > Validation image sets…** while stopped:

1. Create/name a set and choose the root image source to substitute. Ordinary file,
   camera and compatible synthetic image sources can be replaced for validation. Its
   connected outputs must be `image`, `info` or `ticket`.
2. Choose a root Final ticket judgment, or automatic terminal results.
3. Add images and assign an expectation to each. Enter a preferred reason **code** for
   expected failures when the failure mechanism matters. Notes are optional.
4. Choose **isolated** (fresh program state per image) or **sequence** (preserve temporal
   state in table order). Move images up/down to control sequence order; unknown images
   can warm up a temporal model without being scored.
5. **Save set**, then save the project to retain definitions. **Run image set** also saves
   the edited definitions into the current project before running. Import/export set JSON
   is available; images are referenced rather than embedded/copied.
6. Run and inspect the score/per-image results; **Export report…** writes a JSON report.
   Stop retains a partial report marked cancelled. Existing export files are preserved.

| Expectation | Scoring and acceptance |
| --- | --- |
| Must pass | Required PASS; mismatch fails required checks |
| Should pass | Advisory PASS; mismatch reduces score |
| Unknown | Observed but excluded from score; can remain unresolved during warmup |
| Should fail | Advisory FAIL, with optional preferred code; mismatch reduces score |
| Must fail | Required FAIL, with optional preferred code; mismatch fails required checks |

Score is the unweighted percentage of matching **known** expectations. A preferred code
must appear among that selected judgment's reasons to match an expected failure. Required
checks pass only when every must criterion matches, no execution/cleanup error occurred,
the set contains images and the run was not cancelled. Advisory mismatches do not fail
required checks. An all-unknown set has no numeric score. Execution errors or unreadable
images never satisfy an expected FAIL.

Definitions and paths are stored with the program. Relative image paths resolve against
its saved project directory (working directory for an unsaved program). Save exported
sets with their referenced images when sharing; exported set references use absolute paths.
The catalog supports up to 64 sets and 10,000 total image entries.

## Execution and reports

Validation runs in a separate worker/program instance, preserving the live job's learned
runtime state, previews and production totals. Trained assets/configuration and the saved
default recipe are copied. The chosen source uses immutable image bytes with the original
source's color/depth conversion settings; reports record their SHA-256, program digest,
OpenCV version, actual decision, reasons, errors and timing. Per-image wall time also
includes validation setup/file reading; processing time uses the ticket timing boundary.

Background queues run synchronously in validation. Other camera sources are rejected
before acquisition. Output blocks are skipped by default; enable them explicitly when
their side effects are part of the test. Skipped output data ports emit no values, so a
decision downstream of such a port will be unresolved. Trusted custom plugin code still
executes as part of the program. The initial workflow substitutes one root image source;
multi-camera validation bundles require a future extension.

Try [validation image set](../examples/inspection/validation_images/validation_images.json). Its five synthetic cases cover all
expectations and should score **100%** with required checks passing; the must-fail image
reports `part_count`. No camera/download is needed.

```powershell
.\.venv\Scripts\python -m opencv_dataflow.validation examples/inspection/validation_images/validation_images.json --set validation_smoke --report .local/validation-report.json
```

Choose a new report filename on each invocation. `--plugin path/to/trusted_plugin.py` loads
custom block definitions before opening the program. Exit codes are 0 for required checks
passing, 1 for validation failure/cancellation, and 2 for setup/report errors. Advisory
mismatches can therefore produce a score below 100% with exit code 0.
