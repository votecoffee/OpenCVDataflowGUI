# Block diagnostics

Open **Results > Block diagnostics**, or press **Ctrl+Shift+D**, to show the bottom
dock. It can be closed, moved or floated using the usual dock controls.

- Select a graph node to see only that block's retained diagnostic events.
- Clear the selection to see events from all blocks, including nested tools and workers.
- Choose a **Minimum severity** from TRACE, DEBUG, INFO, WARNING, ERROR and CRITICAL.
  For example, ERROR includes errors and critical events; TRACE includes every level.
- Search messages, parameter values, ticket identities and exception tracebacks.
- Select a row to inspect its complete message/event context, parameters and exception.
  Error rows with exceptions open the Exception tab automatically. Parameters and stack
  traces use a fixed-width font and horizontal scrolling.
- Pause to inspect a frozen snapshot; Resume includes newly captured events. Turn off
  **Follow latest** to retain your scroll position while messages arrive.

The default filter is INFO. Severity colors distinguish trace (gray), debug (blue),
info (light), warning (amber), error (red) and critical (purple). Filtering affects the
view only; DEBUG and TRACE remain captured. The dock shows at most the latest 2,000
matching rows and reports matching/retained/evicted counts.

Selecting a ToolBlock shows that wrapper block's events. Enter it and select a child
to inspect the child. Parallel part instances of the selected logical child share one
view, with their full runtime paths, tickets and invocation IDs available in Details.
Identically named tools or repeated child IDs in different ToolBlocks remain isolated
by logical path. Diagnostic severity is separate from vision pass/fail: logging an error
that a tool handles does not automatically fail its inspection.

## Per-block logger API

Every block owns `block.logger`, including standalone blocks. Use it in custom tools:

```python
self.logger.trace("Entering detector")
self.logger.debug("Checking %s candidates", len(candidates))
self.logger.info("Measurement complete", parameters={"score": 0.94, "count": 3})
self.logger.warning("Low contrast")

try:
    run_optional_measurement()
except ValueError:
    self.logger.exception("Optional measurement failed")
```

The logger supplies `trace`, `debug`, `info`, `warning`/`warn`, `error`, `critical`,
`exception`, and `log(level, message, *args, ...)`. Levels accept the names above or
Python logging's numeric constants (TRACE is 5). `%s`-style positional formatting is
supported. `exc_info=True`, an exception object, or an exception-info tuple can attach
traceback details to a log call. `logger.exception` defaults to the current exception.

Without explicit `parameters=`, logging captures a private snapshot of the block's
current parameters, including effective recipe overrides during processing. Providing
`parameters=` supplies the diagnostic's own structured parameter object. Invocation,
ticket and parent-ticket context are inherited inside `process`; use `values={...}`
to provide packet context when logging outside an invocation. `block.logger.snapshot()`
returns only that logical block's retained events, including automatic lifecycle events.

Standalone blocks have a bounded local store. Adding them to a graph transfers their
early messages to the common job collector. Graph/worker clones have their own logger
handles and use the shared job store, preserving histories across F5 and Stop/Resume.
Existing `report_event` markers also appear in the diagnostic list. Lifecycle input and
registration events are TRACE, starts and normal queue backpressure are DEBUG, successful
finishes are INFO, failures/rejections/contention are WARNING, and exceptions are ERROR.

## Exceptions and retention

Automatic execution, acquisition and per-part worker exception handlers include exception
type, message and the formatted traceback, including chained causes. Parameter snapshots
are captured at the event time. Error-finish rows also carry the execution exception for
inspection alongside the final timing/result. Tracebacks retain text, without keeping
Python frames or their local image arrays alive. A handled/logged exception can coexist
with a passing tool result.

The diagnostic fields (`severity`, `message`, `parameters`, `exception`) extend the shared
`block_event_v1` records. Existing activity timeline JSON exports and headless `--report`
receipts include them. See [activity events and timelines](activity-timelines.md).

Histories are bounded runtime data. The job collector retains up to 20,000 events and
approximately 16 MiB; a standalone logger retains up to 2,000 events and 4 MiB. Parameter
snapshots are capped at approximately 16 KiB, messages at 8 KiB, and traceback text at
32 KiB, with explicit truncation/omission markers for oversized content. Nested parameter
collections also have depth/item limits; image/array parameters record shape and dtype
instead of pixels. The overall event byte budget can shorten diagnostics further and
sets `diagnostic_truncated`. Ordinary exception chains are shown in full within these
limits. Opening another project creates a new job history; export records you need to keep.
