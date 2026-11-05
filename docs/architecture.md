# Architecture and extension contracts

## Blocks and results

The root graph owns a shared [recipe catalog](recipes.md). Immutable revisions
hold settings and sealed pattern assets; frames carry compact `recipe_ref` values.
Bound parameters/models are temporary invocation overrides, restored in `finally`.
Captured frames resolve their pinned revision rather than mutable global state.
Nested/parallel graphs share the catalog, with parallel recipe activation disabled.

`Block` is the extension boundary. Subclasses declare `type_id`, `title`, `category`,
`inputs`, `outputs`, and JSON-compatible `defaults`. The registry rejects duplicate
identifiers. Parameters configure a block; ports carry values between blocks.

`process(inputs)` returns a `Result` containing exactly the declared outputs. Image
ports accept nonempty NumPy arrays, numeric ports accept numbers, and data ports
accept JSON-compatible structures. An `any` input accepts any upstream port kind.
Built-in processing expects OpenCV-compatible images. Ordinary Grayscale interprets
tagged color formats and supports uint8/uint16/float32; Canny requires uint8
grayscale. Invalid native inputs become
visible block errors rather than implicit color conversions.

Image-file inputs additionally support explicit color encodings and pixel depths.
Tagged `ImageArray` values preserve encoding through snapshots; built-in image
consumers interpret those tags before native processing. Color interpretation does
not imply automatic depth conversion. See [image-formats.md](image-formats.md).

Results also carry a pass/fail flag, optional score in [0, 1], message, and a map of
named graphic images. Graphics are diagnostic visualizations; declare an image
output as well if downstream tools should consume one. Inspection fail means a
valid negative result, distinct from execution error. The node header indicates
state; score adjusts its brightness. Text provides the same state information.

The base executor captures `last_inputs`, `last_result`, `last_output_images`,
`last_graphics`, `state`, `error`, and `elapsed_ms`. Runs reset scheduling state;
failure-gated inputs retain their last processed previews when an event is rejected.
ticket counters and bounded control-flow buffers survive until reset/configuration changes.
Input snapshots and output copies isolate sibling branches and retained histories
from mutation in another block. This favors debugging clarity over throughput;
only the latest processed invocation's images are retained. Multiple stream emissions from that run are retained
for downstream execution and nested boundaries. Implement `close()` to release optional resources.

Sources with `input.*` IDs receive runtime-managed ticket terminals/settings. Tickets
are payload metadata, not part of image encoding or spatial coordinates. A Result
can mark an inactive output `NO_VALUE` or supply explicit `emissions` as lists of
ValuePackets. Ordinary plugins still return one value per declared port. Buffered
operators implement `reset_runtime` and `copy_runtime_from`; queues never enter
project serialization. See [control flow](control-flow.md) for identities, bounds,
FIFO expectations and fan-in policies.

## Scheduling and G-API

[Coordinate inputs](input-coordination.md) is an explicit buffered arrival policy.
Its idle polls check ingress-time deadlines without blocking the graph, including
inside ToolBlocks. Ordinary nodes retain positional/broadcast stream scheduling;
sources remain serial and a blocked source prevents deadline service until it
returns. This is not independent acquisition or a hard real-time timer service.

The graph rejects cycles, self-connections, multiple producers for one input,
unknown ports, hidden ports, and incompatible types. Fan-out is supported. Runtime
execution follows topological order. Missing required inputs fail that block;
dependent blocks are skipped after upstream errors. Independent branches continue.
An inspection that returns `passed=False` still feeds downstream result writers.
Port exposure controls editor visibility only; hiding a required port does not
make it optional. Connected ports cannot be hidden until the wire is removed.

Default Grayscale (`cv.gray`), Gaussian blur (`cv.gaussian_blur`) and Canny
(`cv.canny`) execute ordinary OpenCV calls. Shipped projects use these observable
blocks. The legacy `gapi.gray`, `gapi.blur` and `gapi.canny` IDs remain loadable
under **Advanced > G-API**; `GAPIBlock` caches one native `cv2.GComputation` per
parameter set without an eager fallback. The Python orchestration graph is not
fused. [Compiled ToolBlocks](opencv-backend.md#compiled-toolblocks-future) are
future opt-in scope, with selected diagnostic outputs and a measured-benefit gate.

The desktop worker owns a separate execution graph. After a run, it signals the UI
and waits while Qt copies a stable result snapshot into the display graph. Qt never
paints or configures the worker's mutable block objects. Editing is disabled during
execution. Continuous execution uses one worker and never overlaps graph runs.
An explicit Split can opt a graph into 1–8 branch workers. Independent ready nodes
execute concurrently, while source acquisition, buffered fan-in and ToolBlock
boundaries stay on the owning execution thread. ToolBlocks can enable their own
internal branch workers. Unbatch expands entries serially through each node;
singleton ancillary inputs can broadcast, while mismatched stream lengths require
explicit join/batch operations. Inactive paths do not execute; waiting queues are
normal runtime states. UI snapshots also copy ticket clocks and transient queues
so repeated Run once commands continue the same stream session.

`flow.inspect_each` owns a separate bounded part-job pool and an embedded inspection
ToolBlock definition. Each detected pose receives an isolated clone; the parent
emits aggregate data after jobs settle. Project load restores that definition using
the active registry, with the existing nesting limit and explicit plugin policy.
The GUI can navigate the inspection graph and retain one part's internal snapshots.
See [parallel inspection](parallel-inspection.md) for early-stop and coverage rules.

Image queues own a frame lease until the consumer graph iteration exits, including
error/cancellation paths. A queue can retain incoming serial emissions and return
one image; queue-containing ToolBlocks preserve that gate across their public
boundary. Captured upstream snapshots can supply an iteration without rerunning
its source. In background mode a private producer graph runs on an acquisition
thread, while consumer graph runs remain serialized. Queues synchronize storage
and snapshots with a condition variable. The UI receives a final queue snapshot
after joining acquisition so Stop/Resume preserves captures accepted after its
last display update. See [input queues](input-queues.md) for limits and lifecycle.
Stopping is cooperative between blocks; it cannot interrupt a blocking camera
driver, OpenCV call or custom Python function. Closing waits for that work to return.
Camera handles are reused within a continuous session and released on worker exit.

## Projects and plugins

Version 1 JSON projects persist type identifiers, instance IDs, names, parameters,
positions, exposed ports, preview choice, and connections. They omit frames, history,
native camera handles, and executable code. Load validates schema version, types,
ports, positions, and graph topology before replacing the current project. Writes
use a neighboring temporary file and replace the destination on success. Treat
project paths and parameters as configuration that may point to input/output files.

Python plugin modules export `register(registry)` and add `Block` subclasses. Load
plugins explicitly from a trusted local `.py` file; registration is staged so a
failure does not partially update the registry. This is not process isolation or a
security sandbox. Extensions can use the full Python environment. A project with
unavailable plugin types fails clearly; load the plugin and reopen the project.
Plugins are session-local, and projects never execute embedded scripts.

ToolBlocks implement the same base block contract with an embedded graph and typed
public-terminal bindings. Projects embed their definitions; reusable definition
files can be exported/imported separately. Execution, failure handling, cancellation,
cleanup and display-result copying recurse through nested graphs. The same canvas
edits each graph level, while save and run actions retain root-job scope. See
[toolblocks.md](toolblocks.md) for contracts, interface editing and reuse semantics.

## Output behavior

Image and XML writers use exclusive creation to preserve existing files. CSV writes
one JSON payload column and appends records under a fixed `payload` header. SQLite
uses a local `results` table with an integer primary key and JSON payload text;
inserts are parameterized. XML creates a `results/payload` document containing JSON.
These are intentionally basic sinks rather than a user-defined reporting schema.
Each output block executes for accepted input events; configure it accordingly.
Graph runs are not transactions: earlier external writes remain if a later block fails.

## Workflow references

- [Cognex QuickBuild](https://docs.cognex.com/vpromx_1000/web/EN/visionpro/Content/Topics/users-guide/quickbuild/quickbuild-intro.htm):
  visual prototyping and reusable ToolBlocks.
- [MVTec MERLIC connections](https://www.mvtec.com/doc/merlic/26.10/manual/en-us/Content/Creator/Merlic_concepts/Connections/connections.html):
  connected tool flow showing image-processing dependencies.
- [OpenCV Graph API](https://docs.opencv.org/4.x/d0/d1e/gapi.html):
  native operation graphs and execution backends.
- [Qt Graphics View](https://doc.qt.io/qtforpython-6.8/overviews/qtwidgets-graphicsview.html):
  desktop scene, item, and view model.

See [ticket results](ticket-results.md) for per-connection outcomes, per-input
failure gates, the shared thread-safe event consumer and explicit final judgments.
`current_emissions` describes this iteration separately from retained preview data.
Only a gate's accepted invocations update its retained inputs/results; an inactive
gate's retained `last_result` must never be used to infer new stream events.

These inform workflow expectations; this independent prototype does not implement
Cognex, HALCON, or MERLIC compatibility or their commercial tool catalogs.
