# Architecture and extension contracts

## Blocks and results

`Block` is the extension boundary. Subclasses declare `type_id`, `title`, `category`,
`inputs`, `outputs`, and JSON-compatible `defaults`. The registry rejects duplicate
identifiers. Parameters configure a block; ports carry values between blocks.

`process(inputs)` returns a `Result` containing exactly the declared outputs. Image
ports accept nonempty NumPy arrays, numeric ports accept numbers, and data ports
accept JSON-compatible structures. An `any` input accepts any upstream port kind.
Built-in processing expects OpenCV-compatible images (normally uint8); grayscale
expects BGR input and Canny expects grayscale input. Invalid native inputs become
visible block errors rather than implicit color conversions.

Results also carry a pass/fail flag, optional score in [0, 1], message, and a map of
named graphic images. Graphics are diagnostic visualizations; declare an image
output as well if downstream tools should consume one. Inspection fail means a
valid negative result, distinct from execution error. The node header indicates
state; score adjusts its brightness. Text provides the same state information.

The base executor captures `last_inputs`, `last_result`, `last_output_images`,
`last_graphics`, `state`, `error`, and `elapsed_ms`. Every run clears previous results.
Input snapshots and output copies isolate sibling branches and retained histories
from mutation in another block. This favors debugging clarity over throughput;
only the latest run is retained. Implement `close()` to release optional resources.

## Scheduling and G-API

The graph rejects cycles, self-connections, multiple producers for one input,
unknown ports, hidden ports, and incompatible types. Fan-out is supported. Runtime
execution follows topological order. Missing required inputs fail that block;
dependent blocks are skipped after upstream errors. Independent branches continue.
An inspection that returns `passed=False` still feeds downstream result writers.
Port exposure controls editor visibility only; hiding a required port does not
make it optional. Connected ports cannot be hidden until the wire is removed.

`GAPIBlock` builds and caches a native `cv2.GComputation` per parameter set. Current
native operations are BGR-to-gray, Gaussian blur, and Canny. The orchestration DAG
is a Python graph containing those native computations alongside eager OpenCV,
custom Python, acquisition and output operations. There is no silent eager fallback
and no claim that the full graph is fused or compiled. Shared-subgraph compilation
is future work.

The desktop worker owns a separate execution graph. After a run, it signals the UI
and waits while Qt copies a stable result snapshot into the display graph. Qt never
paints or configures the worker's mutable block objects. Editing is disabled during
execution. Continuous execution uses one worker and never overlaps graph runs.
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

## Output behavior

Image and XML writers use exclusive creation to preserve existing files. CSV writes
one JSON payload column and appends records under a fixed `payload` header. SQLite
uses a local `results` table with an integer primary key and JSON payload text;
inserts are parameterized. XML creates a `results/payload` document containing JSON.
These are intentionally basic sinks rather than a user-defined reporting schema.
Each connected output block executes every iteration; configure it accordingly.
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

These inform workflow expectations; this independent prototype does not implement
Cognex, HALCON, or MERLIC compatibility or their commercial tool catalogs.
