# Saved-project API, version 1

The installed `opencv-dataflow-api` command and
`python -m opencv_dataflow.assistant_cli` are equivalent. On Windows from a source
checkout, use `.\.venv\Scripts\python.exe -m opencv_dataflow.assistant_cli ...`.
On POSIX use `.venv/bin/python`. Commands do not create a QApplication or open
windows. They operate on saved configuration, independently of an open GUI.

## Response and discovery

Every command except `--help` emits one JSON object on stdout:

```json
{"api_version":1,"ok":true,"data":{}}
```

Errors use `{"api_version":1,"ok":false,"error":{"type":"...","message":"..."}}`.
Configuration errors also include `error.issues` with scope, block, fields and
message. `validate` returns its issues under `data`, even when invalid. Plugin
stdout is redirected to stderr. Clients should check the version and exit code:
**0** successful request / passing run; **1** validation or run did not pass;
**2** usage, loading, configuration or I/O error; **3** revision/write conflict.
`ok=false` with `data` is a completed negative result, not a protocol error.

```text
info
catalog --query "GaussianBlur"
describe cv.gaussian_blur
compatibility number integer
examples --root <checkout> --query "histogram"
inspect <project.json>
validate <project.json>
```

`info` reports Python/OpenCV/application versions and the source documentation
path when installed from a checkout. `catalog` returns stable IDs, titles,
categories/subtypes, OpenCV function names and whether the class is already loaded.
It does not import optional providers. A loaded block can still depend on an
optional library/model at execution time. `describe` constructs just the requested
block and returns inputs/outputs, exposure, parameter defaults/choices/help/ranges,
capabilities, asset fields and a full default `template`. Source ticket ports and
the hidden `result_graphics` port are included. Specialized nested blocks are
constructed through the ToolBlock API or an existing definition, rather than an
ordinary default block template.

`compatibility` uses the graph's conversion rules; it cannot prove that a particular
runtime scalar, shape, depth, frame or semantic data schema satisfies the receiver.
`examples` indexes public runnable projects in the checkout's example categories,
excluding private/CognexTools directories. It returns paths, adjacent README and
block IDs without loading plugins or processing images. It is a content index,
not a license audit; read source/license notes before publishing reused images.
In a Git checkout it also honors ignored local files, while including new public
untracked examples. An archive without Git metadata uses the public categories
and known private-directory exclusions.

`inspect` returns the applied project, instance ports for all nested scopes, static
validation and `revision` (SHA-256 of exact loaded bytes). JSON schema version is
currently **1**. The API protocol version is independent of the project schema.

## Authoring operations

Store a JSON **list** of operations in an operations file. Duplicate JSON keys and
nonfinite constants are rejected by the CLI. Explicit IDs make edits and review
repeatable. Parameters omitted from `add` inherit actual block defaults.

```json
[
  {"op":"add","type":"input.demo","id":"source"},
  {"op":"add","type":"cv.gray","id":"gray"},
  {"op":"add","type":"cv.threshold","id":"mask","parameters":{"value":100}},
  {"op":"connect","source":"source","output":"image","target":"gray","input":"image"},
  {"op":"connect","source":"gray","output":"image","target":"mask","input":"image"},
  {"op":"update","id":"mask","fields":{"description":"Segment bright pixels"}},
  {"op":"arrange"}
]
```

```text
new --operations <operations.json> --output <new-project.json>
apply <project.json> --operations <edits.json> --dry-run
apply <project.json> --operations <edits.json> --output <new-working-copy.json>
apply <project.json> --operations <edits.json> --expected-revision <sha256>
```

`new --directory <original-base>` defines the base for project-relative input paths
before Save As. It defaults to the output's parent. Existing output files are
preserved. Parent directories must already exist. `apply` defaults to its input
path. Writing an existing destination requires **that destination's** revision,
including when using `--output`. A new output requires no revision argument.
Dry-run edits an ephemeral session only and reports `saved=false`.

| Operation | Required / optional fields |
| --- | --- |
| `add` | `type`, nonempty `id`; optional `parameters`, `name`, `position` |
| `update` | `id`, `fields`; parameters merge by key, other fields replace |
| `insert` | `block`: complete persisted item, useful for a prepared ToolBlock |
| `replace` | `block`: complete persisted item with an existing ID |
| `remove` | `id`; requires `disconnect:true` if wires would be removed |
| `connect`, `disconnect` | `source`, `output`, `target`, `input` |
| `settings` | `fields`: root/scope settings listed below |
| `arrange` | no additional fields; organizes nodes with existing layout algorithm |

Every operation accepts optional `scope:["outer_block_id","inner_block_id"]` to
address a containing ToolBlock or per-part inspection graph. An empty list is root.
The complete resulting graph is loaded and validated once before replacing the
session graph. Invalid parameters, unknown fields, duplicate incoming wires,
incompatible kinds, invalid exposure and cycles reject the whole batch. A draft
may have unconnected required inputs; `validate` reports those before running.
Image-dependent parameter checks still require execution. Constructors and custom
validators are trusted Python code; only `process()` is guaranteed not to run here.

Allowed update fields: `name`, `description` (plain text), `documentation` (HTML),
`parameters`, `position` `[x,y]`, `node_size` (`small`, `medium`, `large`),
`exposed_inputs`, `exposed_outputs`, `thumbnail`, `preview_output`, `graphics_mode`,
`recipe_bindings`, `recipe_model`, `input_levels`, `input_failure_policies`,
`ticket_settings`. Replace a nested definition through `insert`/`replace` or use
the Python ToolBlock API. Connected ports must remain exposed; a rewire must
explicitly disconnect the old incoming connection first.

Allowed settings: `run_profile`, `branch_workers`, `graphics_mode`, `groups`,
`viewport`, `validation_sets`, `recipes`. These use the existing project schemas.
`run_profile` is `debug`, `balanced` or `lean`; `graphics_mode` is `inherit`, `on`
or `off`. `branch_workers` is `auto` or a supported integer (currently 1–8).
The API does not maintain a second set of scheduler or conversion rules.

Disk saves preserve project-relative inputs and sealed assets through the existing
Save As contract. The API uses a unique temporary file, exclusive new-file
publication, a sidecar writer lock and revision recheck. Existing non-API writers
do not share the lock; use separate working copies for concurrent GUI edits.

## Execution and review

```text
run <project.json> --repeat 2 --report <new-receipt.json> --artifacts <new-directory> --summary
run <project.json> --all-images --max-runs 1000 --profile debug
export <project.json> --format html --output <new-guide.html>
export <project.json> --format xml --output <new-snapshot.xml>
```

`run` is explicit execution, including configured sink/device side effects. It
validates static configuration, then uses Graph/QueuedExecution scheduling.
Options `--profile`, `--result-graphics` and `--branch-workers` override the run
without saving the project. `--all-images` requires a finite recorded source with
`repeat=stop`. Default execution is once. `max-runs` bounds recorded replay and
must be at least `repeat`; a truncated replay is incomplete and returns exit 1.
`--summary` returns counts/status, failed block identifiers and the report/preview
manifest paths instead of embedding the full receipt in stdout. The saved report
still contains all run records. Use this for inexpensive chat feedback, then read
the relevant report entries when investigating a result.

The receipt includes `passed`, `complete`, `stop_reason`, `runs`, ticket results,
activity/diagnostics and active recipe. `stop_reason` is `requested_runs`,
`sources_exhausted`, `max_runs` or `cancelled` (Python callback). Each run contains
block states, score, message/error, timings, numeric/data outputs, payloads and
emission counts, recursively for ToolBlocks. Images/pyramids are excluded from
the JSON sample values. A valid negative inspection is `fail`; exceptions are
`error` with diagnostic traceback information. Waiting/inactive are normal stream
states; check actual counts/judgments against application requirements rather
than treating the runner's success as an inspection acceptance specification.

An artifacts directory contains up to 128 PNGs from the **last available** output
images and diagnostic plots (including nested graphs), plus `manifest.json` with
block/scope, state, output name, payload, source dtype/shape, display size and
retained/emitted status. Drawable layers are overlaid where their source frame
matches. These are the editor's 8-bit display conversions, limited to 1024 pixels
per side. They do not replace original image/depth/cloud exports and are not a
per-ticket filmstrip for every run. Lean mode may retain fewer images. Use Debug
for review. Existing reports/artifact directories/exports are preserved.

Exceptions are caught by the normal block executor. Python and native code remain
in-process; cancellation cannot preempt a blocking native call or arbitrary plugin.
Execution is not a transaction: earlier writes remain if a later block fails.

## Python sessions

```python
from opencv_dataflow.assistant_api import ProjectSession, make_registry

registry = make_registry(plugins=["trusted_tool.py"])
with ProjectSession.open("inspection.json", registry) as project:
    project.apply([
        {"op": "update", "id": "mask", "fields": {"parameters": {"value": 110}}}
    ])
    # A new path creates a review copy; saving in place checks the loaded revision.
    project.save("inspection-review.json")
    report = project.run(repeat=1)
    print(report["passed"])
```

`ProjectSession(registry=None, directory=None)` creates an empty graph.
`open`, `inspect`, `apply`, `save`, `run`, `close` are the public working-session
operations. `apply` resets runtime state by replacing the graph after validation.
`save(path=None, expected_revision=None)` remembers its own saved revision; an
existing different destination requires its revision explicitly.
`run(..., cancelled=callable)` checks cancellation between blocks/iterations.
Always close sessions to release native/plugin resources.

Module helpers: `make_registry(plugins=())`, `catalog(registry=None, query="")`,
`block_schema(type_id, registry=None)`, `validate_graph(graph, require_complete=True)`,
`file_revision(path)`. `RevisionConflict` preserves changed destinations;
`ProjectValidationError.issues` carries structured diagnostics.
`assistant_artifacts.export_previews(graph, directory, max_images=128, max_side=1024)`
exports review images to a new directory.

For advanced composition, `session.graph` exposes the normal `Graph` API:
`registry.create`, `graph.add`, `graph.connect`, `Graph.from_dict`, ToolBlock
`encapsulate`/definitions and recipe APIs. Keep direct graph mutations valid;
operation batches provide the transactional edit boundary. Do not call
`Block.process` directly: use `execute` for isolated block tests and `Graph.run`
or `ProjectSession.run` for applications.

## Plugin selection

The global `--plugin <trusted.py>` option precedes the command and may be repeated:

```text
opencv-dataflow-api --plugin my_tool.py describe user.my_tool
opencv-dataflow-api --plugin my_tool.py run inspection.json
```

Host-known optional integrations resolve by stable type ID when installed. Explicit
local plugin files are never taken from project JSON. Loading Python is trusted
code execution; the API is a local authoring interface, not a sandbox, remote MCP
server or GUI automation service.
