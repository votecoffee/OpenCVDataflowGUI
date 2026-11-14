# Python block plugin contract

The portable [mean inspection template](../assets/mean_inspection.py) is a complete
plugin. Copy it to a working file, change its stable type ID and adapt its actual
algorithm. It demonstrates optional mask/region inputs, original image output,
enum/range/cross-field validation, pass/fail reasons, logging and drawable results.
The checkout's `examples/user_tools/assistant_workflow/README.md` creates and
replays a saved graph through the assistant API.

## Core interface

Import `Block`, `Port` and `Result` from `opencv_dataflow.core`. Export
`register(registry)` and call `registry.register(YourBlock)`. IDs must be unique;
the registry stages plugin registration so a load failure does not partially add
classes. A local file is selected explicitly by the user/client, never by project
JSON. Imports/constructors should not start acquisition, training or device I/O.

| Extension | Contract |
| --- | --- |
| `type_id`, `title`, `category`, `block_type`, `subtype` | Stable persistence ID and toolbox organization; external Python plugins appear under User |
| `inputs`, `outputs` | Tuples of `Port(name, kind="image", required=True, visual_type="", required_when_connected=False)` |
| `defaults` | JSON-compatible parameter values; undeclared keys are rejected |
| `parameter_choices` | Key → readable enum values; creates inspector dropdowns |
| `parameter_ranges` | Key → `NumericRange(minimum, maximum, whole, odd, zero_allowed)` |
| `parameter_help`, `input_help` | Units, conditional behavior and constraints |
| `validate_parameters(parameters=None)` | Call base validation, then check cross-field rules without processing images |
| `process(inputs)` | Return `Result`; execution uses the base `execute` wrapper |
| `close()` | Release native handles/workers/resources; safe when called repeatedly |

Port kinds include `image`, `depth`, `pyramid`, `cloud`, `graphics`, `any`, `data`,
`number`, `integer`, `float`, `boolean`, `text`. Pose/region/collection schemas may
travel on `data` ports with semantic visual types. Use the receiver's schema checks;
JSON-compatible alone does not imply semantic compatibility. Connections allow
supported conversions and validate their actual values at execution.

`Result(outputs, ...)` contains exactly the algorithm's declared outputs. The
runtime adds ticket/result-graphics terminals where applicable. Optional absent
stream values use `NO_VALUE`, and multi-emission blocks use `ValuePacket` emissions
with lifecycle contracts described in the checkout's control-flow documentation.
Do not use a retained `last_result` as evidence of a new emission.

Use `passed=False` for a valid inspection failure. Set optional normalized
`score` in [0,1], `message`, JSON `event_data`, `fail_reasons` (codes/messages),
and `final_judgment` only when the tool owns a final ticket decision. Raise
`ValidationError(message, *field_names)` for invalid settings/input and an ordinary
exception for unexpected algorithm failures. The executor records state, timing,
traceback and per-block logger diagnostics and gates downstream work. Do not catch
an error and silently return a passing result.

## Images, payloads and graphics

The executor supplies private input snapshots and manages ticket/outcome/spatial
metadata. `self.input_payloads[port]` holds the processed invocation's payload.
`Result.payloads` supplies deliberate output metadata overrides. A source using an
`input.*` ID receives runtime-managed ticket settings and ports. Preserve ticket
correlation when aggregating or emitting multiple events.

Interpret tagged images through `images.image_format`/`opencv_image` or helpers in
`processing.py`; do not assume every color array is BGR or every float is uint8
scaled. `selection.selection_mask(inputs, image)` intersects optional masks and
regions and validates their frame. Declare what selected processing means (which
pixels contribute, which are written, and how boundaries behave).

Use `GraphicDrawing` and `graphic_object` from `graphics_layers` for lines, labels,
markers and polygons. Drawings identify the input source by default; use
`from_output=True` for an output image's coordinates. The runtime exposes the
hidden `result_graphics` connector and maps matching frames when composing them.
`Result.graphics` is for raster plots/response maps. Avoid baking result geometry
into primary numerical image outputs. Skip optional graphics when
`self.generate_graphics` is false. A crop/resize/warp needs an explicit spatial
mapping; copying old payloads after moving pixels is incorrect.

## State, files and execution options

`preview_safe=True` allows isolated last-input previews: opt in only for a tool
with appropriate state/resource/side-effect behavior. `parallel_safe=True` permits
independent branch scheduling. It is not permission to share mutable model state
without locks. OpenCV/NumPy native work may release the GIL; pure Python threads
do not guarantee multicore CPU throughput. Use bounded workers for per-part graphs.
`process_inputs_readonly=True` permits lean profile optimizations only when input
arrays remain unchanged. `reset_runtime` and `copy_runtime_from` support stateful
tools; saved projects must omit live queues, camera handles and locks.

Declare project-relative input fields with `project_path_parameters`. Sealed
relative assets use `project_asset_parameters` (base includes `recipe_path`);
override `project_assets()` for dependent files. Save As copies sealed assets
without changing their pinned bytes, rejects conflicts and rebases ordinary input
references. JSON parameters and `serialize_extra`/`restore_extra` hold persisted
state. Test saved/reopened trained models rather than assuming a live cache persists.

## Optional popup editors and training

Set `plugin_dialog_label`, implement `create_plugin_dialog(parent, parameters,
context)` and optionally `plugin_dialog_modeless`. Edit the supplied draft;
acceptance stages values, and Apply configuration commits them. Keep GUI work on
the GUI thread and compute in the existing worker mechanism. Use the context's
snapshot methods rather than reading a running graph from a worker.

Read `docs/plugin-dialogs.md` in the source checkout before implementing these
APIs: output subscriptions, image pickers, processed-input bundles, cloud captures,
worker lifetime, cancellation and stale publication checks. Snapshot identity
describes an editor publication/processed invocation; it does not prove physical
exposure synchronization between multiple cameras or robot data sources.
For a new/untrained block, `context.snapshot_connected_inputs(ports)` freezes wired
values together into a `PluginConnectedInputs` bundle with publication/snapshot
IDs, per-port payloads/source metadata and a list of absent optional ports. It
does not require a successful run of the target and never executes the graph.
Use the owning GUI thread; hand the private copy to the training worker.

## Verification

Use `--plugin <file>` before `describe`, `new`, `validate` or `run`. Check valid
and negative inspections, enum/range/cross-field errors, dtype/shape behavior,
mask/region semantics, ticket/spatial payloads and graphics-off behavior relevant
to the tool. Save/reopen the graph; test trained assets and cleanup when used.
The host catches processing exceptions but is not a process-isolation sandbox.
