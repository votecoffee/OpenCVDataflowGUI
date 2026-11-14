---
name: opencv-dataflow-plugins
description: Develop reusable Python block plugins and optional configuration or training dialogs for OpenCVDataflowGUI, including ports, validation, ticketed results, graphics, persistence and offline checks.
---

# OpenCV Dataflow plugins

Use the user's application checkout and environment. If
`references/installation.md` is present, read it for local paths. Confirm the
environment with `python -m opencv_dataflow.assistant_cli info`.

Before adding a tool, search the live catalog and inspect related block schemas.
Prefer a ToolBlock for reusable graph composition. A plugin is appropriate when
an algorithm, external library, custom data contract or training workflow needs
its own implementation. Keep interfaces general enough for the actual use cases;
avoid adding speculative settings.

Read [references/plugin-contract.md](references/plugin-contract.md) for a complete
block, extension points and failure/validation rules. In a checkout,
`docs/architecture.md` defines shared runtime behavior, `docs/plugin-dialogs.md`
defines popup/capture/worker APIs, and `docs/toolblocks.md` covers nested graphs.
Read only the contracts relevant to the change.

## Implement

Declare a stable namespaced `type_id`, typed `Port`s, JSON-compatible defaults,
enum choices, help and `NumericRange`s. Use `validate_parameters` with
`ValidationError(..., field_names)` for coupled controls. Return a `Result` with
every declared output, separating valid inspection failures from exceptions.
Let the base executor own timing, history, tickets, exception diagnostics and
connection validation. Do not override `execute` or call `process` in application
code. Use `self.logger` for diagnostics.

Use drawable result objects for image geometry; use diagnostic raster images for
plots/maps. Respect `generate_graphics`. Preserve image encoding and spatial/
pyramid context; geometric resampling needs the corresponding coordinate mapping.
Declare project-relative files and sealed training assets using the existing
contracts. Opt into preview or parallel execution only after checking state,
resources and side effects. Stateful tools require lifecycle behavior, not a
`parallel_safe` declaration alone.

Load a local plugin explicitly with `--plugin <trusted.py>` or the editor's Load
Python plugin action. Saved project JSON must not select executable plugin files.
Export `register(registry)`; avoid import-time acquisition, training or device I/O.

## Verify and hand off

Use the application API to describe the loaded type, build a minimal graph, run
realistic valid/invalid inputs, and save/reopen it. Compare numerical behavior to
an independent/native reference when useful. Check enum/range/cross-field errors,
inspection fail vs exception behavior, payload/graphics provenance and resource
cleanup for the features used. Test dialogs offscreen where possible.

Keep customer images private. Put a focused example and README in the appropriate
example category and record attribution when needed. Deliver the plugin source,
installation/loading instructions, project and focused validation evidence.
