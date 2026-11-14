---
name: opencv-dataflow-apps
description: Plan, build, edit and verify image/video/depth inspection applications in OpenCVDataflowGUI using its block catalog, saved-project API, ToolBlocks and offline run receipts.
---

# OpenCV Dataflow applications

Use the application's installed Python environment and its saved-project API.
If `references/installation.md` is present, read it to locate the checkout and
interpreter. Otherwise locate the user-selected checkout or installed
`opencv-dataflow-api` command. Run `info` to confirm the actual environment and
documentation location. Do not assume that a system Python uses the same OpenCV
or optional plugins as the app.

## Discover and plan

Read [references/api.md](references/api.md) for exact commands and contracts.
Search `catalog --query ...`, then `describe <type>` for real ports, defaults,
enum names, numeric limits and exposed connectors. Optional IDs are discoverable
without loading their packages; resolve only the types needed. The native binding
inventory and roadmap contain unimplemented work, so they are not a block catalog.

Find related projects with `examples --root <checkout> --query ...` and read their
README/license before reuse. Use the checkout's `docs/README.md` to navigate
specific runtime topics. For a substantial application, adapt
`docs/application-plan-template.md`: inputs/encoding/depth, calibration and units,
ticket/recipe correlation, outputs and judgment limits, dataset/labels, graph
proposal, dependencies, acceptance checks and remaining physical evidence.

Reuse available blocks and subgraphs. Name the existing baseline before proposing
a gap. Prefer a ToolBlock for a reusable graph pattern; use a Python plugin when
the missing algorithm needs its own contract. Do not add a new block merely to
duplicate a composition that already solves the task well.

## Build and co-work

Use `new` or `apply` with explicit block IDs, typed connections and the real
parameter keys. Batches are transactional in the working session. `apply --dry-run`
checks without saving. Missing required connections may remain in a draft;
`validate` reports them before execution. Changes to port exposure must accompany
connections to hidden ports. Nested edits use containing block IDs in `scope`.

Work on a separate saved copy when the user is also editing. `inspect` returns a
disk revision; in-place CLI edits require `--expected-revision`. On conflict,
inspect the new file and reconcile the intended changes. Do not bypass the guard
or overwrite unsaved work. The API does not attach to a live GUI or see its drafts.
Keep layout, descriptions, documentation, groups and unrelated settings intact.

## Verify and deliver

Run static validation first. Then explicitly execute permitted offline inputs and
inspect the JSON receipt: block states, actual emissions, scores, numeric/data
outputs, ticket judgments/reasons, exception details and timing. Retained previews
are not evidence of a new output event. Use `--artifacts <new-directory>` for
review PNGs; these are display conversions rather than original high-depth data.
Use `export` for an HTML program guide or XML snapshot.

Graph execution can invoke configured file/device sinks; validation does not
execute processing. Use offline inputs and sinks unless live integration is within
the request. Plugin loading executes trusted Python and is not a sandbox.

Validate a representative pass and fail case, plus meaningful boundary/invalid
inputs for changed controls. Prefer targeted tests and example replay; broaden
when shared runtime contracts change. Deliver the runnable project, asset/plugin
requirements, expected outcomes, review artifacts and any remaining acceptance
limits. Distinguish synthetic/software evidence from optical or machine accuracy.
