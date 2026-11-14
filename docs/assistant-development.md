# Developing applications with a chat

A chat can plan, author and test saved vision projects using the same graph
runtime as the desktop editor. It can discover blocks and parameter constraints,
apply batches of edits, validate connections, run recorded images, inspect ticket
and diagnostic receipts, and export review previews or program documentation.

## Skills and setup

The repository contains two portable skills:

- [opencv-dataflow-apps](../skills/opencv-dataflow-apps/SKILL.md): choose tools,
  plan inspection applications, build graphs, co-work on saved files and validate results.
- [opencv-dataflow-plugins](../skills/opencv-dataflow-plugins/SKILL.md): implement
  missing reusable blocks, parameter validation, result graphics and optional dialogs.

Install the application as described in the root README. To install these skills
for local Codex chats, run from the source checkout:

```powershell
.\.venv\Scripts\python.exe scripts/install_assistant_skills.py
```

This copies the skill folders to `~/.codex/skills` and adds a local installation
reference naming this checkout and its interpreter. Existing different skill files
are preserved and reported as a conflict. `--destination <folder>` supports another
skill directory or a package for a different assistant host. Other ChatGPT hosts
can use the same instructions and JSON API when their environment has access to
this Python installation and project files; this does not install a remote ChatGPT
connector or expose a network service.

Start a subsequent chat with “Use $opencv-dataflow-apps to build an inspection
for …” or give it the skill's `SKILL.md`. Discovery of newly installed skills
depends on the assistant host refreshing its skill catalog.

## A practical workflow

1. Define the acquisition sources, desired detections/measurements, tolerances,
   calibration, final judgment and output consumers. Use the
   [plan template](application-plan-template.md) for substantial applications.
2. Run `opencv-dataflow-api info`, search `catalog`, inspect `describe <type>` and
   find a related public `examples --root <checkout>`. Use working copies rather
   than editing a shared example or the user's only project.
3. Build or change the project with a JSON operation list or Python
   `ProjectSession`. Validate before running. Resolve optional plugin requirements
   explicitly. The project cannot choose a Python plugin file to execute.
4. Run on permitted offline images. Review numeric/data outputs, pass/fail reasons,
   tickets, exceptions, timing and PNG previews. Synthetic checks establish software
   behavior; inspection acceptance still depends on representative labeled images.
5. Hand back the project, plugin/asset requirements, a concise plan, expected outcomes
   and the report/previews. The user opens the saved project normally in the GUI.

The [API reference](../skills/opencv-dataflow-apps/references/api.md) contains exact
commands, response fields, operation schemas and Python examples. A complete
[authoring example](../examples/user_tools/assistant_workflow/README.md) also develops
a small custom block with constrained fields and a controlled failure case.

## Co-work boundaries

Use **Save As** in the editor or `apply --output <new-file.json>` to give the chat
its own working copy. `inspect` returns the SHA-256 revision of the loaded bytes.
In-place CLI edits require that revision through `--expected-revision`; stale
revisions fail without overwriting the project. Python sessions remember their
loaded revision. API writers use an exclusive sidecar lock and recheck the file
immediately before publication.

This is saved-file collaboration. It neither reads unsaved GUI parameter drafts
nor hot-swaps an executing graph. Desktop and third-party writers do not share
the API lock, so a narrow check/replace race remains: keep simultaneous writers on
separate copies and review before reopening the result. An interrupted API writer
can leave a `.project.json.assistant.lock`; check its owner before removing that
specific lock. Do not remove a lock belonging to an active writer.

Inspection, validation and editing do not run `process()`, but constructing an
explicitly trusted plugin can execute its Python code. An explicit run uses the
configured sinks and integrations; it can write files or talk to devices. Run
offline versions of industrial workflows until live actions are requested. Block
exceptions become diagnostics; native crashes, out-of-memory failures and arbitrary
plugin code are not isolated by a separate process. Cancellation is cooperative.
