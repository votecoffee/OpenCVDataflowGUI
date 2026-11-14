# Working in OpenCVDataflowGUI

This Python desktop application builds and runs vision inspection graphs. Start
with [the documentation map](docs/README.md) rather than reading every block module.

- To plan or build a vision application, use
  [opencv-dataflow-apps](skills/opencv-dataflow-apps/SKILL.md). Discover actual block
  IDs, ports and constraints with `python -m opencv_dataflow.assistant_cli` in the
  application environment. The JSON API and `ProjectSession` share the editor's
  graph validation, conversions, scheduler and failure handling.
- To implement a reusable Python tool or dialog, use
  [opencv-dataflow-plugins](skills/opencv-dataflow-plugins/SKILL.md),
  [architecture](docs/architecture.md) and [dialog contracts](docs/plugin-dialogs.md).
- For application requirements and acceptance evidence, start from the
  [application plan template](docs/application-plan-template.md). Reuse existing
  blocks and ToolBlocks before adding a new wrapper. The
  [coverage checklist](docs/vision-coverage-checklist.md) distinguishes implemented,
  partial and planned functionality; a listed OpenCV binding is not proof of a block.
- Keep Qt behavior in desktop modules; graph authoring and execution must also
  work through the headless API. Run focused checks for changed contracts and
  broaden when a shared-runtime concern warrants it. See [Contributing](CONTRIBUTING.md).
- Co-work on separate saved project copies. Revision checks detect changed disk
  bytes; they cannot see an editor's unsaved drafts. The API does not attach to a
  running GUI. Keep stable block IDs and inspect the resulting project before
  handing it back. Opening/validating configuration never means hardware acceptance.
- Public examples belong in their appropriate [example category](examples/README.md).
  Customer images and private examples stay ignored. Explicit plugin loading runs
  Python code; validation must not execute processing or connect to equipment.

Runtime entry points: `core.py` (Block/Port/Result/Registry/Graph), `blocks.py`
(registry), `assistant_api.py` (working sessions), `assistant_cli.py` (JSON CLI),
`runner.py` (human-readable headless runner), `toolblocks.py` (nested graphs),
`plugins.py` (trusted loaders). These are under `src/opencv_dataflow/`.
