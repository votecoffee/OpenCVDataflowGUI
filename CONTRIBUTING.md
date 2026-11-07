# Contributing

Install the editable package with the `dev` extra as described in README.md.
Keep UI-independent behavior in the runtime and Qt behavior in the desktop modules.
New blocks declare a stable type identifier, typed ports, JSON-compatible parameter
defaults, and `process(inputs) -> Result`. Add checks for meaningful execution,
serialization, and failure behavior. Avoid tests requiring cameras or network services.

Expose meaningful native algorithm parameters instead of hard-coding them. Declare
enum constants in `parameter_choices` using readable names so the inspector renders
dropdowns; add conditional behavior and units in `parameter_help`. Represent image
arguments as typed ports and leave destination buffers to the runtime. Preserve
saved-project defaults when extending a wrapper. Compare nondefault controls to
native OpenCV calls and test invalid combinations. Regenerate the parameter catalog
with `python scripts/block_parameter_reference.py`, then verify it with `--check`.

For a new OpenCV wrapper, place its implementation file or type ID in the
module map in `src/opencv_dataflow/toolbox.py`. Add an API search name when
the OpenCV function name differs from the block label. Built-in workflows
without a module assignment appear under **Custom**.

Use small, verified commits. Run pytest and Ruff locally. Automated GitHub Actions
are intentionally absent; future CI must use `workflow_dispatch` unless the project
owner requests a different policy. Preserve the existing license ownership.

Plugins run with the current user's Python permissions. Never auto-load code named
by a project file. Do not commit personal inspection images, credentials, device
addresses, reports, `.venv/`, or generated captures. `.local/` is ignored for local work.
