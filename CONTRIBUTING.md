# Contributing

Install the editable package with the `dev` extra as described in README.md.
Keep UI-independent behavior in the runtime and Qt behavior in the desktop modules.
New blocks declare a stable type identifier, typed ports, JSON-compatible parameter
defaults, and `process(inputs) -> Result`. Add checks for meaningful execution,
serialization, and failure behavior. Avoid tests requiring cameras or network services.

Use small, verified commits. Run pytest and Ruff locally. Automated GitHub Actions
are intentionally absent; future CI must use `workflow_dispatch` unless the project
owner requests a different policy. Preserve the existing license ownership.

Plugins run with the current user's Python permissions. Never auto-load code named
by a project file. Do not commit personal inspection images, credentials, device
addresses, reports, `.venv/`, or generated captures. `.local/` is ignored for local work.
