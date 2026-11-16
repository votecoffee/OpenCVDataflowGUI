# Tested Windows CPU worker environment

These locks retain explicit wheel URLs, sizes and SHA-256 hashes for Windows x64
CPython **3.13.9**. `pylock.runtime.toml` has 17 third-party packages; the separate
24-package validation lock adds pytest/SciPy. The original model, object mapping
and fit gates are unchanged. PyTorch stays in the external worker.

Use the separately supplied `opencv-simple-gdrnpp` and original geometry package
wheels from the same source release. Set `$gdrnppWheel` and `$geometryWheel` to
their filenames, then run from the app checkout root into a new worker directory:

```powershell
uv venv .local/gdrnpp_reproduced --python 3.13.9
$worker = ".local/gdrnpp_reproduced/Scripts/python.exe"
$envspec = "examples/Simple Plugins/gdrnpp/runtime_environment"
$lock = "$envspec/pylock.runtime.toml"
uv pip sync --python $worker --require-hashes --only-binary :all: --strict $lock
uv pip install --python $worker --no-deps $gdrnppWheel $geometryWheel
uv pip check --python $worker
& $worker "$envspec/check_environment.py"
```

Tested installer: uv 0.12.19; its support for the standard
[pylock format](https://packaging.python.org/en/latest/specifications/pylock-toml/)
is a preview feature. The explicit platform marker rejects other targets.
Sync removes unlisted packages: run it before installing the two source wheels,
only in a dedicated worker. Reinstall those wheels after any later sync. These
dependency locks do not pin the separately supplied custom source code.

The checker reads interpreter/version metadata without importing Torch/OpenCV or
loading a model. A compatible environment is separate from model readiness and
inspection acceptance. Explicitly select this worker using **Choose runtime** in
Setup; no user runtime profile is changed by installing it. Model weights still
need their separately documented, verified external folder.

Two fresh non-editable installations reproduce the frozen independent T-LESS
pose, fit, decision, overlay pixels and native XYZ/RGB arrays. The validation
environment passes 66 worker tests, with two host-only module skips and four
subtests. The runtime-only environment passes two actual app CLI runs and needs
no pytest, SciPy or FoundationPose neural weights. Its offline install used an
already populated uv wheel cache; a new machine needs those wheel downloads or
an explicitly supplied cache. Incorrect artifact hashes and unsupported targets
are rejected. Saved qualification, exact source-module hashes and wheel hashes
are under `validation/`. No new cycle-time, broad accuracy or robot claim is made.

`generate_locks.py NEW_DIRECTORY` proposes fresh locks using the included inputs
and reviewed constraints; it installs nothing. A changed environment needs new
clean-install tests and frozen-result comparison before replacing these files.
The source distribution includes build instructions and the complete test suite.

Credits remain in [THIRD_PARTY.md](../THIRD_PARTY.md): the unchanged real capture
is T. Hodan et al., T-LESS, WACV 2017,
[CC BY 4.0](https://cmp.felk.cvut.cz/t-less/). Pose labels never enter inference.
Weights remain external.
Automatic preparation of actual app execution instances is still pending the
host resource lifecycle hook; these installation checks do not inspect images.
