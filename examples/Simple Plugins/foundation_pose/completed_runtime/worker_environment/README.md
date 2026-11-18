# Qualified isolated Windows worker

This snapshot records the 16 exact third-party versions used for the Windows x64,
CPython 3.13.9 CPU validation. `requirements.validation.txt` pins artifact hashes;
`requirements.in` records the selected versions. Model weights stay external.
This worker does not need Qt or the host application installed.

From this example folder, choose a new worker directory:

```powershell
uv venv .local/foundation_cpu --python 3.13.9
$worker = ".local/foundation_cpu/Scripts/python.exe"
uv pip sync --python $worker --require-hashes --only-binary :all: worker_environment/requirements.validation.txt
uv pip install --python $worker --no-deps releases/opencv_simple_foundation_pose-0.1.0.dev4-py3-none-any.whl
uv pip check --python $worker
```

Sync removes packages outside its lock, so use it in this fresh worker before
installing the custom wheel. A later sync requires reinstalling that wheel.
Select the worker and the separately verified ONNX model directory in the native
Runtime dialog. Installation does not change the app's current runtime profile,
inspect a capture or establish pose/robot accuracy. The explicit plugin loads
the matching checked code into its subprocess without installing heavy inference
dependencies into the host.

Retain the code notices and dependency/model terms in `../THIRD_PARTY.md` and
the wheel. Requalify changed Python/platform/dependency/model combinations; these
hashes are not a claim that other targets were tested.
