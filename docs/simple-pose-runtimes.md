# FoundationPose and GDRNPP runtime setup

The app loads these blocks automatically. Their neural inference runs in a
separate Python environment with separately downloaded, checksum-verified weights.
Opening a project or viewing saved results does not run inference.

## FoundationPose

Open `run_project.json` in either
[held_out_frame13](../examples/Simple%20Plugins/foundation_pose/held_out_frame13/README.md)
or [wrong_object](../examples/Simple%20Plugins/foundation_pose/wrong_object/README.md).
Select the pose block, open its configuration popup, then **Runtime on this machine…**.
Choose the prepared isolated Python executable and the folder containing
`refiner_net.onnx` and `score_net.onnx`. Choose the revision subfolder itself,
not a parent containing revision folders. The supported weights revision is
`18d8309afc9790cddc03a1d50bc69954dc058693`.

These choices persist in
`%LOCALAPPDATA%/OpenCVSimpleTools/foundationpose-runtime.json`. Optional environment
overrides are `SIMPLE_FOUNDATION_POSE_PYTHON` and `SIMPLE_FOUNDATION_POSE_MODELS`.
Executable paths are machine settings and cannot be supplied by project JSON.

The two examples run actual ONNX inference on retained RGB-D images, with supplied
single-object masks and coarse pose proposals. Both return **FAIL** under the
saved fit limits: each produces two rejected hypotheses and no accepted pose.
That is an inspection result; a red validation/exception message is an execution
problem. These examples do not establish general pose accuracy.

## GDRNPP

Open `run_project.json` in
[accepted_parts](../examples/Simple%20Plugins/gdrnpp/accepted_parts/README.md),
[rejected_part](../examples/Simple%20Plugins/gdrnpp/rejected_part/README.md) or
[live_review](../examples/Simple%20Plugins/gdrnpp/live_review/README.md).
In the pose block's setup popup, choose the isolated worker with **Choose runtime**
and select the verified checkpoint folder. This folder contains `checkpoint.json`
and its pinned `model_final_wo_optim.pth` for T-LESS object 6.

The runtime and a browsed, verified default model folder are remembered in
`%LOCALAPPDATA%/OpenCVSimpleTools/gdrnpp-runtime.json`. A block's explicit
`models_dir` takes precedence over `SIMPLE_GDRNPP_MODELS`, then the saved local
default. Projects with an empty model folder can therefore run on a configured
machine without embedding that machine's paths in shared examples.
`SIMPLE_GDRNPP_PYTHON` overrides the local interpreter selection.

The accepted-parts example produces five proposals with one accepted pose; the
rejected-part example produces one proposal with none accepted. All candidates
remain available for review. A rejected candidate is distinct from an exception.

## Environments and saved reviews

The main app uses Python 3.14 and OpenCV 5. The prepared pose workers use isolated
Python 3.13.9 environments for their tested ONNX/Torch dependencies. These workers
do not replace the app's Python or OpenCV. GDRNPP's
[locked environment and checker](../examples/Simple%20Plugins/gdrnpp/dev/runtime_environment/README.md)
describe reproducing its dependencies; install the matching current adapter and
FoundationPose helper wheels into that dedicated environment with `--no-deps`.
Model weights remain external and are verified before inference.
The [GDRNPP dev3 source](https://github.com/votecoffee/OpenCVSimpleTools/tree/5bb7d36/GDRNPP)
contains the adapter and model-folder setting.

Use `review_project.json` for file-only results without a model or worker.
It displays recorded results and does not rerun inference. Runtime/model choices
are local setup changes; project training edits still need **Stage** and **Apply**.
Restart the app after updating bundled adapters so existing Python imports and
workers use the new version.
