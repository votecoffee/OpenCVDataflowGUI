# Author an inspection with the assistant API

This example combines a reusable Python plugin with API-authored connections,
drawable result graphics and an explicit final ticket judgment. The input is an
original MIT synthetic image; no customer or third-party image assets are needed.

From the source checkout, load
[`mean_inspection.py`](../../../skills/opencv-dataflow-plugins/assets/mean_inspection.py)
using **File → Load Python plugin…**, then open
[`assistant_workflow.json`](assistant_workflow.json) and run once. The mean
intensity falls inside the configured 20–100 DN range; the measurement and final
judgment pass. Select **Overlay result graphics** for the annotated result.

The custom fields demonstrate validation: minimum/maximum must be between 0 and
255 and minimum must not exceed maximum. Comparison uses a dropdown. These are
8-bit grayscale sample units; the primary image retains its original color type.
Mask/region inputs optionally restrict which pixels contribute to the mean.

## Build and replay without the GUI

The [`operations.json`](operations.json) file is a complete declarative build.
Use a new output path; the API preserves existing projects and reports.

```powershell
$python = ".\.venv\Scripts\python.exe"
$plugin = "skills/opencv-dataflow-plugins/assets/mean_inspection.py"
New-Item -ItemType Directory -Force .local | Out-Null
& $python -m opencv_dataflow.assistant_cli --plugin $plugin describe example.mean_inspection
& $python -m opencv_dataflow.assistant_cli --plugin $plugin new --operations examples/user_tools/assistant_workflow/operations.json --output .local/assistant-inspection.json
& $python -m opencv_dataflow.assistant_cli --plugin $plugin validate .local/assistant-inspection.json
& $python -m opencv_dataflow.assistant_cli --plugin $plugin run .local/assistant-inspection.json --report .local/assistant-pass.json --artifacts .local/assistant-pass-previews --summary
```

Create `.local` first if it does not already exist. Each command returns one JSON
response. Read the report's mean measurement, ticket judgment, timings and
diagnostics, and use the preview manifest to find the labeled image.

[`fail_edits.json`](fail_edits.json) is a valid negative inspection: it changes the
accepted mean range to 200–255 DN. Apply to a new copy, then run it:

```powershell
& $python -m opencv_dataflow.assistant_cli --plugin $plugin apply .local/assistant-inspection.json --operations examples/user_tools/assistant_workflow/fail_edits.json --output .local/assistant-fail-inspection.json
& $python -m opencv_dataflow.assistant_cli --plugin $plugin run .local/assistant-fail-inspection.json --report .local/assistant-fail.json
```

The second run returns exit **1**, `passed=false` and a `mean_intensity` failure
reason; this is an inspection result, not an exception. In-place edits require the
file revision returned by `inspect`. `--dry-run` previews edits without saving.

[`build_example.py`](build_example.py) regenerates these tracked files
deterministically using `ProjectSession`. See the
[assistant guide](../../../docs/assistant-development.md) and
[API reference](../../../skills/opencv-dataflow-apps/references/api.md).
