# Locator capture controls

Load the parent `plugin.py`, then open either saved control:

| Project | Expected result |
| --- | --- |
| [Matching cloud metadata](metadata_only/metadata_only.json) | No explicit locator ID input. Matching scene/depth cloud IDs bind the report; the strict adapter converts one proposal and the jaw planner retains zero eligible alternatives. |
| [Conflicting pair](mismatched_pair/mismatched_pair.json) | The depth cloud declares a different capture ID. The locator errors before loading its model or starting matching; no current pose or jaw result is emitted. |

These reuse the real first-return frame, unchanged recipes and gates from the
[composed locator example](../locator_chain/README.md). The conflicting pair is
an authored metadata control over copied geometry, not a second measured capture.
Matching ID strings are caller declarations; they do not establish sensor or
robot-time synchronization. See the [capture contract](../LOCATOR_CAPTURE.md).

`checks.json` records run and reopen checks. `portable-verification.json` records
reopen and Save As outcomes with byte-preserved depth, clouds, records and recipes.
From the parent example folder, run:

```powershell
python build_capture_controls.py . C:\path\to\new-checks --verify-existing
```

Source and derived T-LESS assets are **CC BY 4.0**, separate from MIT example code.
Preserve [attribution and source pins](../tless/ATTRIBUTION.md) and
[frame transformations](../locator_chain/README.md#source-changes-and-attribution).
The added capture ID conflict is authored for this software check. No endorsement
by the dataset authors or physical grasp qualification is implied.
