# Paired inspection input shifts

Open `defect_stable.json` and run once. Follow Baseline, Exposure and Pose branches
through the same inspection, then select **Evaluate inspection input shifts**.
Its Report contains per-view scores, label errors, truth IoU and mapped-baseline
agreement. Shift Support is also connected to each inspection's mask.

| Project | Purpose |
| --- | --- |
| `clean_stable.json` | Clean image passes small exposure and two-pixel translation. |
| `defect_stable.json` | All views find the same 64-pixel defect; truth/agreement IoU are 1. |
| `high_depth.json` | Same comparison while preserving original uint16 image depth. |
| `exposure_sensitive.json` | Larger offset causes a false reject on the clean image. |
| `blur_sensitive.json` | Blur changes the known texture and causes a false reject. |
| `pose_sensitive.json` | One-pixel translation changes patch phase and causes a false reject. |
| `loose_threshold.json` | An overly loose threshold accepts known defects; evaluation fails. |
| `mixed_parts.json` | Clean and defective sources have separate baseline/pose pairs. |
| `missing_variant.json` | Incomplete declared comparisons are rejected. |
| `wrong_identity.json` | A valid changed reference revision cannot be mixed into one evaluation. |
| `cropped_truth.json` | Pose crops away the defect; capture rejects insufficient truth retention. |

The sensitivity examples have one passing baseline and one false-rejected shifted
view: 50% false rejects overall, 100% for that shift. `loose_threshold` has 100%
false accepts. These are intentionally failing evaluations, not execution errors.
The final three projects deliberately exercise validation errors.

Scores are raw patch distances, not calibrated confidence. These small synthetic
controls expose evaluation behavior and known model weaknesses. The derived views
are not independent acquisitions, industrial accuracy evidence or camera calibration.

`truth.json` pins images/reference/model assets and the exact shift specifications.
`benchmark.py` independently computes image changes, patch distances and mask metrics;
`benchmark.json` retains the values. `build_example.py` regenerates into an empty
directory using the original adjacent `local_anomaly` assets and preserves existing
assets. Running the saved projects uses only this folder's included files.
See [sources](SOURCES.md) and [contracts](../../../docs/input-shifts.md).
