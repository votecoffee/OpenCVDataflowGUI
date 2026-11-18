# Monocular depth: explicit quantities and source geometry

These eleven projects demonstrate `ml.monocular_depth`. Nine ship tiny arithmetic
ONNX controls and run immediately. Two use the pinned MIT MiDaS v2.1 Small model
on a real T-LESS RGB-D capture. See [sources and licenses](SOURCES.md) and the
[block contract](../../../docs/monocular-depth.md).

Open a project with the application and select Run Once. In Debug, Preview is
sent to the filmstrip override; the source image remains unchanged. Inspect
Estimate, Valid mask, Info and Receipt in the block results. Lean intentionally
omits diagnostic previews. Relative estimates do not emit the physical Depth
connector, so downstream physical geometry does not run accidentally.

| Project | Purpose |
| --- | --- |
| [relative](relative.json) | Known positive ramp in relative inverse-depth units; no physical depth |
| [metric](metric.json) | Known metric depth and 713 camera-ray XYZ points |
| [inverse_metric](inverse_metric.json) | Interpolate inverse meters, then invert to depth |
| [millimeters](millimeters.json) | Explicit fixed scale to mm, preserving physical meaning |
| [letterbox](letterbox.json) | 31x23 source, 16x16 input, 8x8 head; 589 supported pixels |
| [invalid_hole](invalid_hole.json) | Sixteen invalid model samples remain invalid, with 697 valid pixels |
| [selected](selected.json) | Circular write selection; excluded image remains model context |
| [empty](empty.json) | Off-image selection: zero coverage, ordinary Fail, no model I/O |
| [coverage_alarm](coverage_alarm.json) | Deliberately impossible minimum count: ordinary Fail |
| [native_relative](native_relative.json) | MiDaS RGB/cubic stretch and original source-sized relative output |
| [native_letterbox](native_letterbox.json) | Explicit alternative geometry and conservative padding support |

## Prepare the two native projects

From the repository root in its installed application environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/monocular_depth/prepare_model.py"
```

This explicitly downloads the unchanged 66,764,249-byte upstream ONNX, checks its
pinned SHA-256 and seals two configuration-specific `.local` bundles. Python's
standard library and the application are sufficient; Torch/ONNX authoring packages
are not required. Reuse identical existing bytes without downloading with
`--model "path/to/model-small.onnx"`. Repeating preparation verifies existing
bundles offline and preserves differing existing files. Execution itself never
downloads a model. Sealed Save As includes the prepared model, contract and notices.

MiDaS embeds its RGB mean/std normalization: the wrapper supplies unit RGB with
zero mean/unit std. The model output is **relative inverse depth**, not meters.
Changing the convention to Metric would be an unverified caller declaration.
Do not use this example's relative values as physical XYZ or picking clearance.

## Known geometry and real-depth check

The arithmetic model selects the source red channel: `q = 1 + 2*R/255`.
The metric example declares q in meters; inverse-metric uses `1/q`; the mm
example applies a fixed factor of 1,000. The camera block uses fx=80, fy=90,
cx=15, cy=11, allowing exact independent XYZ checks. These models test conversion
contracts, not learned depth accuracy.

T-LESS scene 6/frame 2 has 387,684 positive depth samples, stored as uint16 with
0.0001 m/sample. Reference depth is a separate block and never feeds the model.
`tless_camera.json` retains the published camera matrix. The benchmark runs the
unchanged model directly and checks source-center resampling independently:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/monocular_depth/benchmark.py"
```

[Recorded results](benchmark.json): near/far ordering is 97.20% on 3,609 eligible
pairs. A diagnostic inverse-depth fit on the left half gives 51.36 mm RMSE on
the right half, worse than the 40.70 mm constant-depth baseline. Both halves
belong to the same acquisition. The fit is evaluation only and is never applied
by the runtime; this is not metric calibration or another-scene validation.

`build_example.py` recreates saved projects and exact arithmetic assets without
overwriting differing existing assets. Toy-model authoring additionally uses the
optional ONNX development package; opening and running the shipped projects does
not. On-canvas notes identify sources and the deliberate failure controls.
