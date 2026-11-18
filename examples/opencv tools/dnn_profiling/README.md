# DNN native layer profiling

Open a project below and press **Run once**. Select **Profile** and inspect its
**Profile** data and the native-layer chart in Results. All images and small
ONNX models are included; no downloads, learned weights or hardware are needed.
The blocks are under **OpenCV > Deep Neural Networks (dnn) > DNN profiling**.

| Project | What to inspect |
| --- | --- |
| [classic_fused.json](classic_fused.json) | Classic CPU Conv→ReLU with fusion: a zero-tick layer is visible. |
| [classic_unfused.json](classic_unfused.json) | Same input/model, Classic fusion off: separately timed layers, identical scores. |
| [new_cpu.json](new_cpu.json) | New engine CPU names, zero-based layer IDs and converted ticks. |
| [auto_cpu.json](auto_cpu.json) | Requested Auto and CPU selection; actual per-layer placement remains unverified. |
| [disabled.json](disabled.json) | Same inference with no added timing queries, forward timers or chart. |
| [receipt_off.json](receipt_off.json) | Native profile remains available with portable inference receipts off. |
| [plot_off.json](plot_off.json) | Full timing data with the optional chart disabled. |
| [optional_budget.json](optional_budget.json) | One-record budget: timings unavailable, valid inference retained. |
| [required_budget.json](required_budget.json) | Expected profiling error: no current output event, rather than a fabricated zero total. |
| [multi_input.json](multi_input.json) | Two prepared image tensors rebound to left/right, one profiled native forward. |
| [batch_tail.json](batch_tail.json) | Three ticketed images, batch size two: two native calls with real rows 2/1. Run All completes once. |
| [sealed.json](sealed.json) | Checked model bundle; profiling controls do not change its algorithm revision. |
| [high_depth.json](high_depth.json) | uint16 source with matching normalization; primary pixels retain original depth. |

The original 1×1 convolution transforms the prepared RGB channels by adding
`[0.15, -0.1, 0.05]`; ReLU clips negative values and spatial averaging yields
three scores. Multi-input sums the two normalized images first. Independent
image arithmetic verifies these known values, while the timing checks compare
the profile to the same forward's native ticks. These are API/graph controls,
not trained classification or industrial accuracy demonstrations.

Native timings vary with warmup and workload; compare repeated runs on the same
machine. A zero-tick layer may be fused/skipped/below clock resolution. An
unavailable total is null. Batch times belong to each submitted native batch,
not separately measured images. The chart shows the last measured call and is
hidden in Lean mode; the enabled data profile remains available. Off skips the
wrapper's added measurements, without changing native internal counters.

[Detailed contracts and limits](../../../docs/dnn-profiling.md) ·
[Sources and licenses](SOURCES.md). `truth.json` pins all seven model/image/bundle
assets by SHA-256. `build_example.py` regenerates the original assets/projects in
a new destination with the development `onnx` package, and refuses to overwrite
an existing asset folder. Normal execution does not need `onnx`.
