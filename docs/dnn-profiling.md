# DNN layer profiling

**OpenCV > Deep Neural Networks (dnn) > DNN profiling** contains three opt-in
variants of the existing image, multi-input and batched ONNX inference blocks:
`ml.dnn_profile`, `ml.dnn_multi_profile` and `ml.dnn_batch_profile`. The ordinary
inference blocks keep their parameters and execution path. Profiling does not
run a second inference or change global OpenCV settings.

## Controls and outputs

| Parameter | Behavior |
| --- | --- |
| Profile mode: Optional (default) | Keep inference outputs if native timing is unavailable; emit its reason and a null native total. |
| Profile mode: Required | Treat unavailable timing as an execution error; emit no current inference/profile event and clear native state. |
| Profile mode: Off | Skip added forward timers, timing queries and the chart. Emit an explicit disabled profile. |
| Maximum profile layers | Bound named layer records across all native calls in one activation; default 4,096, maximum 16,384. A call exceeding the remaining budget is unavailable in full, never silently truncated. |
| Profile plot layers | Plot the top 0..50 layers of the last measured call; default 20, zero hides the chart. |

**Profile** is a `dnn_layer_profile_v1` data record. Each call records exact model
SHA-256/byte count, native import evidence, input shapes, complete forwarded and
retained output names, cache reuse, requested/selected engine, backend, target
and fusion setting. It contains native total ticks, tick frequency, converted
milliseconds, named layer IDs/timing slots, zero-tick flags and forward wall time.
Batch profiles also retain the start index, real rows and submitted batch shape
for each native call. They measure calls, not individual images within a batch.

`native_total_ms` is present only when every call has a complete measurement.
When some calls are unavailable the status is `partial`, the full total is null
and `measured_native_ms_sum` explicitly contains only the measured calls. When
all are unavailable the status is `unavailable`. Off reports no observed calls,
null timings and an unknown native-call count; it does not claim inference ran
zero times. Timing values vary with warmup, model, machine and concurrent work.

Forward wall time covers just `net.forward`. Native layer timing has the native
engine's scope. Neither includes model import, input preprocessing, portable
output conversion or profile collection. Existing per-block timings continue to
cover the whole execution. A native layer sum and the native total are retained
separately; do not infer end-to-end ticket latency from either.

The optional result graphic is a diagnostic bar chart, independent of the source
image. The single-image variant also passes the original image through unchanged,
including depth, encoding and coordinate/ticket metadata. Batch source ownership
and multi-input context guards follow the existing inference contracts. Graphics
off/Lean hides the chart while retaining an explicitly enabled data profile.
Receipt mode is independent. Profiling controls are diagnostic observations and
do not change the algorithm revision in a sealed model contract.

## Native qualification and limits

The Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 Windows CPU controls qualify
Classic, New and requested Auto. Each profile reads the tick-vector overload of
`getPerfProfile()` immediately after the same forward. Classic uses one-based
layer IDs; New uses zero-based IDs. Actual IDs align names with timing slots,
including reordered names. Missing/duplicate names, ambiguous IDs, mismatched
vectors, nonfinite/negative values, all-zero samples and alternate unqualified
overloads produce explicit unavailability. No guessed name/timing association
is published.

Zero ticks can mean fusion, skipping, nonexecution or clock resolution. They are
labelled `zero_ticks` and never presented as proof of zero cost or as individually
confirmed fused operations. The included Conv→ReLU control has one zero-tick layer
with Classic fusion on and none with fusion off. This establishes the behavior
of that control, not a general native fusion inspection API.

Only selected OpenCV/default CPU execution is qualified for these reports.
Other targets return unavailable, with the chosen mode governing whether inference
is retained. Existing explicit CPU fallback records both request and selection.
Per-layer placement and the engine ultimately chosen by Auto are not proven;
`native_placement_verified` is false. There is no accelerator or throughput claim.
The off switch removes this wrapper's added observation work; it does not disable
the native engine's internal tick bookkeeping.

State is private to one block: each activation starts fresh observations, while
native model caching remains reusable. Reset, close and native/required-profile
errors clear the cache and observations. Exceptions use the common block executor.
Rejected failed inputs retain their prior input/result history and emit nothing.

Try [the thirteen original projects](<../examples/opencv tools/dnn_profiling/README.md>)
for fusion, disabled/unavailable measurements, multi-input and batch-tail calls,
independent chart/receipt controls, high-depth images and sealed Save As/XML.
See [fixture sources and native references](<../examples/opencv tools/dnn_profiling/SOURCES.md>).
