# Run profiles and observation overhead

Choose **Run profile** in the execution toolbar while stopped. The choice is saved with
the program and applies to the root graph, nested ToolBlocks, parallel part jobs and
background acquisition. Existing projects default to **debug**.

| Observation / behavior | debug | balanced | lean |
| --- | --- | --- | --- |
| Diagnostic capture | TRACE and above | INFO and above | WARNING and above |
| Successful start/finish activity bars | Yes | Yes | Omitted |
| Routine parameter snapshots | Yes | Omitted | Omitted |
| Warning/error parameters and exception traces | Yes | Yes | Yes |
| Retained input previews | Yes | No | No |
| Auxiliary `Result.graphics` retained | Yes | No | No |
| Output image previews | Yes | Yes | No |
| Ticket event history | All steps | All steps | Sources, final decisions, failures/errors and failed chains |
| Continuous GUI snapshots | Every iteration | At most every 200 ms | At most every 500 ms |
| Fixed development pause between iterations | 50 ms | None | None |

These are capture/execution settings, separate from the diagnostic dock's display filter.
Filtering TRACE out of a Debug view still captures it. Balanced/Lean skip filtered capture
before formatting messages, walking ticket context or taking parameter snapshots. Existing
history remains available within retention limits; changing the profile does not erase it.
The diagnostic dock and timeline identify the current capture profile.

Use Debug while developing or training from retained frames. Balanced retains sampled
output previews and complete activity spans for tuning. Lean targets a mature job using
its actual output blocks and final decisions, with fault diagnostics available. Switch
back to Debug and run once to collect full previews again. Profile changes preserve live
processing state rather than resetting queues or learned temporal models.

## Result graphics overrides

Use **Graphics default** in the toolbar to follow the run profile or set optional
result graphics **On** or **Off** for the project. Set **Result graphics** in a
block's inspector to inherit the project default or override it. For example, turn
graphics off for the project and enable them on the pattern matcher you are tuning.
Both settings are saved. Changes clear disabled graphics from the display and take
effect on the next run without resetting flow or sequence state.

Optional graphics are separate from declared image outputs such as a histogram plot
or an annotated overlay. The **Off** override also preserves exposed **Result
graphics** outputs that the run profile would produce, including layers exported
through ToolBlocks. With the default policy, Balanced and Lean continue to omit
those layers. Use **On** to enable them under either profile. Output image preview
retention still follows the selected profile.

## Processing contract

Profiles preserve every declared output, parameter, mask/region operation, ticket identity,
failure reason, final judgment, queue ordering and production statistic. Port visibility
still controls the editor's sockets. A declared overlay, histogram plot or flow visualization
is computed even when it is not wired downstream. Required output writes are executed.

Input validation and private mutable-input isolation remain enabled. Grayscale, Gaussian
blur and Canny explicitly declare read-only processing, allowing the leaner profiles to
avoid an extra process-input copy. Custom block classes keep the existing private snapshots
by default; a read-only declaration must appear explicitly on the class, so an overriding
subclass does not accidentally inherit the optimization.

Lean avoids image transfers into the display graph and exposes a lightweight result summary
there. The worker still owns the real port values, runtime state and processing buffers.
Continuous processing proceeds between stable UI publications. On Stop, the editor shows the
most recent completed iteration, including a result held for a throttled update. Resume carries
queues, ticket counters and temporal state forward. Actual camera pacing remains under its
source/driver settings. Faster execution can
change queue fill/drop behavior under the configured overflow policy.

Lean ticket history summarizes successful work while retaining failed chains and final
events. Root completion still updates end-to-end timing and running counts. Detailed
success timelines cannot be reconstructed for work whose events were not captured. Inactive,
rejected and waiting-frame semantics are unchanged. Error traces and failure codes remain
available, but image previews are disabled in Lean.

## Headless operation

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/inspection/validation_images/validation_images.json --profile lean --repeat 100
```

`--profile debug|balanced|lean` overrides the saved choice for that invocation without
rewriting the project. Use `--result-graphics inherit|on|off` to override the
project graphics default for the invocation. Block overrides still apply. JSON
receipts identify the effective profile and graphics default. Without `--report`,
the CLI avoids building/retaining per-iteration structured receipts. A requested report
still incurs the cost of describing the outputs it contains.

## Custom blocks

Avoid constructing auxiliary images when they are not requested:

```python
graphics = {}
if self.capture_graphics:
    graphics["diagnostic overlay"] = draw_overlay(image, measurements)
return Result({"measurements": measurements}, graphics=graphics)
```

Use `self.generate_graphics` for drawable objects that can feed the **Result graphics**
port as well as the inspector. Returning a graphic that was already computed cannot
recover the drawing time. The runtime can skip its validation, retention and display
copies. Declared image outputs remain
part of the processing contract. For expensive diagnostic arguments, guard their creation
with `self.logger.isEnabledFor("DEBUG")`; Python evaluates call arguments before invoking
the logger. A custom class can explicitly declare `process_inputs_readonly = True` after
auditing its processing and finalization against the input snapshot contract.

## Measured example

`python scripts/benchmark_run_profiles.py` compares the same 1280×720 BGR frame through
ordinary grayscale → Gaussian blur → Canny blocks. It rotates profile order over three
rounds, warms each graph, collects 60 samples per profile and verifies the output SHA-256
matches. Measurements include private copies, validation, payload handling and statistics;
they exclude GUI snapshots and the development pause.

Two local runs on Python 3.13.9 / OpenCV 5.0.0 with one OpenCV thread measured:

| Median graph time | debug | balanced | lean |
| --- | --- | --- | --- |
| Run 1 | 21.37 ms | 19.94 ms | 18.56 ms |
| Run 2 | 24.43 ms | 20.14 ms | 19.41 ms |

Lean achieved approximately **1.15–1.26× headless throughput** on this workload, with
identical pixels. These are local observations, not a performance guarantee for other
programs. Removing GUI pacing/copy overhead is an additional effect measured separately
from this benchmark. Native-heavy algorithms may see a smaller proportional improvement.
Use the benchmark script or the job's timing/counter views to measure the relevant chain.
