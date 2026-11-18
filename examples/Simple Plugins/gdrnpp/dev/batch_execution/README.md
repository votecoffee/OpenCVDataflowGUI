# CPU batch execution and measured agreement

**Keep batch size one as the default.** Larger batches run correctly on the
tested CPU snapshot, but tiny numerical differences remain and complete-cycle
improvement is not established across applications. The recipe, four-thread
policy, crop transforms, checkpoint and depth/ICP gates remain unchanged.

## What the control does

The setup popup's **Instance batch size** groups crops from the current capture
into network calls, up to eight per call. It does not combine physical frames,
change training, or promise more CPU parallelism. The existing worker already
uses the recipe's explicit Torch CPU-thread setting.

Reports now expose `batch_execution`: requested size, instance count, actual
group sizes and forward-call count. Five proposals at sizes 1/2/4/8 use
`[1,1,1,1,1]`, `[2,2,1]`, `[4,1]`, and `[5]`. Empty captures run no forwards.
Tail groups preserve input order and crop/output ownership; cancellation between
groups publishes no partial result. The default, existing trained archives and
all original source images/receipts are preserved.

## Numerical audit

The [all-case audit](numerical/parity.json) compares all
32 frozen real T-LESS inputs at batch sizes 2/4/8 against fresh single-instance
baselines, in original and reversed grouping order: **192 comparisons, zero
accept/reject changes**. All fresh single-instance raw heads and final poses
reproduce the frozen baseline. This uses the original CPU checkpoint and four
threads throughout.

For this API test, crops from different captures are grouped explicitly at the
network interface. Each crop retains its own camera/transform, and refinement
and verification use its original RGB-D frame. The inputs are not presented as
one physical scene. The separate complete-worker study below uses five real
proposals in one actual capture through the normal saved-recipe workflow.

Largest raw or final pose difference: **0.000288 mm / 0.000394 degrees**; maximum
absolute learned-field difference: 0.00002575. These are differences against
single-instance execution, not errors against ground truth. Fifty-six fit
dictionaries differ numerically, including six rendered-sample counts and three
support/contradiction fractions. Maximum support change is 0.00983 percentage
points and maximum RMS difference is 0.000611 mm. Do not claim bit identity.

Diagnostic limits were fixed before this audit: raw pose 0.01 mm / 0.01 degrees,
final pose 0.1 mm / 0.1 degrees, plus unchanged decisions. No comparison exceeds
those limits. They are audit tolerances, never inspection acceptance thresholds.
All 224 baseline/comparison depth reports are retained, including rejections.
An [interactive table](audit.html) links each outcome.

These remain assisted development captures with supplied rectangles; positive
views share one arrangement and severe occlusion is unqualified. New-object
training, automatic detection and production picking remain separate.

## Complete worker timing and process ownership

The study prepares one worker per graphics mode, then runs one warmup and three
measured rounds at each batch size, alternating order. It measures normal
private-packet transfer, capture preparation, network, ICP, final verification
and requested source-aligned image/native colored cloud. Acquisition,
caller-side file decoding and GUI composition are excluded. Nested scopes overlap.

The [corrected samples](worker_tree_batches.json) retain
every timing, fit, process snapshot and returned-array size. Medians and full
sample ranges are shown in the [comparison figure](comparison.png).
These are sequential shared-workstation observations on one capture, not a
capacity guarantee or a GPU result.

| Batch | Actual groups (5 proposals) | Graphics off median / range (s) | Image + cloud median / range (s) |
| --- | --- | --- | --- |
| 1 | `[1, 1, 1, 1, 1]` | 3.940 / 3.897–3.990 | 5.193 / 4.470–27.800 |
| 2 | `[2, 2, 1]` | 3.903 / 3.809–3.985 | 4.749 / 4.515–5.667 |
| 4 | `[4, 1]` | 3.934 / 3.571–4.235 | 4.777 / 4.225–4.840 |
| 8 | `[5]` | 3.646 / 3.631–3.675 | 4.359 / 4.210–4.365 |

The corrected run measured about 2.1–2.5 mean logical cores busy across the
whole request. Batch eight medians improve 1.08x / 1.19x here, while the first
pass did not show consistent improvement. The 27.800-second graphics-on
batch-one sample is retained, not discarded. This is insufficient evidence
for a universal batch default or application cycle-rate guarantee. Keep the
default rather than selecting a batch size from one application's timing result.
The outlier's worker packet-writing stage took 23.287 seconds, compared with
2.043 seconds of network execution. The profile identifies a transport stage
worth investigating; it does not establish the underlying cause of the delay.

**Windows profiling correction:** the virtual-environment `python.exe` launcher
can remain idle while a child Python process owns the network. Sampling only
the launcher reported zero CPU and roughly 4 MB of memory, despite completed
inference. The new read-only sampler includes the launcher and its owned
descendants. It returns unknown for incomplete/changing process sets or reused
PIDs, rather than treating missing counters as zero CPU.

Mean core occupancy is summed process CPU time divided by request wall time.
Summed process RSS includes shared pages; individual peak working sets are
cumulative and cannot isolate one batch size's transient peak. The first pass is
[retained with an explicit invalid CPU/memory label](launcher_only_first_pass.json).
Earlier cycle/preparation receipts that sampled `client.process.pid` alone have
the same launcher-only memory limitation. Their immutable numbers must not be
interpreted as complete worker memory; wall timing and numerical checks remain
separate evidence. Direct in-process model memory measurements are unaffected.

## Reproduce and review

Update the external worker package and reset/restart a loaded worker. No new
host SDK, host ML dependency or concurrency feature is required. The host
already exposes the complete report, so the new batch counters need no port change.

`run_batch1.json` and `run_batch4.json` use the same
five-proposal RGB-D reference with sealed batch-one and batch-four setups. Load
the trusted plugin, choose the verified external checkpoint folder, and run
either project. The native fly-around cloud is the default result thumbnail.
The saved `review_project.json` opens recorded baseline/batched colored clouds,
fit images and the timing figure without weights; file-source PASS means reading.

From this folder, use an updated external worker, verified model folder and fresh output:

```powershell
& $worker benchmark_batches.py batch1.gdrnpp $models ../preprocessing/capture.json new_batch_timing.json
```

The full source distribution supplies the frozen-source acquisition instructions
and the parity script's helper modules. No benchmark code is selected by a saved
recipe. Actual batch execution, complete-cycle timing and arbitrary-object
accuracy are separate checks.

## Attribution

T. Hodaň et al., **T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects**, WACV 2017, [CC BY 4.0](https://cmp.felk.cvut.cz/t-less/). Existing real
first-return RGB-D/calibration and supplied visible rectangles are reused.
Derived batches, sealed reference assets, fit reports, tables, charts and colored
clouds are modifications. Annotation poses/class labels do not enter inference;
no weights are bundled or endorsement implied. [Model and dependency notices](../THIRD_PARTY.md).
