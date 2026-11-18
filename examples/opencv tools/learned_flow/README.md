# Learned optical flow: explicit first-to-second motion

Open a project and Run Once. Nine projects ship small arithmetic ONNX controls;
three use explicitly prepared RAFT weights and a licensed industrial image.
All tools are built in. See [the block contract](../../../docs/learned-flow.md)
and [sources/attribution](SOURCES.md).

| Project | Demonstration |
| --- | --- |
| [plane](plane.json) | Known (2,-1) motion and supported forward image reconstruction |
| [subpixel](subpixel.json) | Known (1.5,1) source-pixel displacement |
| [concatenated](concatenated.json) | Combined model binding, first channels then second |
| [letterbox](letterbox.json) | Rounded 16x12 content in 16x16, with a smaller head |
| [head_units](head_units.json) | Explicit head-pixel units and independent x/y scaling |
| [selected](selected.json) | Circle on the first image; full model context retained |
| [second_mask](second_mask.json) | Second-image exclusion tested at predicted endpoints |
| [empty](empty.json) | Empty selection: ordinary Fail, current empty support, no model I/O |
| [coverage_alarm](coverage_alarm.json) | Deliberate Fail: required coverage exceeds supported endpoints |
| [native_positive](native_positive.json) | RAFT on artificial (4,-2) motion; forward warp also shown |
| [native_negative](native_negative.json) | RAFT on artificial (-3,2) motion |
| [native_zero](native_zero.json) | RAFT on identical images, with legitimate zero-motion truth |

Flow is `(dx,dy)` in original first-image pixels: first `(x,y)` corresponds to
second `(x+dx,y+dy)`. It is not velocity. First owns the output ticket/grid; both
input identities are retained even when their filename tickets differ. Swap inputs
to estimate reverse flow. Negating vectors alone does not change their grid.

Inspect Flow, Valid mask, Info and optional Receipt. The filmstrip override shows
the diagnostic direction/magnitude map. Result graphics also contain drawable
arrows on the original first image. Magnitude/Angle ports are hidden initially;
exposing them requests numeric maps. Lean skips unexposed maps, previews and
drawings without changing vectors. Invalid vectors are NaN, not zero.

The Plane project's Warp uses Source to target with first as Image, second as
Target and Valid mask as Flow support. Default content bindings reject changed
image anchors or contradictory flow direction. Saving `.flo` fields requires
Preserve invalid vectors when the mask excludes samples.

## Prepare the native projects

From the repository root in the installed application environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_flow/prepare_model.py"
```

The helper downloads 64,119,337 unchanged bytes from a pinned OpenCV Zoo revision,
verifies SHA-256 and seals `.local/raft`. No training package or remote Python
code is needed. If you already have identical weights:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_flow/prepare_model.py" --model "path/to/optical_flow_estimation_raft_2023aug.onnx"
```

Repeat preparation verifies the bundle offline; differing files are preserved.
Execution never downloads models. The bundle uses Classic/CPU, inputs `0`/`1`,
and named full-size output `12006`. Raw RGB model input is produced from unit RGB
with std=1/255. Do not choose a head by output-list position. Sealed settings must
be changed by exporting a new contract rather than editing them independently.

## Independent checks and limits

The arithmetic ramps have known channel gradients and translations. Their ONNX
head solves dx/dy from channel differences; it is an original numerical control,
not a learned estimator. Native projects area-resize a public T-LESS capture to
240x180, then translate a copy with black uncovered borders. They are artificial
motion controls, not recorded motion or occlusion ground truth.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_flow/benchmark.py"
```

[Recorded results](benchmark.json) give mean endpoint errors 0.01861/0.01457/0.00297
source pixels for positive/negative/zero motion on a fixed interior. All three
retain that evaluation region. Coverage is reported separately and is not
matching confidence. Other scenes, physical motion and accelerators need their
own evaluation.

`build_example.py` recreates exact projects/assets while preserving differing
existing images. Authoring the toy models uses the optional ONNX development
package; opening and running the shipped projects does not.
