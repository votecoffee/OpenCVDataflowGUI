# Learned stereo: original pixel units and known shifts

Open a project and choose Run Once. The first nine ship tiny ONNX arithmetic
controls and require no preparation. Three native projects use unchanged HITNET
ETH3D weights with a public industrial image. All tools are built in; no plugin
is required. See [the contract](../../../docs/learned-stereo.md) and
[sources and licenses](SOURCES.md).

| Project | Demonstration |
| --- | --- |
| [plane](plane.json) | Known 3-pixel disparity, declared camera/rig, native rectification and calibrated XYZ |
| [subpixel](subpixel.json) | Known 1.5-pixel disparity |
| [concatenated](concatenated.json) | One model input, left channels then right channels |
| [letterbox](letterbox.json) | 16x16 network with rounded content and an 8x8 head; conservative support |
| [head_units](head_units.json) | Head-grid pixels explicitly converted to original pixels |
| [selected](selected.json) | Circle on the left image; excluded pixels remain model context |
| [right_mask](right_mask.json) | A right-camera strip suppresses its corresponding left pixels |
| [empty](empty.json) | Empty selection, ordinary coverage Fail and no model I/O |
| [coverage_alarm](coverage_alarm.json) | Deliberate Fail: minimum coverage exceeds correspondence support |
| [native_8](native_8.json) | HITNET, 8-pixel artificial shift of licensed T-LESS texture |
| [native_16](native_16.json) | HITNET, 16-pixel artificial shift |
| [native_zero](native_zero.json) | Identical views; inspect near-zero predictions and invalid negative contributors |

The nodes expose Disparity, Valid mask, Info and optional Receipt. Disparity is
always `x_left - x_right`, in original left-image pixels, with invalid entries NaN.
Preview is sent to the filmstrip override. Its colors are display-only. Lean
omits that preview while retaining numerical outputs. Coverage indicates retained
support, not matching accuracy. Both image sources use an explicit shared fixed
ticket with filename tickets disabled; this is a controlled replay pair, not
proof of synchronized cameras.

## Prepare native HITNET

From the repository root in the installed application environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_stereo/prepare_model.py"
```

The helper streams the publisher's approximately 440 MB multi-format archive and
retains only the pinned 3,548,840-byte ETH3D ONNX member. It checks the exact SHA-256
and seals `.local/hitnet`. It does not extract arbitrary archive paths, execute
remote code or install a training framework. If you already have the exact ONNX:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_stereo/prepare_model.py" --model "path/to/model_float32.onnx"
```

Repeating preparation verifies the existing bundle offline and preserves differing
files. Runtime never downloads weights. This pinned artifact uses **New/CPU**;
Classic rejects GatherND on the supported build. Input is unit grayscale, left
then right, [1,2,480,640]; output [1,480,640,1] is in network pixels. The wrapper
converts to the 320x240 source grid and source-pixel disparity. Sealed settings
cannot be edited independently of the contract; export a new contract for changes.

## Independent references

The arithmetic model uses `(right_red - left_red) * 255/4` on unit RGB. This is a
quantity/geometry control, not a learned matcher. The plane project has a declared
0.1 m baseline and fx=fy=80, giving Z=8/3 m for its 3-pixel plane. Camera rays and
cloud samples can be checked independently.

Native pairs resize T-LESS scene 6/frame 2 to 320x240, then shift a copy left by
0, 8 or 16 pixels, filling the uncovered strip black. They are artificial planes
using real texture, not actual stereo images of those parts. The separate native
benchmark reports error against those known shifts:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/learned_stereo/benchmark.py"
```

[Recorded results](benchmark.json): mean absolute error is 0.03025/0.03049 source
pixels for the 8/16-pixel interior controls, with full retained evaluation support.
The zero plane's full-region error is 0.20038 pixels and retained support is
94.12%; small negative predictions are rejected rather than clipped. Numerical
accuracy and retained coverage are reported separately. Physical stereo,
occlusion and other-scene accuracy still need their own evaluation.

`build_example.py` regenerates exact assets/projects and preserves differing
existing images. Authoring the tiny toy models additionally uses the optional
ONNX development package; opening or running the shipped projects does not.
