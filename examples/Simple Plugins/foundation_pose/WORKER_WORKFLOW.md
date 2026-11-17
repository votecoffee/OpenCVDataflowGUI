# FoundationPose live CPU execution

Load **plugin.py** explicitly with the editor's Load Python plugin action, then
open a project in **worker_execution**. The bundled numerical wheel is checked
against its SHA-256 before registration. Loading or validating does not start
inference, install dependencies or download weights. The original file-only
review projects remain available without the plugin.

This is an experimental single-instance pose tool. It uses a supplied visible
instance mask, registered uint8 RGB/physical axial depth, rectified pinhole
calibration and an original-origin metric mesh retained in `trained.sfp`.
Setting up the object retains these inputs; it does not train new network weights.
The native setup/mask editor is the next milestone and is not provided yet.

## Runtime setup

Keep ONNX Runtime, Trimesh, Pillow and the model weights in a separate trusted
Python environment. Do not install another OpenCV wheel into the app environment.
The isolated environment must contain the dependencies listed in the bundled
wheel's `worker` extra. The bridge supplies this exact package to that interpreter
from the wheel; installing another version there is unnecessary.

Set the following variables before launching the app, using your own paths:

```powershell
$env:SIMPLE_FOUNDATION_POSE_PYTHON='C:/path/to/isolated/Scripts/python.exe'
$env:SIMPLE_FOUNDATION_POSE_MODELS='C:/path/to/pinned/weights'
```

The package also provides `machine.configure_runtime(python, weights)` to store
these two paths in a machine-local profile. Saved project JSON cannot choose an
executable. Neither recipes nor this folder include weights. The explicit
`fetch_models.py` script downloads the pinned files after you review the model
terms in [THIRD_PARTY.md](THIRD_PARTY.md). Both weight sizes/hashes are checked
again before network sessions open.

## Projects and results

Each part has two variants:

| Project suffix | Initialization | Meaning |
| --- | --- | --- |
| `_global.json` | 252 global rotation hypotheses | Full registration with supplied visible mask; no annotated pose input. CPU runs take several minutes. |
| `_assisted_replay.json` | Two prior raw network proposals | Short execution/cache/graphics demonstration with additional initialization assistance. This does not measure independent global localization. |

Both variants share the same sealed training and physical acceptance settings:
two refinement passes, batches of eight, four ONNX threads, 5 mm depth tolerance,
70% minimum support and 10% maximum free-space contradiction. Do not tune the
individual projects to turn rejection into acceptance. The ranking network sees
the complete candidate cohort in one call. Its logits depend on that cohort and
are not probabilities; the shorter replay has different scores by design.

The original held-out and wrong-object global results are both rejected. The
short assisted examples also reject. No accepted `pose` is emitted on rejection;
`poses` contains accepted alternatives for **one** instance, not separate parts.
All raw candidates remain in `report` and the fit table. These data do not qualify
object detection, CUDA rendering, general pose accuracy, collision-free grasping
or physical robot picks.

Select the **cloud** output for the app's native colored fly-around thumbnail
and popout. It contains measured scene rows followed by derived model/axis rows,
whose ranges are identified in its metadata. The top rejected proposal remains
visible for diagnosis. Green agrees with measured depth, red contradicts free
space, purple lies behind a measured surface, and orange has unknown/outside
evidence. Those vertex graphics are explanatory; acceptance uses rendered first
surfaces against **raw** depth, never filled/model-derived observations.

The image-list result is the RGB pose/depth-fit overlay. Named `pose_depth_fit`
result graphics provide the same source-grid diagnostic. Both image and cloud
axes use the original part origin. Graphics-off suppresses review generation
without changing candidates or acceptance. Capture inputs stay immutable.

The worker persists across calls and caches the sealed recipe/mesh and both
networks; asset identity/stat changes invalidate the caches. A request deadline,
cancellation or failure retires its owned process tree and discards stale replies.
Independent tool instances use independent processes; same-instance calls are
serialized. Whole-request and cache timing is reported separately from inference.

The original [pilot README](README.md) and its checksum manifest describe that
earlier saved-output release. This document and `worker_execution/sha256.json`
describe the additional execution release; the earlier assets remain intact.

## Data and model attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017. [Publisher](https://cmp.felk.cvut.cz/t-less/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Copied captures, retained metric training and derived reviews use those terms.
The visible instance masks are supplied label assistance; annotated poses enter
evaluation only. Original selected-member provenance remains in `source`.

FoundationPose: Bowen Wen, Wei Yang, Jan Kautz and Stan Birchfield, CVPR 2024.
The separately published NVIDIA ONNX weights use the NVIDIA Open Model License;
they do not relicense the original noncommercial research source. The worker does
not import or redistribute that research implementation. See
[THIRD_PARTY.md](THIRD_PARTY.md), bundled MIT/Apache-2.0 notices and weight pins.
No endorsement is implied.
