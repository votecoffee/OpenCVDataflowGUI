# Unordered rotating-camera panoramas

Open a project below with **File → Open…**, then Run. Select the panorama block
to inspect native cameras, pairwise checks, Coverage and Report. Corner
measurements from the first source are mapped onto the new panorama grid and
combined with source-boundary graphics in the final filmstrip image.

| Project | Expected result | Demonstrates |
| --- | --- | --- |
| [unordered.json](unordered.json) | Pass, complete | Left/right/center inputs; the first two do not overlap, but center connects them. |
| [reordered.json](reordered.json) | Pass, complete | Different input order with the same named center anchor. |
| [disconnected.json](disconnected.json) | Fail, no panorama | An unrelated fourth image is rejected under the default policy. |
| [allow_partial.json](allow_partial.json) | Pass, incomplete | Explicitly allow the connected three-camera component. |
| [fail_partial.json](fail_partial.json) | Fail, partial panorama | Keep a diagnostic partial image with a failed judgment. |
| [uint16_plane.json](uint16_plane.json) | Pass, complete | Plane projection retains original 16-bit samples above 255. |
| [masked.json](masked.json) | Pass, complete | Per-source rectangle masks restrict registration, rendering and source-point acceptance. |
| [distorted.json](distorted.json) | Fail, no panorama | An anisotropically distorted view violates the common rotating-pinhole model. |

The optional connectivity chart has a separate pixel grid. In Debug it is a
fallback filmstrip image for failed registration. Lean omits charts and graphics;
normal numerical outputs and judgments still run. Partial failed outputs remain
observable, subject to downstream failed-input policies.

## Independent controls

The original fixture generator renders three **480×360** pinhole views with a
known **420-pixel focal length**, centered principal point and yaw **−36°, 0°,
+36°** from one optical center. A seeded textured scene gives left/center and
center/right overlaps without a left/right overlap. The unrelated image and
distorted view are deliberate negative controls. The uint16 sources are original
8-bit values multiplied by 257. Only the corner detector's branch converts to
8-bit; panorama rendering uses the original 16-bit inputs.

[truth.json](truth.json) records known cameras, independent world-ray landmarks,
case expectations and hashes for all eight PNGs. Tests bound focal relative
error to **10%**, rotation error to **3°**, and landmark RMS to **5 panorama
pixels**. These controlled-fixture acceptance limits differ from the block's
configured **3-source-pixel** pairwise camera-model limit. No estimated cameras
are used to fabricate expected truth. Other focused tests compare projection
and rendering to analytic rays and check masks, tickets, frames and errors.
Neither these examples nor the panorama fit provide physical calibration.

Run one project from the repository environment:

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/rotating_panorama/unordered.json" --profile debug --summary
```

Regenerate projects, source images and independent truth with:

```powershell
python "examples/opencv tools/rotating_panorama/build_example.py"
```

Generation does not run camera estimation. Exact regeneration checks compare the
eight PNGs, eight saved projects and truth record. See the
[block contracts](../../../docs/rotating-panorama.md) for assumptions, parameters,
nonlinear coordinate ownership and partial policies. The
[camera refinement examples](../panorama_refinement/README.md) add native solver
controls, fixed/refined intrinsics and observations reserved from fitting (S1-147).

## Attribution

Images, projects, generator and controls are original work by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
There are no customer images or third-party assets. Every project includes an
on-canvas attribution and assumptions comment.
