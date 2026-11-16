# Seeded MIL tracking

Five small projects exercise **OpenCV → Video Analysis (video) → Object tracking → Track object
box (MIL)** using an original, annotated synthetic clip. Open a project and use
**Run all**; a single run only initializes the model. Inspect `ready`, `status`,
`bounds`, `pose` and drawable results. All files are self-contained; no download,
camera, plugin or external service is required.

| Project | Expected result |
| --- | --- |
| [translation.json](translation.json) | Seven frames of a 32×32 textured part moving two pixels/frame. First two activations withhold results; the next five report the tracked box/center. |
| [region_seed.json](region_seed.json) | Editable rectangular region supplies the seed and gates accepted centers. |
| [template_reacquisition.json](template_reacquisition.json) | Template matching supplies fresh seeds. A blank frame deliberately fails and expires the track; the returning part initializes a new track ID and becomes ready after warmup. Overall replay reports failure for the lost observation. |
| [wrong_region.json](wrong_region.json) | A configured seed outside the inspection region initializes, then updates fail the center gate and expire. No ready box/pose is emitted. |
| [work_limit.json](work_limit.json) | Deliberately tiny work budget rejects processing before native model creation. Invalid-input errors are expected. |

Headless replay, including all frames:

```powershell
opencv-dataflow-run "examples/opencv tools/mil_box_tracking/translation.json" --all-images --profile debug
opencv-dataflow-run "examples/opencv tools/mil_box_tracking/template_reacquisition.json" --all-images --profile lean
```

The latter command returns a failure exit code because of the intentional lost
frame. Wrong-region and work-limit projects also return failure. Start each
experiment with reset/reopen; saved files do not contain a trained native model.

The search radius is explicitly set to 8 for this two-pixel-per-frame clip;
the native default radius 25 showed substantial drift in repeated replays.
MIL has a fixed-size box and no native confidence score. Pose angle zero denotes
image axes, not measured rotation. Warmup 3 withholds an observed inaccurate
bootstrap estimate; it does not guarantee localization. Tests check subsequent
accepted centers within 8 pixels on these frames. Stochastic native tracking and
this controlled clip do not establish real-scene identity/occlusion reliability.
Region selections gate centers and seed bounds; they do not mask native training.
See [controls, state and limitations](../../../docs/mil-tracking.md).

All PNGs, annotations and project content are original work by OpenCVDataflowGUI
contributors, under the repository MIT license. On-canvas comment blocks retain
this attribution. `truth.json` records rectangles and SHA-256 image hashes.
`build_example.py` regenerates the nine PNGs and five projects deterministically;
it does not run the tracker. No customer or third-party images are included.
