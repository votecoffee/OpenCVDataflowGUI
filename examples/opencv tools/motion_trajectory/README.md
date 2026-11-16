# Save and replay camera motion

Open a project below with **File → Open…**, then choose **Run all**. Five original
frames are batched and corrected using a sealed trajectory. Inspect Image, matching
Corrected/Coverage/Transforms groups and Report. Unbatch emits individual current
frame tickets; invalid borders are black with zero coverage.

| Project | Expected result |
| --- | --- |
| [replay_files.json](replay_files.json) | Pass; known affine corrections undo five integer shifts in PNG captures. |
| [replay_video.json](replay_video.json) | Pass; the same original pixels decoded from lossless FFV1, with checked media indices/timestamps. |
| [highdepth.json](highdepth.json) | Pass; uint16 pixels and corrections retain their original depth. |
| [invalid_reject.json](invalid_reject.json) | Deliberate fail; one explicitly invalid correction withholds all corrected frames. |
| [invalid_partial.json](invalid_partial.json) | Deliberate fail with four valid outputs and an omitted-frame reason. |
| [invalid_skip.json](invalid_skip.json) | Pass with explicit partial status; four frames are emitted, without an invented correction. |
| [changed_pixels.json](changed_pixels.json) | Deliberate validation error; a one-pixel change rejects before warping. |
| [reordered_frames.json](reordered_frames.json) | Deliberate validation error; original content in the wrong order is incompatible. |
| [native_fit_capture.json](native_fit_capture.json) | Pass; fit and capture a new centered trajectory, then replay it identically without a second fit. |
| [time_guard.json](time_guard.json) | Deliberate validation error; edited media timing rejects even when pixels match. |

Known source shifts are `(0,0), (2,-1), (-1,2), (3,1), (0,-2)` on 160×112 images.
Frozen source-to-target matrices are their inverse translations, with independent
pixel/coverage truth. Native fit/capture is separate from those frozen references.
Indexed PNG trajectories use nominal 40 ms periods; video trajectories require
the recorded 25 FPS media timestamps. Neither asserts physical exposure timing.

To keep a newly fitted trajectory, add **Save motion trajectory**, connect Capture's
Trajectory output, and choose a new `.json` destination. The default preserves
existing files. Later connect **Load motion trajectory** and the original Frames
group to Replay. The examples do not automatically overwrite any assets.

All frames, FFV1 data, records, generators and projects are original work by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
No customer/third-party images are included. Attribution travels in canvas comments.
`truth.json` pins all 16 source/trajectory assets. Rebuild from the repo root:

```powershell
.venv\Scripts\python.exe "examples/opencv tools/motion_trajectory/build_example.py"
```

The video projects require the wheel's FFmpeg/FFV1 support. Tests verify source
hashes/regeneration, exact corrected pixels and masks, current tickets, fresh
acquisition roots, reset, native fitting parity and expected Debug/Lean outcomes.
See [the full contract](../../../docs/motion-trajectory.md) for schema, timing,
omission policies, Save As and bounded 1–64-frame operation. These fixtures do not
establish physical camera identity or general stabilization accuracy.
