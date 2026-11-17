# Video frames selected by media time

Open a project and choose **Run once**. No download, camera or plugin is needed.
Double-click a filmstrip image to enlarge the selected part and its blob overlay.
Inspect the source's Info for requested/returned time, signed error and whether
native positioning or a fallback scan supplied the image.

| Project | Expected result |
| --- | --- |
| [native_exact.json](native_exact.json) | Native-only request at 250 ms returns frame 5. |
| [nearest.json](nearest.json) | 175 ms is halfway between two frames; the earlier tie returns frame 3 at 150 ms. |
| [at_or_before.json](at_or_before.json) | Returns frame 3 at 150 ms. |
| [at_or_after.json](at_or_after.json) | Returns frame 4 at 200 ms. |
| [error_guard.json](error_guard.json) | Deliberately rejects a 225 ms request: the nearest frame is 25 ms away and only 10 ms is allowed. No image/ticket is emitted. |

The native setter overshoots a 175 ms request to 200 ms on the pinned wheel. The
default method notices this and scans the bounded clip to verify the bracket.
Each Run once is an independent sample with a fresh ticket and pixel frame.
Repeated runs do not advance through the movie. To sample different times,
edit Time ms or expose its optional numeric input. For sequential movies use
Video file or Paced video file instead.

```powershell
opencv-dataflow-api run "examples/opencv tools/video_time/nearest.json" --repeat 2 --summary
opencv-dataflow-api run "examples/opencv tools/video_time/at_or_after.json" --summary
```

The eight lossless 160 by 96 frames are specified at 20 FPS (50 ms apart). A
28 by 24 pixel part has area 672 and center `[31.5 + 8*i, 41.5]` at frame `i`.
`truth.json` records these independently specified times/centers, original pixel
hashes and the AVI hash. `build_example.py` regenerates in a fresh folder and
rejects an existing clip with different pixels/time/count rather than overwriting it.

Clip, generator, annotations and projects are original work by OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE). No customer
or third-party images are included; each graph has an attribution comment.
See [media-time sampling contracts](../../../docs/video-time-seeking.md) for
selection, timing, lifecycle, resource limits and qualification boundaries.
