# Track image detections through gaps

Open either project and choose **Run All**. Twenty original annotated PNG frames
feed Find blobs, Track image detections, selected result graphics and the final
filmstrip image. No plugin, model download or camera is required. Reset flow state
to replay from the start.

| Project | Expected behavior |
| --- | --- |
| [Oriented gaps](oriented_gaps.json) | Keep identity through two-frame and four-frame disappearances. Predict position, scale and orientation; reacquire the same track. |
| [Expire and reacquire](expire_reacquire.json) | Keep the short gap; expire after the third consecutive miss in the long gap. The returning part starts a fresh tentative identity. |

The first detection is tentative and hidden by default. Green annotations are
observed; amber annotations are predictions. Hidden frames 6-7 and 12-15 still
show predictions until the configured expiry. Frame 14 reports `missed_frames`
in the expiry project; frame 16 starts the new track and frame 17 confirms it.
Angle crosses the undirected +/-90-degree boundary near the beginning.

Select Track image detections to inspect uncertainty, association costs,
observed/confirmed flags, confidence and expiry reasons. Change Maximum missed
frames, Maximum unobserved ms, angle period or uncertainty limits and replay.
Predicted poses are estimates, not evidence of a currently visible/pickable part.
Nominal acquisition time is 100 ms here; use timestamp input or video media time
for real sampling intervals. Confidence is a heuristic rather than probability.

`truth.json` contains independent intended centers, scale, angles, visibility,
timestamps and SHA256 hashes. Actual OpenCV measurements, rather than those
annotations, drive the graph. Automated checks require center error below 0.4 px,
wrapped angle error below 2 degrees and local-size error below 8% for this
rasterized synthetic ellipse sequence. They verify identity, expiry, tickets and
visible final overlays. This demonstrates a controlled baseline, not general
industrial accuracy.

All images, annotations, graphs and [generator](build_example.py) are original
OpenCVDataflowGUI contributor work under the repository MIT license. Attribution
is also included as a canvas comment in both projects. There are no customer
images or third-party source images.

Regenerate with the application environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/image_tracking/build_example.py"
```

See [tracking contracts](../../../docs/image-tracking.md) and [all examples](../../README.md).
