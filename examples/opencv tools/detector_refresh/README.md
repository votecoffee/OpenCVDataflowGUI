# Periodic detector refresh

Open a project and choose **Run All**. Each consumes sixteen original images
without a plugin or download. Reset flow state before replaying.

| Project | What to inspect |
| --- | --- |
| [Periodic](periodic.json) | Find blobs runs at frames 0, 4, 8 and 12. Local image evidence refines tracks between refreshes. |
| [Every frame](every_frame.json) | Comparison with sixteen full-frame detections instead of four. |
| [Freshness limit](freshness_limit.json) | Deliberately expire tracks after two activations without a detector match, even if local tracking succeeds. Subsequent refreshes create new identities. |

The upper textured part moves three pixels per acquisition and disappears at
frame 9. The lower-right part enters at frame 5: Periodic discovers it at frame 8,
Every frame at frame 5. The upper part predicts on frames 9-10, then expires on
frame 11 in those two projects. The first observation of a new identity is
hidden while tentative by default; local or detector observations can confirm it.

Green graphics mean detector evidence, blue local image evidence, and amber
prediction. Select Track with periodic detections to inspect local fit reasons,
detector age, expiry and current-image poses. The final filmstrip includes those
selected graphics. Cadence Refresh emits every image, while Cadence Image only
activates the detector when due. The tracker receives the original image directly.

Period, maximum detector age, local support/error gates and ordinary tracking
limits are editable. A shorter LK window can help local rotation; a larger one
can help translation. Tune against the intended captures. A lost local fit waits
for full detection rather than learning replacement background from a prediction.
The freshness example intentionally uses a limit shorter than the cadence.

The source images, annotations, generator and projects are original
OpenCVDataflowGUI contributor work under the repository MIT license, with a
canvas attribution comment in each project. No customer or third-party images
are included. `truth.json` records independent intended centers, pixel extents,
visibility, timestamps and SHA256 hashes. Actual native blobs drive the graph.

Automated checks compare measured centers within 0.5 pixels for visible parts,
verify discovery/expiry, current tickets, overlays and exact detector invocations.
Four versus sixteen detector calls is a call-budget result, without a measured
runtime speedup or claim of industrial tracking accuracy.

Regenerate with the application environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/detector_refresh/build_example.py"
```

See [contracts](../../../docs/refresh-tracking.md) and [all examples](../../README.md).
