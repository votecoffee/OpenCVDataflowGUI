# Capture setting negotiation

Open any project below and use **Run All** to inspect three recorded frames.
The **Info** output shows `capture_negotiation`; the optional **Diagnostic**
output shows a separate property table. No camera or external service is used.

| Project | Expected behavior |
| --- | --- |
| [unchanged.json](unchanged.json) | Pass: no writes, exact 128×96 BGR uint8 images at backend-reported 24 FPS. |
| [report_only.json](report_only.json) | Pass with warning: file backend rejects requested 640×480, 30 FPS and exposure; actual 128×96 images remain usable. Polls readback per frame. |
| [reject_settings.json](reject_settings.json) | Fail: required setters reject width/FPS; structured reports and tickets emit, images do not. |
| [reject_format.json](reject_format.json) | Fail: requesting MJPG cannot change the FFV1 file's decoder; readback/format mismatch remains visible. |
| [converted_host.json](converted_host.json) | Pass: host conversion emits grayscale uint16 while the report keeps decoded BGR uint8 and codec observations separate. |

In a real camera graph, use **Camera** with the same request/readback policies.
The default requires successful setters; matching readback is a stricter option,
and report-only is useful for diagnosing a backend. A matching getter is not a
physical camera qualification. Rates here are backend values rather than measured
acquisition cadence. [Detailed policies and limits](../../../docs/capture-negotiation.md).

Run without the desktop:

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/capture_negotiation/report_only.json" --all-images --summary
```

Rejection projects intentionally finish with failed judgments and a nonzero CLI
exit code, without unhandled exceptions. Lean skips diagnostic charts while
preserving negotiation reports and the same judgments.

Original MIT fixtures, generator and annotations by OpenCVDataflowGUI
contributors; no customer images or third-party assets. The on-canvas comment
records attribution. `truth.json` records dimensions, rate, exact decoded pixel
hashes, asset hash and expected outcomes. `assets/reference.avi` has three
lossless FFV1 frames; this backend reports codec FourCC as lowercase `ffv1`.
`build_example.py` verifies any existing clip and regenerates projects; copy the
folder to a fresh location when generating a replacement clip. The video
container bytes may differ across encoders while decoded pixel hashes must match.
