# Network video and reconnect

Open either project and choose **Continuous**. Its first run launches the included
loopback-only MJPEG server automatically. Stop releases the decoder; ending the
program session stops its owned server. The next run starts a fresh connection.
No camera, download, plugin or separately started server is required. Port 18764
must be free; to change it, edit both the source URL and program service arguments
and readiness text in **Program settings**. If the server cannot start, the normal
program diagnostics report the failure without adopting an unrelated process.

| Project | Demonstrates |
| --- | --- |
| [live_stream.json](live_stream.json) | Repeated eight-image stream, image conversion, threshold, blob measurement, tracking and selected final graphics. |
| [reconnect_reset.json](reconnect_reset.json) | Server closes after four frames; the source reconnects, emits Stream start and explicitly resets the image tracker. |

Double-click filmstrip images to inspect the selected result graphics. Expose
Info to inspect frame/connection indices and timestamp provenance. Ticket order
continues across reconnection; image coordinates and track identities start fresh.
The default source ticket is numeric; fixed and external modes also work.
This is an infinite live source: use Continuous or bounded CLI repeats.

```powershell
opencv-dataflow-api run "examples/opencv tools/network_video/live_stream.json" --repeat 6 --summary
opencv-dataflow-api run "examples/opencv tools/network_video/reconnect_reset.json" --repeat 10 --summary
```

The first project emits frames 0..5 of its connection. The second receives local
indices 0,1,2,3,0,1,2,3,0,1 over three connections; its emitted index remains 0..9.
Each image contains one 28 by 24 pixel bright rectangle, area 672. Known centers
are `[31.5 + 8*i, 41.5]`. JPEG is lossy; the threshold-based centers/areas remain
exact for this fixture. Tracking centers are filtered estimates, not raw truth.

Images, generator, server and projects are original work by OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE); no customer or
third-party images are included. An on-canvas comment records that attribution.
`truth.json` pins JPEG hashes and intended geometry. Regenerate in a fresh folder
with `build_example.py`; an existing image with different bytes is rejected.

See [network-video contracts](../../../docs/network-video.md) for cancellation,
buffering, resource limits, endpoint privacy and physical-camera qualification.
