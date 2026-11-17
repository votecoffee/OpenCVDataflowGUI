# Capture settings and readback

`input.camera` opens a local camera only when executed. The stable ID and old
device/backend/width/height/FPS/buffer fields remain compatible with saved graphs.
`input.capture_file` (Recorded capture with settings) uses the same negotiation
logic with a recorded video, retaining Video file's ticket, range, stride, loop,
Run All and host pixel-format conversion behavior. Both appear in **OpenCV →
Video I/O**. Loading or validating either block never opens its source.

## Requests and policy

Zero width, height, FPS and buffer size leave those settings unchanged. Optional
FourCC is exactly four printable ASCII characters, including meaningful spaces.
Exposure, gain and auto-exposure have explicit enable modes so zero and negative
backend values remain valid requests. Auto-exposure encoding and exposure units
are driver-specific; the application does not translate them into milliseconds.
Manual focus/white balance disable their automatic control before applying the
requested focus/temperature. Auto modes explicitly enable the automatic control.

The JSON **Additional controls** field accepts named brightness, contrast,
saturation, hue, sharpness, gamma, zoom, backlight, pan, tilt, roll and iris values.
It rejects arbitrary property IDs and UI/trigger/raw-decoding controls. Every
value must be finite and within the declared limits. Hardware can impose much
tighter limits. Controls are applied once per opened handle, not on each frame;
changed settings, reset or reopening cause a fresh negotiation.

| Negotiation policy | Image emission |
| --- | --- |
| `require_setter` (default) | All requested setters must return success. Readback/geometry differences remain warnings with their actual values. |
| `require_readback` | Setters must succeed, property readback must match and requested dimensions must agree with each actual decoded frame. |
| `report` | Keep valid actual frames even when requested properties are rejected or differ. |

Rejected policy emits an explicit failed ticket, `info` and optional diagnostic,
with reason `capture.negotiation_rejected`; it withholds the image. A camera
setter rejection under either strict policy is reported without reading a frame.
Other acquisition/validation exceptions use the ordinary block error containment
and diagnostics. Failed camera handles are released and can reopen next time.
Recorded sources consume their selected frame even when its settings policy
fails, allowing a finite Run All to finish with reported failures.

## Report and images

`info.capture_negotiation` is strict JSON containing each request, native property
ID, setter result/exception, raw after-set and after-frame observations, match
state, actual decoded dimensions/channels/depth, backend and FourCC text. Frame
format and codec pixel format are separate raw backend observations. The final
`output_frame` describes any host conversion independently of the decoded frame.
Property FPS is backend readback, not a measurement of delivered cadence; use
[video timing diagnostics](video-timing.md) for timing evidence.

Readback defaults to after-open and the first decoded frame. `every_frame` polls
again on each frame. Cached reports retain their original frame index/timestamp
and explicitly say `readback_fresh: false`; actual frame geometry is checked on
every activation regardless. Scalar comparisons use absolute/relative tolerances;
size, FourCC, buffer size and boolean controls compare exactly.

Zero and minus one can mean unsupported properties or legitimate settings. The
report preserves them and flags their ambiguity. Even a readback match is a
backend observation; `hardware_support_proven` remains false. OpenCV explicitly
warns that a successful setter does not prove the device accepted its value.
See the [OpenCV VideoCapture interface and property contracts](https://github.com/opencv/opencv/blob/5.x/modules/videoio/include/opencv2/videoio.hpp).

Image payloads contain a compact capture summary, handle generation, frame index
and spatial grid; full property tables stay in the `info` output. Camera reopen
starts a fresh coordinate root. The optional `diagnostic` is a separate chart
with its own pixel grid, hidden as a connector by default. Graphics-off/Lean
skips chart generation while preserving settings checks and structured reports.

Decoded pixel/byte guards bound accepted arrays, not allocations already made
inside a driver or decoder. Local camera calls can block inside native code;
this feature adds no hard camera-open/read deadline. For owned-process network
decoding with bounded response waits, use [network video](network-video.md).

## Acceptance scope

[Five recorded projects](<../examples/opencv tools/capture_negotiation/README.md>)
use original, lossless images with exact pixel hashes, size, rate and codec
references. Controlled backends cover accepted/ignored/clamped/rejected replies,
exceptions, unavailable properties, ambiguous zero/minus-one values, geometry
disagreement, polling, recovery and teardown. Native FFmpeg file decoding verifies
actual resolution, rate, codec, rejected camera setters, conversion, loop and EOF.
These qualify the shared software contract. Individual camera models, transport
drivers, frame-rate accuracy and physical exposure/focus response remain
device-specific acceptance work; no physical camera was opened for these checks.
