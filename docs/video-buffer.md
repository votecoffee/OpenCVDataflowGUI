# Bounded encoded-video decoding

`cv.decode_video` (**OpenCV → Video I/O (videoio) → Video decoding**) decodes a
bounded, complete encoded clip through OpenCV's FFmpeg `IStreamReader` interface.
It does not create a temporary video file. Use it for encoded video bytes from
a plugin or other producer, or select **Path** for a bounded local file snapshot.
For one frame per graph activation from a long recording, use the existing
[Video file](video-files.md) source instead.

## Inputs and outputs

Optional **Encoded** accepts bytes, bytearray, or a uint8 vector, including a
row/column vector. Its immutable snapshot takes precedence over Path. Pass
`bytes(view)` for a memoryview; memoryviews do not support the runtime's ordinary
input snapshots. Relative Path uses **Path base**, with project references
preserved when saving elsewhere. Loading/validating a graph does not decode it.

**Frames** is an ordered `FlowGroup` of 1–64 images. Connect it to **Unbatch serial
entries** to dispatch ordinary per-frame inspections, or **Select group entry**
to choose one. These use the existing scheduler and ticket framework. Unbatch
can process all group members in one graph activation; this is not a replay source
that needs Run all. The **Image** output previews one selected group position.
**Frame count** and **Info** report the selected ordinals, times, source SHA-256,
resource counters and stop condition. See the [six saved examples](<../examples/opencv tools/video_buffer/README.md>).

Start frame is a zero-based decoded ordinal. Stride selects every Nth frame;
Frame count caps the selected group. Decoding scans from the beginning and
includes skipped frames in Maximum scan frames. It does not depend on a native
frame-seek claim or a reported total frame count.

Native decoding is uint8 BGR. Output formatting supports the ordinary color/depth
controls, including RGB, alpha expansion, gray, HSV/Lab and float32. HSV/Lab with
uint16 is rejected before native allocation. Increasing output depth converts
the decoded samples; it does not recover encoded high-bit-depth video data.
Native orientation handling is explicit, off by default. Dimensions must remain
consistent with the opened header; a changed layout fails without publishing a
partial group.

## Times, tickets and coordinates

Each frame gets a fresh ticket, labeled `filename#000001` or `buffer#000001`.
Repeated decoding creates a new ticket session. Group members and preview share
the same frame ticket. An incoming Encoded ticket is retained separately as
`video_source_ticket`; arbitrary payload fields such as a lot ID are preserved.
Input spatial/pyramid/raw-sensor/alpha context is discarded. Frames get a new,
shared image grid for the decoded clip, with per-frame video metadata. Encoded
bytes do not establish a physical calibration or a camera acquisition identity.

`video_frame.timestamp_ms` is **video presentation time**, not a camera exposure,
wall clock, PLC time or processing duration. Timestamp policy chooses native
decoder times with a labeled FPS estimate fallback, native times only, or an
explicit unavailable value. A missing/invalid time, including zero after frame
zero, follows that policy. FPS estimates do not qualify variable-rate timing.
The shared runtime separately supplies processing timing and pass/fail lineage.

## Limits, end conditions and cleanup

All limits have validation and tooltip rules. Only the explicitly qualified
**ffmpeg** backend is offered. The Python SDK adapts a buffered byte reader;
its reads can return less than requested. The wrapper bounds encoded size,
read chunk, cumulative bytes served including rereads, read/seek calls, frame
dimensions, scanned ordinals, selected count and estimated decoded workspace.
Byte seeks must stay inside the immutable buffer. File size is checked before
reading and a size change during the snapshot is rejected.

Selected images plus estimated conversion/BGR buffers are checked against
Maximum decoded bytes before pixel reads. Native opening/codec internals and
runtime input/history/output copies are outside that estimate. These controls
are not a hard RSS bound or native execution deadline. Decoder threads and native
open/read timeouts are exposed. Cancellation is cooperative between reads and
through callbacks; it cannot forcibly interrupt arbitrary native processing.

When Short clip is **allow**, fewer frames are accepted if at least one and the
requested preview are available. Info reports `selection_satisfied=false` and
`stop_reason=native_end_or_decode_failure`. OpenCV's false read result does not
distinguish every corrupt/truncated stream from clean EOF; this is not a container
integrity check. Choose **error** to require the requested count. Satisfying a
selected range does not mean the rest of the container was decoded or validated.

Callbacks record a sticky failure and return EOF/error positions rather than
raising through the Python/native bridge. After each native call the wrapper
checks that failure. Capture release and reader close run on success, budget
rejection, cancellation and exceptions. An unsuccessful activation publishes no
partial images/group; normal block diagnostics record its error.

## Evidence

The original MIT fixture has eight 64×48 BGR frames at 10 FPS, a moving bright
part, PNG pixel references and a lossless FFV1 AVI. Tests verify exact decoded
samples and presentation times, strided ordering, output conversions, 1/257/4096
byte reads, fresh tickets and grid replacement, file Save As/XML, dropdowns,
short clips, resource limits and teardown after controlled native exceptions.
Saved graphs include per-frame contour inspection and deliberate failures.
The installed Windows OpenCV 5.0.0 contrib build is qualified for this FFmpeg path;
MSMF and growing/live producer streams are not qualified by this increment.

Upstream interface: [VideoCapture](https://docs.opencv.org/5.0/main_modules/classcv_1_1VideoCapture.html),
[Python buffered-reader adapter](https://github.com/opencv/opencv/blob/5.0.0/modules/videoio/misc/python/pyopencv_videoio.hpp).
