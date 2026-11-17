# Sampling a video by media time

**Video frame at time** (`input.video_at_time`) extracts one image for a requested
presentation time. Find it under **OpenCV > Video I/O**, select a video file and
edit **Time ms**. An optional numeric Time ms input overrides that parameter.
Connect Image to inspection tools and use Run once. The
[five original examples](<../examples/opencv tools/video_time/README.md>) show
exact, nearest, earlier/later and deliberately rejected requests.

This is independent frame sampling: each activation opens/closes its decoder,
assigns a fresh ticket after successful selection and uses a new pixel-frame
identity. Repeated requests can return the same pixels with distinct tickets.
Out-of-order requests do not advance a playback cursor or impose a wall-clock
delay. Use [Video file or Paced video file](paced-video.md) for sequential replay,
tracking and media-clock pacing. Continuous repeatedly samples the current
requested time; it does not implicitly increment it. A plain Run once or bounded
CLI repeats is the appropriate way to try these examples.

## Selection and native positioning

| Selection | Returned frame |
| --- | --- |
| Nearest | Closer member of the timestamp bracket; equal-distance ties prefer earlier. |
| At or before | Last bracket member at/before the requested time. |
| At or after | First bracket member at/after the requested time. |

Native then scan first calls OpenCV's `CAP_PROP_POS_MSEC` setter at
`max(0, requested_ms - preroll_ms)`. It reads the actual returned timestamp rather
than treating setter success as proof of exact positioning. If that frame is
at/before the target, sequential reads build the bracket. If native positioning
is rejected, has no usable time, overshoots the target or cannot produce an
eligible frame, the decoder reopens and scans from the start within the same
read budget. Exact matches need no following frame.

**Native only** permits forward bracketing after a usable native seek but rejects
the cases requiring fallback. **Scan from start** avoids the time setter and
counts decoded ordinals independently of the decoder's index property. Preroll
can place a native anchor earlier to reduce overshoot; it is not a keyframe or
timestamp accuracy guarantee. The source does not estimate a seek index from FPS
as a substitute for calling the native media-time setter.

For the original 20 FPS clip, asking for 175 ms natively returns 200 ms on this
wheel. That is an overshoot, so the default method scans to verify 150/200 ms.
Nearest and earlier return 150 ms; later returns 200 ms. The original known pixels
and centers independently verify which frame was returned. An interframe MPEG-4
fixture also matches a separate complete native decode at reordered requests.

An unavailable bracket member is an explicit error: for example, At or after
past EOF. Nearest/earlier can select the final frame if **Maximum error** permits
it. The signed difference `actual_ms - requested_ms` is always reported; an error
above the limit rejects without an image or ticket. Zero requires equality within
a small floating-point tolerance. Time is nonnegative milliseconds, not seconds,
frame ordinals, host receipt time or camera exposure time.

## Timing and index provenance

The default **Decoder only** policy requires finite, nonnegative, increasing
decoder time through the observed bracket. Unavailable, repeated or backward
properties cannot silently become valid timestamps. **Decoder or FPS** explicitly
permits a labeled rate estimate when native FPS and an index are available. A
scan's own ordinal can supply the index even when `CAP_PROP_POS_FRAMES` is absent.
Estimated values report `timestamp_source: frame_rate_estimate`; raw unusable
timestamps remain separately visible. These estimates assume constant rate and
are unsuitable as an independent variable-frame-rate reference.

Info and the image's `video_frame` payload include filename, selected frame index,
`frame_index_source`, decoder index/raw time, effective time/source, native FPS
and backend. Native-selected indices use the decoder property or null; scans
report their counted ordinal. The nested `seek` record includes requested time,
parameter/input origin, anchor, selection/method, setter acceptance, first native
position with effective/raw timestamp and source, fallback reason, signed/absolute error and read/decoded
counts. Each fallback's native probe remains in those work counts.

Auto ticket mode labels filename/index, or filename/time when no index is known,
while retaining a distinct acquisition identity. Fixed and external ticket modes
keep their normal values. A ticket attached to the Time ms input is retained as
`video_request_ticket` provenance; it does not replace the source's ticket. Wire
the source Ticket input and choose external mode when the request must use a
specific PLC/application ticket. Each independent sample gets its own spatial
root, preventing unrelated requests from silently sharing result geometry.

OpenCV's [video-position properties](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html)
depend on the decoder/backend. Qualification here covers constant-rate lossless
FFV1 and interframe MPEG-4 through this FFmpeg build. The local monotonic-bracket
checks cannot prove the accuracy of every container's position properties, all
earlier timestamps or an independent demuxed PTS. Actual variable-frame-rate,
duplicate-PTS and malformed-timing fixtures remain S1-123.

## Work limits and lifecycle

Maximum read frames caps all read requests, including probe, fallback and EOF
checks. Maximum frame pixels and estimated working bytes check known header
dimensions before reading, then check the actual uint8 BGR frame. The estimate
allows the first native candidate, two bracket images, output and conversion
workspace. No whole-movie frame cache is retained. Native decoder allocations
and runtime snapshots are excluded; this is not a process RSS bound.
Changing dimensions inside a bracket rejects. Automatic orientation is disabled;
color/depth conversions use ordinary image rules after selection. HSV/Lab with
uint16 is invalid. No high-depth recovery from native uint8 decoding is claimed.

Stop is cooperative between local-file native calls. Native open/read timeout
settings are exposed but are not hard execution deadlines. Every decoder closes
in a `finally` path for success, selection errors, cancellation and exceptions.
Constructor, static validation, loading, XML and inspector editing do not decode.
Neither native handles nor selected pixels enter runtime-copy or saved-project
configuration. Save As rebases ordinary project-relative video references.

The source compares file size/modification time/inode before/after sampling and
rejects observed changes. This coarse filesystem guard is not a cryptographic
snapshot or proof against timestamp-preserving edits. Public fixture hashes and
independent annotations provide the separate example acceptance evidence.
