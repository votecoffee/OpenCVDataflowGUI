# Timestamp-paced video replay

**Paced video file** (`input.paced_video_file`) releases recorded frames against
a monotonic wall clock. The existing **Video file** stays unpaced for throughput
work and existing projects. Both share native decoding, frame range/stride,
formats, tickets and cursor persistence. No new dependency or hardware is needed.

## Configure and run

Choose a local file/backend, Start frame, exclusive End frame and Stride. Speed
1 follows media intervals; 2 runs twice as fast and 0.5 takes twice as long.
Skipped stride frames are not emitted, but elapsed media time between selected
frames still determines their scheduled release. No additional frames are dropped.
The first selected frame releases without an intentional delay, including when
Start frame is not zero. Processing time consumes the scheduled interval rather
than adding a full extra sleep per frame.

Run All rewinds and consumes the selected range once. Continuous loops it until
Stop. Run once advances one frame. The same source works through the headless
API and before a background acquisition queue. Queue overflow policies remain
separate choices; they can discard captures even though the source itself does
not implicitly drop frames. The toolbox group is **OpenCV > Video I/O (videoio)**.

## Timestamp policies

- **Decoder or FPS** prefers an increasing decoder property. Missing, repeated or
  decreasing values use the valid native frame rate; optional Fallback FPS is used
  only when native FPS is unavailable. Zero disables that configured fallback.
- **Decoder only** rejects missing or nonincreasing decoder values within a cycle.
- **Configured FPS** requires a positive Fallback FPS and uses that declared rate.

Info and the image's `video_frame` payload retain the effective `timestamp_ms`,
`timestamp_source`, raw `decoder_timestamp_ms`, frame index, native FPS/backend
and playback details. Estimates are explicitly labeled; they never masquerade as
decoder time. Speed changes release timing without scaling the media timestamp.
Changing configured source/replay settings restarts the selected range, following
the ordinary Video file settings contract. Playback controls cannot be recipe-bound.

OpenCV defines [CAP_PROP_POS_MSEC](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html)
as a video-position property and documents backend-dependent behavior. This is
not an independent demuxed PTS guarantee. The shipped native checks cover FFmpeg
FFV1 and OpenCV MJPEG. Variable gaps and bad metadata are checked with a controlled
clock/property source; qualification of actual variable-frame-rate/bad-metadata
clips remains S1-123. Network streams and seeking by media time are separate work.

Maximum gap rejects excessive or nonpositive selected-frame intervals before
waiting. It is measured in media milliseconds before Speed. An allowed long gap
can still wait a long time at low speed; Stop remains cooperative during that wait.

## Lateness, Stop and Resume

The default **Rebase** policy starts a fresh wall schedule when processing or
publication is later than Maximum lateness. This avoids a catch-up burst while
preserving every selected frame. **Catch up** keeps the original deadlines and
releases overdue entries in order without an additional wait. It does not promise
that processing or display can sustain the requested rate.

Clock waits request at most 10 ms per sleep and check cancellation between waits.
Stop during a pacing wait closes the decoder, issues no ticket/output event for
the held frame and retains its index. Resume releases that pending frame without
waiting for an obsolete deadline, then starts a fresh wall schedule. Runtime
copies transfer cursor and numeric scheduling state independently; decoder
handles and clock services are not copied or stored in saved projects.

The native decoder's open/grab/retrieve calls cannot be forcibly interrupted by
this clock. Prompt Stop is qualified while waiting on the replay clock, with the
normal cooperative checks during seek/stride. This is not a hard real-time source
or a synchronization guarantee between independent video sources.

Tickets are assigned after the pacing gate and before publishing the image.
Source execution timing includes its wait; ticketed downstream processing begins
with release. Info's playback record reports action, speed, waited/lateness/media
interval, timing policy, cycle and interval provenance for inspection.

## Loop boundaries and stateful tools

Expose the optional **Cycle start** boolean output and connect it to a sequence
block's **Reset** input when a clip boundary must clear history. It is true on
the first frame and each loop's first frame, rather than on an ordinary Resume.
Native media timestamps start over when the range wraps; no artificial continuous
media timestamp is substituted. The example resets image tracking at the boundary.
Different tools may deliberately keep history, so reset is an explicit connection.

Loop spacing uses Stride/native FPS, or the configured rate when needed. If no
rate is available, the last valid media interval is used and labeled as an
estimate. A one-frame loop with no rate has no known duration and is rejected.
Playback metadata distinguishes these boundary estimates from decoder deltas.

## Verification and examples

[Three original projects](<../examples/opencv tools/paced_video/README.md>) use a
lossless eight-frame moving-part clip, native grayscale/threshold/blob detection,
image tracking and selected filmstrip overlays. They cover paced full-range,
faster strided replay and loop/reset wiring. Original pixel hashes, intended
media times and positions are recorded separately from native results.

Fake-clock checks cover variable gaps, speed, processing overhead, lateness,
fallback/strict timing, cancellation, resume and loop estimates without sleeping.
Native checks cover pixels, relative timestamps, stride/EOF, source-runtime copies,
queue ordering and a real offscreen continuous-runner Stop/Resume. Parameter,
clock-failure cleanup, XML, dropdown and no-I/O editing checks cover the extension
contracts. These are local software/decoder checks, not physical-camera or
hard-real-time acceptance.
