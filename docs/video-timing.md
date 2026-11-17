# Video clock diagnostics

**Video timing monitor** (`sequence.video_timing`) observes an existing image
stream without repairing or replacing its timing. Find it under **OpenCV > Video
I/O**. Connect Image from Video file/Paced video file; connect matching Info or
let the tool read `video_frame`/`network_frame` from the image payload. Encoded
video groups retain the same metadata after Unbatch. Pixels, depth, encoding,
ticket and image coordinates remain unchanged.

The [eight original projects](<../examples/opencv tools/video_timing/README.md>)
exercise actual Matroska streams with constant/variable, repeated, backward and
zero timestamps, missing duration metadata and an incorrect declared rate. Their
times are written directly into binary block headers and separately checked by
an independent reader and native OpenCV/FFmpeg decoding. No extra muxer, download,
camera or plugin is required.

## What is checked

Each row records the source index, raw decoder position, effective source time,
origin/estimate label, selected analysis time, delta, normalized per-frame interval,
native FPS, independent reference/error and explicit error/warning codes.

- Missing, negative, repeated or backward analysis times reject the row. Index
  order must increase; explicit positive stride is allowed. A zero raw position
  after the first frame is unusable. A new network connection can legitimately
  start at zero while its global emitted-frame index continues.
- Decoder only requires raw decoder timing and rejects estimated/unknown origins.
  A decoder label with a missing or contradictory effective timestamp also rejects.
  Allow estimates explicitly permits labeled effective rate estimates. Raw values
  remain visible. Allowing estimates does not establish original acquisition time.
- Allow variable accepts unequal positive intervals. Require constant rejects
  observed variation above the larger of absolute/relative tolerances. Match
  nominal FPS requires valid reported FPS and compares each normalized interval
  with its nominal period. A mismatch is only a warning under Allow variable.
- Maximum interval optionally rejects a long media gap between selected frames;
  zero disables it. Delta/index step normalizes cadence for strided input. Neither
  a large interval nor an index step is proof of a lost camera frame.
- Optional independent reference times are indexed by the original frame index.
  They must use the same time origin/units. Excessive absolute error or an absent
  indexed reference rejects. A wired list overrides the configured reference.
  Reference arrays may describe intentionally bad clocks; matching an authored
  duplicate time does not make that clock monotonic or valid.

Cadence is reported as insufficient, constant or variable from positive intervals
in the retained window. It is separate from `ok`; an unreliable stream can still
have some positive intervals. At least two such intervals are needed to classify
variation. Classification describes observations, not every unseen frame or
authentication of a container clock.

## Scope and memory

Latch failures defaults on: `ok` remains false after any rejected row in this
stream context, until Reset/context reset. With it off, the current row controls
the judgment. Session frame/rejected/estimated counters and error-code counts
survive eviction from the bounded History size. The score is the fraction of
accepted rows in that session; it is a diagnostic quality fraction, not a
probability or physical-camera accuracy claim.

The common sequence contract provides warmup, stream limits, memory budget,
context reset/separation and copies. Warmup defaults to one so the first image and
its availability report can pass through; cadence is initially insufficient.
Image/recipe/grid, source filename/backend, reference or parameter changes select
a fresh context. Reset and Stream are optional explicit inputs. Wire paced
**Cycle start** to Reset when replay loops should begin a new timing history.
Without Reset, index/time restart is correctly reported as a discontinuity.
Sequence close clears its owned runtime state; saved JSON/XML stores configuration,
not the row history. Independent runtime copies do not share mutable history.

`max_gap_ms` is the shared ingress wall-clock reset interval. It is different from
Maximum interval, which checks video media time. Memory limits cover stored state
and reserve room for the bounded row window; native processing, graphical outputs
and UI snapshots use additional memory.

## Viewing results and failures

Report and status expose current/window/session facts. `ok` and `variable_rate`
are optional wiring choices for downstream decisions. The optional Chart output
and named diagnostic image plot time in decoded order; blue shows a supplied
reference, amber estimates, red rejected rows and green accepted decoder values.
That chart has its own pixel grid. Drawable timing text belongs to the current
input image and can be combined with other result graphics for the final filmstrip.
Turning graphics off skips chart/drawing construction and retains numerical results.

Bad clock observations are ordinary failed inspections with `video_timing.*`
reasons, not exceptions. Current pixels and reports remain available through the
normal failed-input policy. Current rejected rows issue isolated warning logs;
latched history does not repeatedly log an old failure. Invalid settings, unrelated
Info, absent timing metadata and processing exceptions use the normal contained
block-error path and emit no stale output.

File replay, encoded-video groups, paced replay and network frames preserve
`decoder_timestamp_ms` separately from their existing effective timestamp/origin.
No source's existing timing or ticket policy is changed by observing these fields.
The encoded-video timestamp helper retains its original two-value default API.

OpenCV's [video-position/FPS properties](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html)
remain backend reports rather than independent demuxed PTS or exposure timestamps.
The original fixture subset follows [Matroska timestamp elements](https://www.matroska.org/technical/elements.html)
and [Motion JPEG mapping](https://www.matroska.org/technical/codec_specs.html#v_mjpeg).
Native qualification here covers this Windows FFmpeg build and authored MJPEG
containers, including malformed controls. It does not qualify arbitrary codecs,
camera transport latency, synchronized devices or intentionally falsified metadata.
