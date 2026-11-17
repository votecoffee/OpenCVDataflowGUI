# Live network video

`input.network_video` reads one decoded frame per activation through the explicit
FFmpeg backend. Find **Network video** under **OpenCV > Video I/O**. Supply an
HTTP, HTTPS or RTSP URL, connect Image to processing, and use Continuous. Run once
requests one frame. This live source has no finite image set for Run All.

The [two original loopback examples](<../examples/opencv tools/network_video/README.md>)
start their own local MJPEG service through the existing program lifecycle.
They exercise normal reception and a connection closed after four frames, with
native blob measurements, tracking, selected graphics and final filmstrip images.

## Deadlines, recovery and Stop

The decoder runs in one private Python process. A single bounded shared-memory
slot transfers uint8 BGR pixels; small JSON messages carry commands and metadata.
No frame files are written to disk. The Windows worker enters a kill-on-close job
before it receives settings or starts OpenCV. POSIX uses a private process group.
The owner terminates and joins its worker and releases the slot on Stop, close,
reset, parameter change, terminal failure or cancellation. Constructor, validation,
project loading and inspector editing never connect to a stream.

Worker initialization, opening and each read have separate deadlines. The parent
checks cancellation while waiting and enforces a deadline independently of the
native FFmpeg timeout. Stopping does not wait for a long native open/read timeout.
OS process creation/termination and scheduling are not hard real-time operations;
the configured deadline bounds a response wait, not every cleanup/launch action.
These Windows software checks do not qualify arbitrary camera drivers or OS faults.

After an open/read failure or worker exit, the current activation can reconnect up
to **Reconnect attempts** times after the initial attempt. Delay between attempts
is cancellable. Zero attempts disables retries. Exhaustion raises a contained
block error with no image or ticket emission; the next activation can try again.
Format/resource/protocol errors are terminal for that activation and are not retried.
No stale frame is substituted. Previously assigned tickets remain distinct.

There is no application read-ahead or implicit latest-frame dropping. The server,
transport and native decoder can buffer or discard frames; FIFO at the camera,
bounded latency and lossless acquisition are not guaranteed. Use the explicit
image-queue block for independent acquisition and bounded downstream buffering.
After Stop/Resume, a new connection starts from whatever the server sends; this
does not resume a saved live-camera cursor. Reconnection does not recover lost
frames or prove physical synchronization.

## Controls

| Control | Meaning |
| --- | --- |
| URL | HTTP/HTTPS/RTSP endpoint, up to 4096 UTF-8 bytes; no whitespace/fragment/local-file scheme. |
| Source name | Public label in result metadata/messages, independent of the endpoint. |
| Backend | FFmpeg only; unsupported builds fail explicitly. |
| RTSP transport | TCP or UDP; ignored for HTTP/HTTPS. |
| Startup timeout | Worker import/initialization response wait, 100..60000 ms. |
| Open/read timeout | Per-operation response waits, 50..60000 ms; also passed to FFmpeg. |
| Reconnect attempts/delay | 0..20 retries per activation and 0..60000 ms between attempts. |
| Decoder threads | 1..16 native FFmpeg decoding threads. |
| Maximum frame pixels/bytes | Reject oversized headers/decoded frames; bytes also sizes the single shared slot. |
| Color format/pixel depth | Existing explicit conversions from native uint8 BGR. HSV/Lab reject uint16. |

The resource limits exclude native decoder/internal network allocations, output
conversion buffers and runtime snapshots. This is not a process RSS limit or a
security sandbox. Known oversized header dimensions reject before requesting a
frame; actual decoded size/type is checked again. A dimension change inside an
open connection rejects instead of silently changing its coordinate grid.
Automatic orientation is disabled; rotate explicitly downstream when required.

Live acquisition settings cannot be recipe-bound. Enum controls use dropdowns;
numeric limits and coupled format restrictions use normal inspector validation.

## Tickets, time and coordinates

Image and Info carry the source's normal auto/fixed/external ticket, assigned only
after a current frame has decoded, passed guards and converted successfully.
Cancellation or failed acquisition assigns no new ticket. The block's timing
includes acquisition waiting; a ticket's end-to-end processing clock starts after
acquisition and therefore excludes that source wait.

`network_frame` on the image payload and Info contain:

- Public source name and a zero-based emitted `frame_index`, retained across close
  and reconnect, reset by configuration changes or explicit runtime reset.
- One-based `connection` and zero-based `connection_frame_index`, plus the retries
  in the current activation. Failed opens do not create successful connection IDs.
- `stream_start`, true on the first frame of every successful connection.
- Decoder `timestamp_ms` only when finite, nonnegative and increasing within the
  connection; otherwise null with `timestamp_source: unavailable`. There is no
  invented FPS-based exposure timestamp. A reconnect can restart media time.
- Native FPS when usable and `received_unix_ns`, the host wall-clock time just
  after native decode. Receipt time is not camera exposure time and can jump with
  wall-clock adjustments.

Each connection gets a new spatial root, retaining the decoded dimensions.
Expose **Stream start** and wire it to sequence **Reset** when tracking/background
memory must restart at a reconnection. The example does this explicitly. Numeric
stream state copies between display/execution graphs; worker handles, pixels,
threads and shared-memory names do not enter runtime copies or saved projects.

## Endpoint privacy and qualification

Frame metadata and user-facing error text omit the URL; native stderr is discarded
because some backends print credentials or query strings. Retry diagnostics use
public error codes. Saved JSON/XML, editor parameters and diagnostic parameter
snapshots can still retain the configured URL. Use endpoints without embedded
secrets; this tool is not a credential vault.

Native Windows qualification covers loopback HTTP multipart MJPEG, disconnection,
stalled opening/reading, decoder exit, cancellation, recovery, frame guards and
program-owned server cleanup. RTSP TCP/UDP and HTTPS controls depend on the actual
FFmpeg build/server and have not been qualified against physical equipment here.
The native options are documented in [OpenCV video I/O properties](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html).
Shared-slot ownership follows [Python shared memory](https://docs.python.org/3.14/library/multiprocessing.shared_memory.html).
