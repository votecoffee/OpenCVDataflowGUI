# Animated image codecs

Four blocks share an ordered, bounded animation contract:

| Block | Native operation | Use |
| --- | --- | --- |
| Load animation frames (`input.animation_file`) | `imreadanimation` | Load APNG, GIF or WebP into an ordered frame group. Relative input paths follow Save As. |
| Decode animation from memory (`cv.decode_animation`) | `imdecodeanimation` | Decode an `EncodedAnimation`, raw bytes, or a uint8 vector. |
| Encode animation in memory (`cv.encode_animation`) | `imencodeanimation` | Encode an ordered image group and verify native readback before emitting its buffer. |
| Write animation file (`output.animation_file`) | `imwriteanimation` | Verify a staged native file before exclusive publication to the selected destination. |

The blocks appear under **OpenCV → Image Codecs (imgcodecs)**. File loading is
under **Files**, processing under **Codecs**, and writing under **Images**.
The [six original example projects](<../examples/opencv tools/animation_codecs/README.md>)
cover exact storage, explicit GIF timing quantization, frame inspection, protected
file output and a deliberate validation failure.

## Frames and provenance

`Frames` is a FlowGroup with one ValuePacket for each image, in stored order.
Decode can select a start index and count. A zero count requests all remaining
frames; exceeding the bounded group size requires an explicit smaller selection.
`Image` shows `preview_frame` within that selection. Its optional result graphic
is a drawable frame-index/duration label; raw pixels remain unannotated.

Use **Select group entry** to choose an image or **Unbatch serial entries** to
send every image through ordinary processing blocks. The blocks decode a group
in one activation. They do not sleep for frame durations or add timed GUI playback.

A file acquisition supplies one filename ticket for the group. Each member also
has `animation_frame` metadata: its native frame index, duration, loop count,
background and offset within the selected subset. That offset starts at zero
for the selection; it is separate from a camera acquisition timestamp.

An encoded memory packet preserves each input member's ticket, coordinates,
pyramid context, companion data, logical color format and group label. Decoding
that packet with **Unchanged** restores the logical RGB/RGBA/gray layout. Files
and raw buffers return canonical OpenCV layouts unless a color/depth conversion
is selected. Reopening a file creates new image coordinate identities; it cannot
recover camera calibration or PLC ticket payloads absent from the file.

Frames must share their canvas and channel layout. Encoding accepts uint8 gray,
BGR/RGB and BGRA/RGBA images; gray is expanded to BGR for GIF/WebP. uint16/float
frames are rejected rather than implicitly reduced. File formats do not store
arbitrary per-frame application payloads; that preservation belongs to the
in-memory packet. Runtime buffers and images are omitted from saved projects.

## Timing, controls and verification

Durations come from frame payloads with a configurable default, a uniform value,
or an explicit list. The optional Durations input overrides those choices. Each
frame needs 1..655350 ms. GIF uses 10-ms units: **exact** rejects fractional units,
while **floor_10ms** explicitly floors them and reports both requested and stored
durations. A quantized duration below 10 ms is rejected.

Loop and background controls can use the first frame's metadata or an explicit
override. Native loop count zero means infinite looping. GIF viewers may interpret
finite repeats differently: the report includes both the native count and the
container's repeat field. The blocks verify the native count, not browser playback.

Codec controls are enums or validated numeric fields:

| Codec | Controls and limits |
| --- | --- |
| APNG | Compression and zlib strategy. PNG filter, zlib buffer-size and bilevel options are ignored or unsupported by the native animation backend and are omitted. Background stores BGR, with zero alpha; grayscale background selection is unavailable. |
| WebP | Lossless or lossy mode and lossy quality 1..100. Native animation lossless uses QUALITY=101; the still-image lossless flag is ignored by this backend. BGRA background packing is normalized and checked against the container. |
| GIF | All eight native quality/palette choices, five dither settings, global/local palette and alpha threshold 0..255. Fast modes choose their own dither algorithm. GIF reduces color and has binary transparency. Native background selection is unavailable; use zero BGRA. |

Every encode verifies frame count, durations and loop count on readback and
compares decoded samples in frame order. APNG and lossless WebP require exact visible
samples and alpha. RGB beneath fully transparent pixels is outside that guarantee.
Lossy modes report per-frame maximum sample errors. A completed encode is storage
success; it does not establish an inspection tolerance for lossy data.

Native APNG/WebP can merge identical frames. A changed frame count or duration
is rejected before any buffer or destination file is published. Native file I/O
uses a snapshot/staged file; Python handles the final Unicode source/destination
paths. A destination created during encoding is preserved. Temporary files are
removed on failure.

## Limits and backend qualification

Input/output buffers are capped at 32 MiB. Groups contain at most 64 selected
frames, with a configurable total budget up to 16,777,216 pixels and an 8192-pixel
axis limit. The separate APNG poster counts toward the pixel budget. Pillow checks
container dimensions and frame count before native frame allocation; timed frame
decoding and encoding use OpenCV.

A separate APNG poster is supported for grayscale/BGRA. This wheel corrupts a
three-channel RGB poster, so that input is rejected with guidance to use BGRA.
The native reader can also confuse a poster with a timed frame. The wrapper
extracts the original poster's PNG chunks and decodes them as a standalone native
PNG. Poster pixels, timing and separation from animated frames have independent
known-image checks. Poster output has its own coordinate identity.

The installed OpenCV 5 contrib build qualifies APNG, GIF and WebP. AVIF is
unavailable and remains outside this qualification. Other builds must pass the
same codec/timing checks. Allocation limits are bounded work controls, not process
isolation for native decoders or user plugins.

API references: [OpenCV image codecs](https://docs.opencv.org/5.0/main_modules/imgcodecs.html),
[Animation fields and timing](https://docs.opencv.org/5.0/main_modules/structcv_1_1Animation.html),
[codec flags](https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html).
