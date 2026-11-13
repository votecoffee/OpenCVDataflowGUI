# Video file workflows

## Replay a clip

1. Add **Input > Video > Video file** and select a local video.
2. Set the zero-based start frame, optional exclusive end frame and stride.
   For example, start 1, end 6 and stride 2 emits frames 1, 3 and 5.
3. Connect the image to an inspection graph. Run once advances one frame.
   **Run all** replays the selected range once from the start frame. **Continuous**
   repeats that range until you press **Stop**. **File > Reset flow state**
   returns the source to its start frame.

Each activation emits a frame ticket such as `clip.avi#000003` by default.
The image payload and `info` port contain the original frame index, time in
milliseconds, timestamp provenance, nominal FPS and decoder backend. When
the decoder supplies no usable time, the block estimates it from the frame
index and FPS when available. Headless replay runs at processing speed.

## Record processed frames

Add **Output > Video > Write video file**, then wire a processed image to
it. Select a new destination, FPS, codec and backend. Use FFmpeg with FFV1
AVI for exact uint8 pixels, MJPG AVI for lossy recording, or mp4v MP4 for
lossy color recording. For a portable graph, select the project
path base and use a path relative to the saved project.

Input frames must be uint8 and have fixed, even dimensions. The recorder can
write BGR or monochrome frames. Tagged RGB/RGBA/HSV/Lab inputs are converted
to the requested encoding; alpha is discarded without compositing. Unsupported
pixel depths fail explicitly. Playback uses the configured constant FPS
even when the source is variable-rate or the graph skips frames. The
container is finalized when execution closes. Start each new recording at a
new path. Existing files are preserved.

Run a saved finite replay graph from the command line with:

```powershell
opencv-dataflow-run my_video_project.json --all-images
```

The writer's `info` contains the actual backend, requested codec/FPS, frame
size, input/output encodings, alpha policy and `frames_submitted`. Its
`submission_status=submitted_unverified` acknowledges a native write call.
OpenCV does not return a per-frame encoding acknowledgement. Verify the
finalized file before treating it as a durable recording; ticket metadata
remains in graph connections and is not serialized into the video file.

## Qualify a codec on this machine

```powershell
python -m opencv_dataflow.video_qualification --output-dir new_video_qualification
```

The destination must be new. The command retains six original generated clips
and `report.json`, checking native count/EOF, order, frame size/depth, FPS,
media time, moving-part center and color. FFV1 requires exact pixels. Lossy
cases require mean error no larger than eight counts in each channel and
interior color-patch mean error no larger than twelve counts. These limits
qualify the supplied synthetic fixture, not arbitrary scene fidelity.
Missing codecs report `unavailable` and return a nonzero exit code. Select
one or more `--case` values to qualify another build incrementally.

The pinned OpenCV 5 build passes these combinations:

| Encoder backend | Codec/container | Output | FPS |
| --- | --- | --- | --- |
| FFmpeg | FFV1 / AVI | BGR and gray, exact pixels | 12.5 |
| FFmpeg | MJPG / AVI | BGR and gray, bounded loss | 12.5 |
| FFmpeg | mp4v / MP4 | BGR, bounded loss | 12.5 |
| OpenCV MJPEG | MJPG / AVI | BGR, bounded loss | 10 |

Native probing found that OpenCV MJPEG rounded fractional FPS and emitted
corrupted grayscale frames. The recorder rejects either combination when
the actual backend is `CV_MJPEG`, including selection through `auto`. Use
FFmpeg for fractional FPS or gray output. A rejected open removes only the
newly reserved output; an existing file is preserved. Failed parameter
changes during recording do not append a frame or reset the valid stream.

Open the [original recording/replay example](<../examples/opencv tools/video_codec_roundtrip/README.md>)
to try this in the editor. Automated tests also cover tagged color conversion,
custom FourCC, invalid inputs/backend combinations, recovery, replay ranges,
looping, saved cursors and recording from a background queue.

The decoder backend governs seeking, timestamps and codec support. The reader
uses decoder EOF when no explicit range end is set. A decoder can report an
interruption as end of stream. These local checks do not qualify network,
hardware-accelerated, variable-rate, high-bit-depth or other codec/backend
combinations. See the [OpenCV Video I/O property reference](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html)
and [VideoWriter API](https://docs.opencv.org/5.0/main_modules/classcv_1_1VideoWriter.html).
