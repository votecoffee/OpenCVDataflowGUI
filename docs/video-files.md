# Video file workflows

## Replay a clip

1. Add **Input > Video > Video file** and select a local video.
2. Set the zero-based start frame, optional exclusive end frame and stride.
   For example, start 1, end 6 and stride 2 emits frames 1, 3 and 5.
3. Connect the image to an inspection graph. Run once advances one frame.
   Continuous stops when finite sources are exhausted. Set repeat to loop
   to restart the selected range at its end. **File > Reset flow state**
   returns the source to its start frame.

Each activation emits a frame ticket such as `clip.avi#000003` by default.
The image payload and `info` port contain the original frame index, time in
milliseconds, timestamp provenance, nominal FPS and decoder backend. When
the decoder supplies no usable time, the block estimates it from the frame
index and FPS when available. Headless replay runs at processing speed.

## Record processed frames

Add **Output > Video > Write video file**, then wire a processed image to
it. Select a new destination, FPS, codec and backend. Start with MJPG in an
AVI file for the tested combination. For a portable graph, select the project
path base and use a path relative to the saved project.

Input frames must be uint8 and have fixed, even dimensions. The recorder can
write BGR or monochrome frames. Playback uses the configured constant FPS
even when the source is variable-rate or the graph skips frames. The
container is finalized when execution closes. Start each new recording at a
new path. Existing files are preserved.

Run a saved finite replay graph from the command line with:

```powershell
opencv-dataflow-run my_video_project.json --all-images
```

The decoder backend governs seek accuracy, timestamps and codec support.
The reader uses decoder EOF when no explicit range end is set. A backend
can report a decoding interruption as end of stream. Codec/container
support varies by build and backend. Automated fixtures verify MJPG AVI
recording, replay, stride, loop, reopening at a saved cursor and recording
from a background queue. See the [OpenCV Video I/O property reference](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html).
