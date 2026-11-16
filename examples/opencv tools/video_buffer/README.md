# Bounded video buffers

These six projects decode an original lossless FFV1 clip through native FFmpeg
from a bounded in-memory byte reader. Open a project and use **Run once**.
No camera, network, service, download or plugin is needed. The file Path supplies
an immutable snapshot; a producer can instead connect bytes to **Encoded**.

| Project | Expected result |
| --- | --- |
| [selected_frames.json](selected_frames.json) | Decode ordinals 1, 3, 5, 7 as uint16 RGB; preview position 1 is original frame 3. Times are 100, 300, 500, 700 ms. |
| [group_inspection.json](group_inspection.json) | Decode all eight frames, Unbatch, grayscale, threshold and find one bright part per frame. Each inspection carries its frame ticket and presentation time. |
| [short_clip.json](short_clip.json) | Request 16 frames from an eight-frame clip. Allow mode returns eight and explicitly reports the requested count was not satisfied. |
| [read_limit.json](read_limit.json) | One-byte cumulative read budget deliberately rejects opening; no partial output. |
| [decoded_limit.json](decoded_limit.json) | Tiny decoded workspace budget rejects the header before pixel reads. |
| [malformed.json](malformed.json) | Invalid bytes produce a contained decoder error and no output. |

All projects serve at most 257 bytes per native callback to exercise short reads.
Inspect **Info** for selected indexes, resource counters, source hash and
presentation times. These times are not camera acquisition or processing times.
The group inspection processes eight tickets within one graph activation.

```powershell
opencv-dataflow-run "examples/opencv tools/video_buffer/group_inspection.json" --profile debug
opencv-dataflow-run "examples/opencv tools/video_buffer/read_limit.json" --profile lean
```

The second command intentionally returns a failure exit code. Short-clip allow
is successful with an explicit end/decode condition; native read failure alone
cannot prove a clean EOF or container integrity. Controls and current limits are
described in [the decoder contract](../../../docs/video-buffer.md).

Frames, video, annotations and projects are original OpenCVDataflowGUI contributor
work under the repository MIT license. Each graph includes this attribution in a
comment block. No customer or third-party media is included. `truth.json` records
all ten asset hashes, original pixel hashes and part boxes. `build_example.py`
regenerates the eight PNGs, FFV1 AVI, invalid-byte fixture and six saved graphs
deterministically on the supported build.
