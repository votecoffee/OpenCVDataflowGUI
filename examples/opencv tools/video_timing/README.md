# Video timing controls

Open a project and choose **Run All**. Each clip contains eight independently
timed images. Select the timing block to inspect Report, status and the optional
Chart; double-click results to enlarge them. Timing and blob graphics overlay the
unchanged current image in the filmstrip. Deliberate clock failures keep their
evidence available and produce failed judgments with reasons.

| Project | Expected final observation |
| --- | --- |
| [constant.json](constant.json) | Eight valid frames, constant 40 ms intervals, no rejected rows. |
| [variable.json](variable.json) | Valid variable intervals at 0,40,130,160,340,380,550,600 ms; no rejection. |
| [duplicate.json](duplicate.json) | Two repeated timestamps; two rejected rows and latched failure. |
| [backward.json](backward.json) | One backward timestamp; one rejected row and latched failure. |
| [zero_time.json](zero_time.json) | Missing DefaultDuration and all raw positions zero. Replay's labeled estimates stay visible; seven rows reject strict timing. |
| [wrong_rate.json](wrong_rate.json) | Declared 10 FPS but actual 40 ms intervals; seven rows reject Match nominal FPS. |
| [wrong_reference.json](wrong_reference.json) | Valid variable timing disagrees with a constant-time reference; six rows reject. |
| [paced_variable.json](paced_variable.json) | Valid irregular media intervals replay at speed two. Cycle start resets timing history when Continuous loops. |

Failures remain latched until Reset by default. Switch Latch failures off to judge
only the current row. An unpaced clip's Continuous mode also loops; use Run All
for the single-clip report, or wire an explicit reset when a loop should start over.

```powershell
opencv-dataflow-api run "examples/opencv tools/video_timing/variable.json" --all-images --max-runs 12 --summary
opencv-dataflow-api run "examples/opencv tools/video_timing/wrong_rate.json" --all-images --max-runs 12 --summary
```

The second command deliberately returns failure. Its image-loading and blob
measurement can succeed while the timing monitor rejects the clock.

Images, fixtures, generator, independent annotations and projects are original
OpenCVDataflowGUI contributor work under the repository [MIT license](../../../LICENSE).
There are no customer or third-party images. Each graph has an attribution comment.
`truth.json` pins six complete MKV hashes, actual block times and the moving part's
known center `[31.5 + 8*i, 41.5]` and area 672. JPEG is lossy; these threshold-based
geometric measurements remain exact. Native timestamps are not used to author truth.

`build_example.py` writes a small original Matroska/MJPEG fixture subset directly:
1 ms TimestampScale, unlaced keyframe SimpleBlocks, fixed dimensions and separately
declared (or absent) DefaultDuration. It is a fixture generator, not a general
muxer. See [Matroska elements](https://www.matroska.org/technical/elements.html) and
[Motion JPEG mapping](https://www.matroska.org/technical/codec_specs.html#v_mjpeg).
No separate FFmpeg executable or Python media package is needed. Regenerate in a
fresh folder; existing clips with different bytes are rejected.

The tests independently parse wire timestamps, verify native pixels/known centers,
exercise variable-rate pacing, start/stride/close-resume and time sampling, and
check source/monitor ticket identity. This is software/decoder qualification on
the supported build, not exposure-time, physical-camera or transport certification.
See [timing monitor contracts](../../../docs/video-timing.md) for full controls.
