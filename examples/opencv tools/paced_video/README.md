# Paced video replay

These projects use an original eight-frame lossless clip at 20 FPS. No plugin or
download is required. The pipeline uses grayscale, threshold and blob detection,
then tracks the part and overlays results in the filmstrip.

| Project | Try |
| --- | --- |
| [Native paced](native_paced.json) | Run All consumes frames 0..7 at their original media intervals. |
| [Strided and faster](strided_fast.json) | Run All emits frames 0,2,4,6 at twice normal speed. Their media timestamps remain 0,100,200,300 ms. |
| [Loop with tracking reset](loop_reset.json) | Continuous repeats frames 2,3,4. Cycle start resets tracking on every wrap, creating a new identity instead of false cross-clip motion. |

The clip is short: normal replay takes about 0.35 seconds of scheduled time.
Set Speed to 0.25 for easier observation or use Continuous. Run All always
rewinds and consumes a range once, including the loop-configured project.
Stop interrupts a pacing wait; the pending frame is kept for Resume. No implicit
frame dropping occurs. Slow processing can make actual playback slower.

Select the video block's Info result to inspect decoder/effective times, timing
source, wait/lateness, cycle and action. The optional Cycle start output is
exposed and connected here. Tracking has a fresh tentative result at each cycle's
first frame, then confirms it; raw media timestamps can safely restart with that
explicit reset. Speed affects wall timing, without changing media-time motion.

All clip pixels, annotations, generator and projects are original
OpenCVDataflowGUI contributor work under the repository MIT license. Each graph
also has an attribution comment. There are no customer or third-party images.
`truth.json` records pixel hashes, intended centers and timestamps. The generator
checks an existing clip instead of overwriting it and rejects different contents.

Regenerate in the application environment, or in a fresh output folder:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/paced_video/build_example.py"
```

The headless loop demonstration uses a bounded repeat count:

```powershell
.venv/Scripts/opencv-dataflow-api.exe run "examples/opencv tools/paced_video/loop_reset.json" --repeat 7 --summary
```

See [timing/Stop/Resume contracts](../../../docs/paced-video.md) and
[all examples](../../README.md). Native ROS, network streams, physical camera
timing and hard real-time performance are outside this example's evidence.
