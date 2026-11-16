# Rolling-shutter row-motion correction

Open a project and press **Run once**. Select **Estimate rolling-shutter motion**
for fitting tracks, coefficients, validation evidence and supported rows. Select
**Warp image with dense flow** for the corrected current image, coverage and
output-to-source maps. No plugin or download is needed.

| Project | What it demonstrates |
| --- | --- |
| [project.json](project.json) | Known horizontal and vertical cubic scanline motion. Native corner/LK fitting corrects bent lines and reduces error against an independent sharp reference. |
| [video.json](video.json) | Six FFV1 frames at 10 fps with zero, positive and negative row-motion amplitudes. Use **Run all** to process the complete clip or **Run once** to advance a frame. |
| [highdepth.json](highdepth.json) | Restore 16-bit BGR pixels while tracking in a shared 8-bit working representation. |
| [moving_part.json](moving_part.json) | Fit only the left static strip. Correct the current part at the right, preserving its pixels instead of copying the empty reference. |
| [no_texture.json](no_texture.json) | Intentionally fail with insufficient tracks; no motion field or corrected image emits. |
| [row_gap.json](row_gap.json) | Intentionally fail because fitting evidence covers only a narrow middle band. |

Green fitting arrows are training inliers; yellow arrows are withheld validation
tracks. The report lists every pair, residual evidence and failure reason.
**Support** describes where the row fit can be used; **Coverage** additionally
checks current-image donor support and resampling boundaries. Both are important.
The current capture keeps its ticket and video timestamp; corrected coordinates
belong to the reference grid. Maps record their nonlinear relationship to the
source.

The before/after RMSE blocks compare the same covered pixel indices against
separately generated sharp truth. The moving-part truth contains the new part;
its fitting reference does not. Interpolation means corrected pixel values need
not match sharp truth exactly. Automated checks measure error reduction, known
row geometry near both image edges, part position, coverage and timing in both
Debug and Lean modes.

This is a reference-based row-translation baseline, requiring static texture
across image rows. It does not model general projective rotation, parallax or
arbitrary vibration. Track validation cannot prove absence of defects. See
[parameters, failure behavior and limits](../../../docs/rolling-shutter.md).

All seven PNG images, the FFV1 clip, projects and
[generator](build_example.py) are original OpenCVDataflowGUI fixtures under the
repository's [MIT license](../../../LICENSE). They contain no customer images or
third-party assets. Every project includes an attribution comment. Regenerate
with the application Python:

```powershell
python "examples/opencv tools/rolling_shutter/build_example.py"
```

The [truth manifest](truth.json) records analytic coefficients, video amplitudes,
timing, part coordinates and asset SHA256 hashes. Tests verify exact regeneration
and lossless video sample readback. Underlying tracking uses OpenCV's
[Lucas-Kanade API](https://docs.opencv.org/5.0/main_modules/video_track.html).
