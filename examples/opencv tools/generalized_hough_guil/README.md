# Rotated and scaled Hough matching

Open [generalized_hough_guil.json](generalized_hough_guil.json) with
**File → Open…** and click **Run once**. The upper Ballard branch locates
the unchanged part. The lower Guil branch searches a region around the
enlarged part and finds it near `[117, 81]`, at 1.25× scale and 90°
clockwise. The overlay outlines the transformed template and marks its
anchor.

Find **Find rotated and scaled Hough template** under **Inspection →
Matching**. Connect a search image and a cropped template. The angle
search uses clockwise image angles from 0° through 360°. The returned pose
uses the graph's counterclockwise convention, so this example's pose angle
is −90°. The angle and scale step sizes set the search grid resolution.
The match record includes the estimated scale, transformed corners, and
the position, scale, and angle vote counts. A scale below 1 can locate a
reduced part even when its original template image is larger than the scene.

The block ranks matches by position votes and applies the mask or region
to anchors before suppressing nearby results. The search has a work limit
that accounts for edge density, feature pair buffers, and the angle and
scale grids. A tighter upstream crop, narrower ranges, or coarser steps
reduce the work.
[OpenCV Guil reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1GeneralizedHoughGuil.html)

```powershell
opencv-dataflow-run "examples/opencv tools/generalized_hough_guil/generalized_hough_guil.json"
```

[All examples](../../README.md)
