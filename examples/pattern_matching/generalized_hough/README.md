# Generalized Hough matching

Open [generalized_hough.json](generalized_hough.json) with **File → Open…**
and click **Run once**. The scene contains two copies of the same asymmetric
part. The first branch finds anchors at `[40, 31]` and `[116, 79]`, with
122 votes each. The second branch uses a rectangular region and keeps
only the lower-right match.

Find **Find Hough template** under **Inspection → Matching**. Connect a
search image and a cropped template. Ballard matching uses Canny edges
and gradient directions to locate translated copies at the template's
size and orientation. **Template anchor** sets the origin of each output
pose. The default is the integer center of the template image.

**Levels** sets the number of gradient direction bins. **Dp** sets the
accumulator spacing in source pixels. Matches are ranked by vote count.
A mask or region filters candidate anchors before minimum-distance
suppression. Use **Minimum matches** to set the pass criterion. The
overlay labels each retained match with its vote count and marks its
anchor.

The search accepts up to eight million image pixels and one million
template pixels. Dense edges may exceed the voting budget. Crop the
images or raise the Canny thresholds in that case.
[OpenCV generalized Hough tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/generalized_hough_ballard_guil/generalized_hough_ballard_guil.html)

```powershell
opencv-dataflow-run examples/pattern_matching/generalized_hough/generalized_hough.json
```

[All examples](../../README.md)
