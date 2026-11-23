# Shape-context contour comparison

Open [shape_context.json](shape_context.json) and choose **Run Once**. Select any
**Score** block to see the distance. The three **Find contours** blocks show the
silhouettes used for comparison.

| Compared with the reference part | Distance |
| --- | ---: |
| Same outline | 0.0148 |
| Mildly deformed part | 0.0281 |
| Circle | 0.0650 |

A smaller distance indicates a closer outline. For this fixture, turn on
**Check distance** with the default maximum of **0.05** to accept both part
outlines and reject the circle. The score depends on the settings and input
order. Iterative alignment can give identical silhouettes a small nonzero
score, so measure known examples before selecting a limit.

**Compare contours (shape context)** is under **OpenCV → Shape Distance and
Matching → Contour matching**. Connect ordered closed `[x,y]` point lists or
contour collections. The index fields select one item from each collection.
The block samples 16–128 points at equal intervals along each perimeter and
centers each sample set. This isolates the comparison from the original
positions and uneven contour vertex spacing while retaining the relative sizes.

The block supports thin-plate spline, full affine and partial affine alignment,
four histogram costs, configurable descriptor bins and radii, and optional
rotation-relative descriptors. Earth mover comparisons limit combinations of
sample count, bins and iterations to bound the cost of histogram matching.
The distance uses outline geometry. Appearance-assisted ranking is tracked
separately on the roadmap. [OpenCV shape-context reference](https://docs.opencv.org/5.0/extra_modules/shape.html)

The project and silhouettes are original MIT assets. Rebuild from the repository root.

    python "examples/opencv tools/shape_context/build_example.py" --output "examples/opencv tools/shape_context"
