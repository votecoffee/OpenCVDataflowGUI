# Fast Hough lines

Open [fast_hough_lines.json](fast_hough_lines.json) and choose **Run Once**.
The edge map contains three long lines and two short distractors. **Find
three line peaks** overlays the three recovered lines. **Scale accumulator
for display** makes the bin responses visible in grayscale. **Review vertical
line** shows the result of filtering the line collection by direction.

| Line | Source endpoints (px) | Deskewed bin (column, row) | Sum response | Signed normal offset (px) |
| --- | --- | --- | ---: | ---: |
| Horizontal | (0, 48) to (147, 48) | (121, 353) | 37,740 | 48.00 |
| Oblique | (0, 25) to (147, 82) | (127, 410) | 27,540 | 23.31 |
| Vertical | (36, 0) to (36, 103) | (88, 103) | 26,520 | 36.00 |

**Fast Hough lines** is under **OpenCV → Extended Image Processing →
Features**. The block takes an 8-bit monochrome edge map, such as the output
of **Canny**. The optional mask and region restrict which pixels contribute.
A recovered line extends to the image boundary and can extend beyond the
pixels that supported the peak. Use **Clip line segments** to limit the
visible line to a rectangular inspection area.

The signed normal offset is the dot product of a point on the line and its
unit normal. The reported normal angle is clockwise from the image X axis in
the range [0°, 180°). The collection's angle measures the line's direction,
so a vertical line has a direction of 90° and a normal angle of 0°.

The raw and deskewed layouts use different bin coordinates and convert back
to the same source lines on this target. Bin space gets its own coordinate
frame. The raw numerical accumulator is separate from the normalized preview.
The accumulator is limited to 8,388,608 cells. For larger images, try a crop,
a lower pyramid level, or a narrower angle range. [OpenCV Fast Hough implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/fast_hough_transform.cpp)

The graph, generator and edge image are original MIT assets. Rebuild them
from the repository root after installing the application.

    python "examples/opencv tools/fast_hough_lines/build_example.py" --output "examples/opencv tools/fast_hough_lines"
