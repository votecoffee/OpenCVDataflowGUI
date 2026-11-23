# EdgeBoxes object proposals

Open [edge_boxes.json](edge_boxes.json) and choose **Run Once**. Canny supplies
edge strengths. Two Sobel derivatives and **Cartesian to polar fields** supply
edge-normal angles in degrees. **EdgeBoxes proposals** keeps 12 ranked
rectangles and overlays the two strongest candidates. **Keep high scores**
filters on the collection's **Strength** metric before drawing the final
boxes. The graph runs with the built-in blocks and the included image.

| Part | Reference bounds `[x, y, width, height]` | Highest-ranked matching bounds | Score | Intersection over union |
| --- | --- | --- | ---: | ---: |
| Left | [43, 44, 93, 69] | [40, 43, 97, 72] | 0.416 | 0.919 |
| Right | [215, 112, 75, 77] | [211, 111, 82, 82] | 0.404 | 0.859 |

**EdgeBoxes proposals** is under **OpenCV → Extended Image Processing →
Segmentation**. Its strength and orientation images must use the same pixel
grid. Integer strength maps are scaled to [0,1]. Floating strength maps keep
their original scale. Angle units default to wire metadata, with radians
used for untagged maps. To use a structured forest instead, connect
**Structured forest edges** outputs **Thinned** and **Orientation**. That
pair recovered the same two parts with overlaps of 0.836 and 0.810 on this
target. [OpenCV EdgeBoxes reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_edgeboxes.html)

The optional mask and region filter the centers of the ranked candidates
after the native candidate limit. Additional tuning controls mirror the
native window search, suppression and score. The block stops searches above
one million estimated windows. Larger starting boxes or a lower alpha reduce
search cost.

The graph and target are original MIT assets. Rebuild from the repository root.

    python "examples/opencv tools/edge_boxes/build_example.py" --output "examples/opencv tools/edge_boxes"
