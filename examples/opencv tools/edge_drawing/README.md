# Edge Drawing paths and primitives

Open [the project](edge_drawing.json) and choose **Run Once**. The shape plate
contains a filled rectangle, a circle with radius 34 px, and an ellipse with
106 × 60 px axes rotated 27°. **Paths, lines and round fits** shows the edge
map, ordered paths, and separate blue circle and pink ellipse overlays.
**Keep lines ≥ 80 px** selects the rectangle's two long sides.

| Saved grayscale settings | Measured geometry |
| --- | --- |
| Edge paths | 3 ordered native paths with 240, 196 and 340 samples |
| Long lines | Two lines near y = 27.0 and y = 62.0, each approximately 136 px long |
| Circle | Center (85, 154), radius 34.61 px |
| Ellipse | Center (270, 115), full axes 107.31 × 61.31 px, angle 27.22° |

The fit follows the transition around the rasterized artwork. These measurements
use synthetic geometry and the pinned OpenCV build.

The block accepts grayscale and color images. Grayscale ED converts the input
to 8-bit for detection and exposes Prewitt, Sobel, Scharr and LSD gradient
operators, plus optional parameter-free validation. Native color ED uses
8-bit color gradients. Both modes pass through the original image for the
result overlays. The raw gradient output is uint16. Line records preserve
their source path ID. Round fits publish measured geometry without a confidence
score. Results can be sorted, filtered or drawn through the collection blocks.

A mask or region zeros map pixels outside the selection. Paths that touch the
selection are retained in full. Lines require two selected endpoints and a
retained path. Circles and ellipses use their centers. Detection uses the full
input. The block uses an isolated CPU worker with a configurable timeout and
a 1 GiB Windows job memory limit. It accepts up to 1,048,576 input pixels,
4,096 paths and 262,144 path samples. Turn off line or round fitting to use the
edge outputs alone.

The PNG is an original MIT-licensed synthetic asset. Rebuild with
`python build_example.py --output <directory>`.
[OpenCV Edge Drawing reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_edge_drawing.html)
