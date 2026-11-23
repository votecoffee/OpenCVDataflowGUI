# Find contour ellipses

Open [contour_ellipses.json](contour_ellipses.json) and choose **Run Once**.
The target contains three rotated ellipses among straight edges, a rectangle
and a partial arc. Select **Find three rotated ellipses** to see the detected
outlines and major axes. **Keep elongated ellipses** selects the fit with a
major-to-minor diameter ratio of at least 1.85. **Region from strongest fit**
converts the highest-scoring fit into an inspection region.

| Target center (px) | Detected center (px) | Detected diameters (px) | Target / detected angle |
| --- | --- | --- | --- |
| (90, 85) | (90.07, 85.90) | 106.00 × 61.48 | 27° / 27° |
| (248, 167) | (249.08, 165.52) | 128.00 × 75.52 | 117° / 116° |
| (345, 248) | (346.18, 249.05) | 100.00 × 51.00 | 153° / 153° |

**Find contour ellipses** is in **OpenCV → Extended Image Processing →
Features**. Output angles are clockwise from the image X axis, modulo 180°.
An optional mask or region selects by detection center while preserving the
full contour for detection. The image port preserves the original encoding
and depth. Native detection uses an 8-bit grayscale working image and accepts
up to 2,097,152 pixels and 4,096 pixels on either axis. Use a crop or pyramid
level for larger frames.

The score ranks contour support. OpenCV applies its score and reliability
thresholds before combining them into the reported score, which may be below
the score threshold. Use **Filter object collection** on **Strength** to
apply a cutoff to the reported score. [OpenCV implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/find_ellipses.cpp)

The graph, generator and image are original MIT assets. Rebuild them from
the repository root after installing the application.

    python "examples/opencv tools/contour_ellipses/build_example.py" --output "examples/opencv tools/contour_ellipses"
