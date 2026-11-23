# Match binary line descriptors

Open [line_descriptors.json](line_descriptors.json) and choose **Run Once**.
**Describe query lines** and **Describe train lines** detect segments and
generate 32-byte Line Band Descriptors (LBD). **Match described lines** applies
the two-neighbor distance ratio and reciprocal nearest matching. **Review line
pairs** shows both segments and connects their midpoints.

The train image is the query image rotated 3° about (176, 129), then shifted
by (16, −9) pixels. The graph found 25 and 26 descriptors and accepted
22 pairs. Every pair's endpoints agree with the known transform within
2.80 pixels. The median of the larger endpoint error per pair is 0.67 pixels.
Native Hamming distances range from 1 to 18 differing bits.

The descriptor and matcher blocks are under **OpenCV → Line Descriptors**.
The descriptor computes one native octave per image. Use the image input's
pyramid level control to describe a smaller application pyramid level.
Outputs retain endpoints, frame and level information for both views.
Mask and region selection keeps lines whose two endpoints are inside the
selection. Descriptor sets need the same band width to match.
[OpenCV line features tutorial](https://docs.opencv.org/5.0/tutorials_contrib/line_descriptor/tutorial.html)

The graph and both images are original MIT assets. Rebuild from the
repository root.

    python "examples/opencv tools/line_descriptors/build_example.py" --output "examples/opencv tools/line_descriptors"
