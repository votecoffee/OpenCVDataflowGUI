# Harris–Laplace interest points

Open [harris_laplace.json](harris_laplace.json) and choose **Run Once**.
The upper branch detects Harris–Laplace points in the original image. The
lower branch doubles its width and height with linear interpolation and runs
the same detector. The circles show the native keypoint diameters.

| Measurement | Result |
| --- | ---: |
| Original and doubled detections | 205 and 177 |
| One-to-one pairs within 2.5 pixels after the known resize | 165 |
| Repeatability relative to the smaller eligible set | 93.2% |
| Median center error on the enlarged image | 0.71 px |
| Median paired diameter ratio | 2.00× |

**Harris–Laplace interest points** is under **OpenCV → Extra 2D Features →
Features**. Its keypoint record preserves the center, response, diameter, and
detector octave in the selected input image's frame. It marks orientation
unknown. The scale-space controls include the octave count, two or four
layers per octave, and response thresholds. Masks and regions select by
center. The native pyramid starts with a doubled image, and the block
accepts up to 1,048,576 source pixels.
[OpenCV Harris–Laplace reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1HarrisLaplaceFeatureDetector.html)

Both branches feed **BEBLID descriptors** to **Measure 2× repeatability**. The
evaluation uses the resize's known pixel-center transform for geometry, and
reports descriptor matching separately. At a 20-bit Hamming cutoff, it found
137 unique correct descriptor matches.

The graph and target are original MIT assets. Rebuild from the repository root.

    python "examples/opencv tools/harris_laplace/build_example.py" --output "examples/opencv tools/harris_laplace"
