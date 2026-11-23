# Fast line detection

Open [fast_line_detection.json](fast_line_detection.json) and click **Run**.
Compare the arrows in **Detect on intensity image** and **Detect using edge
map**. The first detector runs its own Canny stage. The second receives
the upstream Canny image and uses aperture **0**. **Longest segments first**
sorts the intensity branch's line collection by length.

The detector exposes minimum length, fitting distance, both Canny
thresholds, Canny aperture, and incremental segment merging. Apertures **3**,
**5**, and **7** run Canny on an 8-bit grayscale conversion. Aperture **0**
takes a monochrome 8-bit edge image and interprets nonzero pixels as edges.
OpenCV can split long physical edges into several image segments.

The native endpoint order is available in **Segments** and shown by the
result arrows. The collection's angle is the undirected orientation in
[0°, 180°), measured from the image's rightward x-axis with y increasing
downward. The collection carries source pixel geometry, spatial context, and
pyramid context for downstream measurement. A wired mask or region keeps
segments whose two endpoints are selected. The detector measures the whole
image before filtering, so a segment may cross an unselected interior.
The image output preserves the original pixels.

Both paths measured the horizontal and vertical references within about one
pixel and recovered a 45° diagonal longer than 60 pixels. The target
is an original MIT-licensed synthetic PNG. Rebuild with `build_example.py`
in the application environment. [OpenCV Fast Line Detector reference](https://docs.opencv.org/5.0/extra_modules/ximgproc_fast_line_detector.html)
