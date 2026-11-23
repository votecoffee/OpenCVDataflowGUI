# Structured forest edges

Prepare the model once from the repository root using the application's Python
environment. The script downloads the OpenCV 5.0.0 reference forest to the
example's local model folder and verifies its size and SHA-256.

    python "examples/opencv tools/structured_edges/prepare_model.py"

Open [structured_edges.json](structured_edges.json) and choose **Run Once**.
Review the **Edges** mask in **Detect and thin edges**, compare it with
**Reference boundaries**, and inspect the grayscale **View continuous thinned
response**. The detector found 460 pixels above the configured threshold.
Of those, 459 were within three pixels horizontally and vertically of a marked
boundary. Every reference pixel had a detection within that same window.

**Structured forest edges** is under **OpenCV → Extended Image Processing →
Edges**. It accepts 8-bit, 16-bit, or normalized float32 color and grayscale.
It runs inference on the full image and zeros the result outside an optional
mask or region. The raw response, edge-normal orientation in radians, thinned
response and binary edge mask share the input image's coordinate frame.
Preserve the float response and orientation for downstream algorithms that need
edge strength and direction. The raw response can exceed 1. [OpenCV 5 API reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ximgproc_1_1StructuredEdgeDetection.html)

The example uses the exact bytes of the OpenCV release forest. Model acquisition
is separate from execution, so the graph can run offline after setup. The sample
images and project are original MIT assets. The structured forests method is
credited to Piotr Dollár and C. Lawrence Zitnick. [OpenCV release model](https://github.com/opencv/opencv_extra/blob/3becd947dd6bf6340047de8ca4445a70271e6073/testdata/cv/ximgproc/model.yml.gz) · [Structured Forests for Fast Edge Detection](https://openaccess.thecvf.com/content_iccv_2013/papers/Dollar_Structured_Forests_for_2013_ICCV_paper.pdf)

Rebuild the project and the two sample images from the repository root.

    python "examples/opencv tools/structured_edges/build_example.py" --output "examples/opencv tools/structured_edges"
