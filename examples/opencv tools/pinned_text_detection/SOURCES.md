# Sources and image rights

- **Model:** PaddlePaddle Authors, PP-OCRv3 English text detector, distributed by
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/text_detection_ppocr).
  Revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`; artifact
  `text_detection_en_ppocrv3_2023may.onnx`, 2,423,490 bytes, SHA256
  `03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587`.
  Weights are unchanged. Apache 2.0; the upstream model directory's license is
  retained in `MODEL-LICENSE.txt`. Preparation downloads explicitly into ignored
  `.local/ppocr_en`, checks exact bytes, and seals the image/execution contract.
- **Preprocessing:** the pinned Zoo `ppocr_det.py` uses BGR, mean
  `[123.675,116.28,103.53]`, scale `1/255`, standard deviations
  `[0.229,0.224,0.225]`. This workflow fits the image to 736x736 with a white
  letterbox, preserving the inverse source mapping. The native network output
  is the named 736x736 probability map `4`; input is `x`.
- **Images/annotations/graphs:** original MIT fixtures by OpenCVDataflowGUI
  contributors, rendered with OpenCV Hershey glyphs. No customer images,
  pretrained-generated labels or copied third-party sample photographs. Source
  glyph bounds and the independent 12-degree affine transformation supply
  the geometric reference; `truth.json` stores image hashes and exact polygons.
- **DB-style expansion:** the decoder fits an external probability contour and
  expands its rectangle by the area/perimeter distance. See the
  [OpenCV 5.0 native DB implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/model.cpp)
  and [official text-spotting sample](https://github.com/opencv/opencv/blob/5.0.0/samples/dnn/text_detection.cpp).
  Our head-grid expansion, half-pixel mapping, external-only contours and ROI
  policies are explicit; this is not a promise of identical native DB polygons.

Joined/split examples test spatial grouping, not semantic word segmentation or
character recognition. The numeric fixture labels are not extracted OCR text.
