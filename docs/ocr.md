# OCR workflows

## Read a known text region

Add **ML > Text inspection > Recognize text (CTC ONNX)**. Supply a local
CTC recognition model and its ordered vocabulary. Use the inline list or a
UTF-8 token file with one token per line. The vocabulary excludes the blank
class at index zero. The model must emit a single CTC tensor with shape
`[time, 1, vocabulary size + 1]`. Prefix beam search needs probabilities.
Greedy decoding also accepts logits. Set the input width, height, color
order, mean and scale to match training.
See the [OpenCV text spotting tutorial](https://docs.opencv.org/5.0/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.html).

Wire a cropped line image to the block, or connect a mask or region on the
source image. A selected region uses its bounding rectangle and fills
unselected pixels with the configured background intensity. The block
emits joined text, individual reads and a count. A minimum read count can
fail the inspection. Recognition uses a supplied model. The test network
checks the data path with synthetic values.

## Detect text before reading it

1. Connect the source image to **DNN ONNX inference**, configured for a
   model that emits one text probability map. Typical map layouts include
   `[height, width]` and `[1, 1, height, width]`.
2. Wire the original image and both DNN outputs to **Text regions from
   probability map**. Choose the tensor output, probability thresholds,
   minimum contour area and rectangle padding for the application.
3. Connect its regions output and the original image to **Recognize text**.
   The recognizer rectifies each quadrilateral to the model input size.

The region adapter restores coordinates through the preprocessing transform,
including letterbox padding. Regions are ordered by image center Y and then X.
Its reported confidence is the mean detector probability inside the contour.
Reading multi-line text in a particular layout may require cropping or grouping
the detected regions first.


## Verify the decoded text

Connect an OCR `text` or `reads` output, or a barcode reader's `codes`
output, to **Verify decoded text**. Configure an exact value, prefix, suffix
or required substring. Length and allowed-character rules work independently.
For multiple reads, choose whether any or all must pass. The block reports
individual failures and preserves leading zeroes. When checking individual
OCR lines, connect `reads` so each line is evaluated separately.

## OpenCV Zoo English CRNN settings

For the [OpenCV Zoo English CRNN model](https://huggingface.co/opencv/opencv_zoo/blob/main/models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx),
use grayscale, a width of 100, a height of 32, mean `[127.5]`, scale
`0.00784313725490196`, the classic engine and greedy decoding. The inline
vocabulary contains digits 0–9 followed by lowercase a–z, in that order.
A smoke test recognized the rendered word `opencv` and read `lot123` as
`lot12z`. Exact verification catches that mismatch when the expected text
is known.
