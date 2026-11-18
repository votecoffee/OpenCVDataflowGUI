# Sources and independent truth

- Probability model and ambiguous/repeated/faint images: original MIT fixtures
  by OpenCVDataflowGUI contributors. The dynamic NCHW model maps each column's
  intensity v to blank probability 1-v and A probability v, then emits TNC.
  `truth.json` enumerates all eight paths independently. Three 102/255 columns
  give A mass 0.688, blank mass 0.216 and AA mass 0.096. The repeat fixture has
  exact A/blank/A input. These are arithmetic controls, not trained OCR weights.
- Stripe image and dynamic logit model: unchanged original MIT assets from the
  neighboring [batch OCR example](../ocr_batch/README.md). The raw read is ABA;
  restricted alphabet/lexicon cases verify explicit rejection and likelihood
  gates. The printed label image is also an unchanged original MIT fixture.
- Optional native CRNN: unchanged
  [OpenCV Zoo English export](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/text_recognition_crnn),
  revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`,
  `text_recognition_CRNN_EN_2021sep.onnx`, 33,823,087 bytes, SHA256
  `a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544`.
  Apache 2.0; the exact upstream license is retained in `MODEL-LICENSE.txt`.
  The Zoo helper credits Shenzhen Institute of Artificial Intelligence and
  Robotics for Society and points to the deep-text-recognition benchmark.
  Grayscale 100x32, mean 127.5, scale 1/127.5, blank zero, digits/lowercase
  vocabulary; input `input`, output `output`, TNC [24,1,37], fixed N=1.
  The independently printed strings are lot2048, pnax12 and pass. This example
  adds those exact lexicon words and rejection settings, leaving weights intact.
- CTC algorithm background: [Awni Hannun, Sequence Modeling with CTC (2017)](https://distill.pub/2017/ctc/).
  The implementation and exhaustive oracle are original project code; no
  third-party decoder code was copied.

Every saved graph embeds attribution. All model/image bytes are hashed in
`truth.json`, except the explicitly prepared native weights whose identity is
sealed in their ignored local contract. No private repository material or
customer images are included. Scores establish model likelihood contracts,
not calibrated text correctness or machine acceptance.
