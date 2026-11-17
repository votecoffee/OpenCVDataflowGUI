# Sources and evidence

- **Dynamic models and stripes:** original MIT fixtures by OpenCVDataflowGUI
  contributors. The tiny native ONNX graph maps five grayscale columns to CTC
  logits, with dynamic N and explicit TNC/NTC variants. These are arithmetic and
  batching controls, not trained OCR models or accuracy evidence.
- **Printed labels:** unchanged original MIT image from the neighboring
  [pinned text detection example](../pinned_text_detection/README.md). The known
  glyph bounds supply source quadrilaterals independently of recognition.
  Expected native strings are lowercase because the model's English vocabulary
  contains digits followed by lowercase Latin letters.
- **Native CRNN:** unchanged
  [OpenCV Zoo English export](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/text_recognition_crnn),
  revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`, file
  `text_recognition_CRNN_EN_2021sep.onnx`; 33,823,087 bytes, SHA256
  `a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544`.
  Apache 2.0; upstream license retained in `MODEL-LICENSE.txt`. The Zoo helper
  credits Shenzhen Institute of Artificial Intelligence and Robotics for Society
  and directs training provenance to the deep-text-recognition benchmark.
- **Native preparation:** grayscale, 100x32, mean 127.5, scale 1/127.5, blank
  index zero, digit/lowercase vocabulary. Input `input`, output `output`, native
  TNC output [24,1,37]. This exact export supports only N=1 in this environment;
  N=2 was rejected. The saved native baseline declares fixed N=1 and executes
  three ordered calls. It is not modified or described as a dynamic export.

`truth.json` records image/model hashes, explicit source quads, dynamic paths and
expected strings. Every graph embeds attribution. No customer images or private
repository material are included. Greedy-path confidence is not a calibrated
probability of a complete string being correct.
