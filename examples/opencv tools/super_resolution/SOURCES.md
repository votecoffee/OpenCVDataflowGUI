# Sources and ground truth

- Images, fixture generation and graph definitions: original MIT work by
  OpenCVDataflowGUI contributors. The native-resolution 256x192 reference has a
  radius-36 dark disk, a rectangular slot, a bead with a two-pixel gap and four
  cracks one through four pixels wide. The seeded noisy version adds Gaussian
  sample noise before the same area reduction. Color, alpha and uint16 variants
  preserve known source values. These are controlled tool fixtures, not real
  production captures or an OCR/detection benchmark.
- Unmodified pretrained model:
  [Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow/tree/6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d),
  revision `6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d`,
  `models/FSRCNN_x2.pb`, 38,973 bytes, SHA256
  `366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c`.
  Apache 2.0; the unchanged upstream license is retained in `MODEL-LICENSE.txt`
  (SHA256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
  The upstream project identifies its training/fine-tuning sets as T91 and
  General100 and describes its sub-pixel variant of FSRCNN. No training data
  or upstream example images are redistributed here. Runtime uses OpenCV's
  frozen-model importer; TensorFlow is not an application dependency.
- Native behavior is checked against
  [OpenCV 5.0.0 dnn_superres source](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/dnn_superres/src/dnn_superres.cpp).
  The native path controls its own channel preparation and reconstruction.
  Wrapper tests compare exact native pixels and separately verify metadata,
  selection, alpha, quantization, integrity and lifecycle behavior.

`truth.json` retains asset hashes and independently specified feature sizes/ROIs.
`benchmark.py` calls the native model directly and compares its result, linear,
cubic and Lanczos interpolation with original source pixels. `benchmark.json`
retains the measured errors, thresholded sizes, missing/false detail pixels and
flat-field statistics. It is separate from block execution and never substitutes
reconstructed output for the native-resolution reference. Threshold 110 and
the stated ROIs define these particular feature tests. No customer content or
private-repository assets are included. Each graph embeds attribution.
