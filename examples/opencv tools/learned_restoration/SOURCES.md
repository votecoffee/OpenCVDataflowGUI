# Sources, conversion and independent references

- Images, arithmetic models and graph definitions: original MIT work by
  OpenCVDataflowGUI contributors. Clean/noisy pairs retain a known circle,
  strong and faint one-pixel cracks and fine low-contrast texture. Noise is
  generated with a recorded seed before integer clipping/rounding. Averaging
  models are numerical controls, not learned denoisers. No customer or private
  repository content is included.
- Optional native DnCNN-25 weights: Kai Zhang's
  [KAIR v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0),
  `dncnn_25.pth`, 2,224,903 bytes, SHA256
  `0451a70de9b672ae037270498fbb1c17a1c1c4403785df586ff65df5b858e5b0`.
  MIT; the unmodified upstream notice is retained in `MODEL-LICENSE.txt`
  (1,066 bytes, SHA256
  `448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5`).
  Code/license reference revision `fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`.
  [Author's test configuration](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/main_test_dncnn.py)
  and [network definition](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/models/network_dncnn.py)
  identify the grayscale, 17-convolution variant with already fused batch norms.
- `prepare_model.py` verifies exact source bytes before weights-only CPU loading.
  It independently builds ONNX Conv/ReLU operators with the same frozen float32
  tensors and returns input minus predicted noise. No upstream Python code is
  imported or executed, and weights are not retrained or modified. Dynamic
  `[1,1,H,W]`, opset 13, IR 8, input `input`, output `output`; unit grayscale.
  Export: 2,223,887 bytes, SHA256
  `6e9058f77ac18172ac201eeb9148fc57d5e839c1435e203eb1b03174878106c3`.
  PyTorch/ONNX are preparation/reference tools only; application inference is OpenCV.
- `benchmark.py` independently runs the exported native model and compares
  original pixels, Gaussian/median baselines, whole-image and tiled inference.
  An optional source-tensor reference uses direct CPU convolution/ReLU arithmetic.
  The retained maximum export/reference difference is 2.2352e-6 normalized units.
  This measures this conversion/model, not arbitrary model interchangeability.

`truth.json` retains all public image/model/contract hashes, source dimensions,
feature definitions and evaluation ROIs. `benchmark.json` retains global errors,
flat noise, false/erased crack pixels and context-margin comparisons. The faint
crack uses threshold 150; the strong crack uses 100. All references are original
known pixels, never a reconstruction presented as ground truth. Prepared native
weights/bundles remain ignored in `.local`; the public graphs carry attribution.
