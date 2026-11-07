# Shared example assets

## Optional Lena test image

Run `python scripts/prepare_lena.py` from the repository to fetch the image needed
by the [Lena inspection](../demos/lena/README.md), [features](../demos/lena_features/README.md),
and [registration](<../opencv tools/lena_registration/README.md>) examples. The download is
explicit and the image is ignored by Git.

Source: [OpenCV samples/data/lena.jpg at commit 1007c06](https://github.com/opencv/opencv/blob/1007c06d44a118566bbc0200577942423b38911d/samples/data/lena.jpg).

SHA-256: `7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07`.

This is a third-party test image, not an original project asset. The repository's
MIT license does not grant rights to the image. It is fetched only when requested
and is not bundled into the Python package or committed in this repository.

## Bundled fixtures

- `rgb_average.onnx` is a tiny synthetic RGB-average model used by the DNN and
  Annoy examples. It is generated with `scripts/prepare_ml_assets.py`.
- `validation/` contains the synthetic pass, fail, and unknown images shared
  by the validation and image-set workspace examples.
- The [recorded stabilization fixtures](<../custom tools/recorded_camera_stabilization/README.md>)
  use a trimmed, grayscale TUM RGB-D recording under CC BY 4.0.
- The [recorded depth variation fixtures](<../custom tools/recorded_stabilization_validation/README.md>)
  use eight reduced TUM RGB-D frames and two registered depth images under CC BY 4.0.
