# Optional Lena test asset

Run `python scripts/prepare_lena.py` from the repository to fetch the image needed
by `examples/lena.json`. The download is explicit and the image is ignored by Git.

Source: [OpenCV samples/data/lena.jpg at commit 1007c06](https://github.com/opencv/opencv/blob/1007c06d44a118566bbc0200577942423b38911d/samples/data/lena.jpg).

SHA-256: `7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07`.

This is a third-party test image, not an original project asset. The repository's
MIT license does not grant rights to the image. It is fetched only when requested
and is not bundled into the Python package or committed in this repository.

`datamatrix_lot.png` is a synthetic Data Matrix generated with ZXing-C++
from `LOT-0042` at scale 8 and padded with a white border for the barcode example.

`synthetic_color_chart.png` draws the OpenCV MCC24 reference colors with a
synthetic channel cast. It exercises the chart-detection and correction
workflow without requiring a photographed target.
