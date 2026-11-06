# Discrete cosine transform round trip

Open [dct_round_trip.json](dct_round_trip.json) with **File → Open…**, then
click **Run once**. The graph computes the cosine coefficients of the
bundled monochrome image, reconstructs it, and checks that the mean squared
pixel error is zero.

**Discrete cosine transform (DCT)** and **Inverse discrete cosine
transform (IDCT)** are under **Image processing → Spectral**. DCT can
transform the whole image or each row independently. Odd transformed
dimensions can be rejected or extended by repeating their last pixel.
The coefficient packet retains the original size, source depth and mode.
IDCT uses that information to crop the padding and restore the original
depth. The DCT preview displays the log magnitude of the coefficients.
[OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html#dct)

```powershell
opencv-dataflow-run examples/image_processing/dct_round_trip/dct_round_trip.json
```

[All examples](../../README.md)
