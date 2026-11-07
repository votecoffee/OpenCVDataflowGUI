# Fourier spectrum filtering

Open [fourier_filter.json](fourier_filter.json) with **File → Open…**, then
click **Run once**. The bundled monochrome image combines a broad brightness
variation with vertical stripes every four pixels. **Filter Fourier spectrum**
rejects the band from 0.20 to 0.30 cycles per pixel, then **Inverse discrete
Fourier transform (IDFT)** reconstructs the image. The four-pixel stripes
disappear and the broad variation remains.

The blocks are under **Image processing → Spectral**.

- **Discrete Fourier transform (DFT)** runs over the whole image or each row.
  Use exact dimensions for periodic samples, or select optimal size to
  zero-pad the transformed axes to OpenCV's next efficient size.
- **Filter Fourier spectrum** supports low-pass, high-pass, band-pass and
  band-stop. Cutoffs are in cycles per pixel, radial for a 2D transform and
  horizontal for a row transform. The mask applies to both components of
  each complex bin.
- **IDFT** crops any padding and defaults to the original image depth.
  Choose a floating output to keep signed high-pass results and values
  outside the integer image range.

The spectrum previews show centered log magnitude. Calculations use the
full complex coefficients. A custom spectrum producer must preserve conjugate
symmetry for real-image reconstruction. [OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run "examples/custom tools/fourier_filter/fourier_filter.json"
```

[All examples](../../README.md)
