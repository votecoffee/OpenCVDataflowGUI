# Combine Fourier spectra

Open [spectrum_product.json](spectrum_product.json) with **File → Open…** and
click **Run once**. An affine warp shifts a synthetic reference 17 pixels right
and 9 pixels up with wraparound. Separate DFT blocks transform the sample
and reference. **Combine Fourier spectra** multiplies the sample spectrum
by the conjugate of the reference spectrum. **IDFT** reconstructs the circular
lag map, and **Image summary statistics** finds the peak at `[17, 351]`.
On this 360-row grid, row 351 represents a lag of −9.

Find **Combine Fourier spectra** under **Image processing → Spectral**.

- **Multiply** combines two spectra bin by bin for filtering or circular
  convolution. It retains the first spectrum's reconstruction frame.
- **Multiply conjugate** conjugates the second spectrum before multiplication.
  The inverse result uses a separate lag frame with zero lag at `[0, 0]`.
- **Conjugate** uses only the first spectrum. Its inverse reflects the
  transformed axes around circular index zero in a separate frame.

Both inputs must have the same full bin dimensions and transform mode. Mixed
float32 and float64 spectra use float64 arithmetic. IDFT crops any padding to
the first input's original image dimensions. Use floating output to preserve
signed or out-of-range values. The dedicated
[Fourier correlation example](../../pattern_matching/fourier_correlation/README.md)
also reports signed lags and supports zero-padding for linear correlation.
[OpenCV spectral multiplication reference](https://docs.opencv.org/5.0/main_modules/core_array.html#mulspectrums)

```powershell
opencv-dataflow-run examples/image_processing/spectrum_product/spectrum_product.json
```

[All examples](../../README.md)
