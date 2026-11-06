# Fourier correlation

Open [fourier_correlation.json](fourier_correlation.json) with **File → Open…**,
then click **Run once**. A warp moves the reference by 17 pixels right and 9
pixels up. **Fourier correlation** recovers the displacement (+17, −9) with
an energy-normalized peak of 1. The lag-map preview marks the peak.

**Fourier correlation** is under **Inspection → Registration**. It takes
two monochrome images. The signed **Shift** locates **Reference** in **Image**.
**Linear** zero-pads the inputs, accepts different image sizes, and reports
all displacements with possible overlap. **Circular** requires equal sizes,
wraps at the image edges, and reports an even-sized half-period as a negative
lag.

**Energy** normalization divides each signal by its whole-image L2 norm
after masking and optional mean subtraction. The same normalization applies
to every lag. Choose **None** for raw sums of pixel products. A mask or
region selects the sample pixels, and **Reference mask** selects the
reference pixels. Excluded pixels contribute zero. **Subtract mean** centers
each signal using its selected pixels.

The float64 **Correlation** output is a lag map. Its `lag_origin` payload
gives the [x, y] index for zero displacement. The block accepts up to
8,388,608 pixels in the padded transform. [OpenCV spectral operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html#mulspectrums)

```powershell
opencv-dataflow-run examples/pattern_matching/fourier_correlation/fourier_correlation.json
```

[All examples](../../README.md)
