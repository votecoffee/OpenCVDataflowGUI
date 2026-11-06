# Gabor texture direction

Open [gabor_texture.json](gabor_texture.json) with **File → Open…**
and click **Run once**. Both filters use an 8-pixel wavelength. The
0-degree filter responds to the vertical stripes on the left, and the
90-degree filter responds to the horizontal stripes on the right.
**Compare pixels** marks locations where the vertical response is
stronger. Neighborhoods near the image border and the boundary
between textures mix the two responses.

**Gabor texture response** is under **Image processing → Filtering**.
Set the odd kernel width and height, envelope sigma and aspect ratio,
stripe-normal orientation, wavelength, and phase. Choose **Signed**
to expose a single correlation response or **Energy** to combine two
responses 90 degrees apart. The pair reduces sensitivity to stripe
phase. **Zero mean unit L2** subtracts each kernel's mean and
normalizes its energy, which suppresses constant backgrounds.

The response is float32 or float64 in the source pixel frame.
Color inputs use grayscale conversion. **Kernels** contains the
applied coefficients as numeric tables. A mask or region zeros the
response outside the selection after filtering the full image.
[OpenCV Gabor kernel reference](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html)

```powershell
opencv-dataflow-run examples/image_processing/gabor_texture/gabor_texture.json
```

[All examples](../../README.md)
