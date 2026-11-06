# Wiener deblur

Open [wiener_deblur.json](wiener_deblur.json) with **File → Open…**
and click **Run once**. The graph blurs a reference image with a
9×9 Gaussian kernel and restores it with the matching point-spread
function (PSF). The two **Image quality** blocks report RMSE against
the sharp reference. RMSE falls from 18.11 to 11.92 gray levels.

**Wiener deblur** is under **Image processing → Filters**. Enter the
Gaussian kernel dimensions and X/Y sigma, or choose **Custom** to
supply an odd-sized table. A connected **Psf** input overrides the
configured model. The custom kernel must contain nonnegative weights,
with the convolution origin at its center. The block normalizes the
weights and exposes the applied kernel on its **Psf** output.

Increase **Noise to signal** to limit noise amplification. Use
**Reflect 101** to extend the working image by reflection or
**Circular** for input blurred with wraparound. Real camera blur
may vary across the image. The block assumes one PSF across the
working image, and border artifacts can remain when the model
differs from the source.

The block processes grayscale or RGB/BGR channels and preserves
alpha. Floating output keeps signed samples in the original units.
A mask or region limits where restored samples are written.
The working image, including reflected padding, is limited to four
million pixels.
[OpenCV Wiener deblurring tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.html)

```powershell
opencv-dataflow-run examples/wiener_deblur/wiener_deblur.json
```

[All examples](../README.md)
