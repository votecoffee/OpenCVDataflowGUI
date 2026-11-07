# Derivative-fed Canny

Open [canny_derivatives.json](canny_derivatives.json) with **File → Open…**
and click **Run once**. **Paired Sobel derivatives** emits signed X and Y
derivatives. **Canny from derivatives** uses them to find the edges of
the rectangular and circular parts. The ordinary Canny branch uses
the same thresholds and a 3×3 aperture. **Compare pixels** reports
zero differences.

Both new blocks are under **Image processing → Edges**. Paired Sobel
accepts an 8-bit image and converts color inputs to grayscale. It
emits signed 16-bit derivatives from a 3×3 kernel. Choose reflect-101
or replicate borders. This example uses replicate borders to match
the ordinary Canny branch.
[OpenCV spatial gradient reference](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html)

Derivative-fed Canny accepts aligned signed 16-bit fields with one
or three matching channels. Configure the lower and upper hysteresis
thresholds in derivative units, and choose whether to use Euclidean
gradient magnitude. Custom derivative scales change the appropriate
thresholds.
[OpenCV Canny reference](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html)

A mask or region on either block zeros its output outside the
selection. Apply the selection at Canny to keep the original
derivatives and let hysteresis use the full image before masking.

```powershell
opencv-dataflow-run "examples/opencv tools/canny_derivatives/canny_derivatives.json"
```

[All examples](../../README.md)
