# Image arithmetic

Open [image_arithmetic.json](image_arithmetic.json) with **File → Open…**,
then click **Run once**. A Gaussian blur estimates the background.
**Signed residual** subtracts that image from the source. **Double residual**
multiplies the difference by a scalar of two. The flat patch has zero
residual and the checker texture retains positive and negative values.

**Image arithmetic** is under **Image processing → Image math**. Choose
add, subtract, multiply, divide, minimum, maximum, or absolute difference.
Connect **Image B** or **Scalar B**, or enter a constant **Scalar** in the
inspector. **Scale** applies to multiplication and division. For a zero
divisor, choose whether to report an error, output zero, or keep the
corresponding sample from Image A.

Images must have equal dimensions. A monochrome Image B broadcasts across
the active channels of Image A. RGB and BGR channel order is aligned
automatically. The default frame check compares known coordinate frames.
Select **Pixel grid** when two sources are already aligned by pixel index.

Math uses raw sample units across input depths. The default float64 output
retains signed results. **Same** keeps the depth of Image A and rounds and
saturates integer results. A mask or region limits the operation, and alpha
from Image A is copied by default. [OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run examples/image_arithmetic/image_arithmetic.json
```

[All examples](../README.md)
