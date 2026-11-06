# Intensity transform

Open [intensity_transform.json](intensity_transform.json) with
**File → Open…**, then click **Run once**. The graph clips a grayscale ramp
to [1, 128], takes its natural logarithm, and uses the exponential to
recover the clipped ramp to floating-point precision.

**Intensity transform** is under **Image processing → Image math**.
Choose **Log**, **Exp**, **Sqrt**, **Power**, or **Clip**. The block
works in raw sample units, so the square root of an 8-bit sample of 100
is 10. The exponent applies to **Power**, and the bounds apply to **Clip**.
Log requires positive samples. Square root requires nonnegative samples.
Fractional powers require nonnegative samples, and negative exponents
require nonzero samples.

Connect a mask or region to limit the transform and domain checks to
selected pixels. Alpha is copied by default. The output defaults to
float64 to retain signed and high-range values. Choose **Same** to
keep the input depth with rounding and saturation for integer pixels.
The **Intensity** preview stretches floating-point outputs for display.
[OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run examples/image_processing/intensity_transform/intensity_transform.json
```

[All examples](../../README.md)
