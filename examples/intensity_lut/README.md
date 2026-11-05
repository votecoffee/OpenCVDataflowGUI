# Intensity lookup table

Open [intensity_lut.json](intensity_lut.json) with **File → Open…**, then
click **Run once**. The RGB ramp feeds **Intensity lookup table**, which
raises the red midtones, keeps green unchanged, and reverses the blue ramp.
The input pixel `[64, 64, 64]` becomes `[109, 64, 191]`.

The block is under **Image processing → Contrast**. It accepts unsigned
8-bit and 16-bit images and preserves the input depth and encoding.
Set the shared **Curve** to a list of `[input, output]` points in [0, 1]
with increasing input values. **Linear** interpolates between points.
**Step** holds each output up to the next point. Values beyond the end
points use the nearest end point.

A channel override uses the same format. Channel numbers follow the
stored order, so channel 0 is red in RGB and blue in BGR. Empty overrides
use the shared curve. Alpha is copied by default. Connect a mask or region
to limit which pixels change. [OpenCV LUT reference](https://docs.opencv.org/5.0/main_modules/core_array.html#lut)

```powershell
opencv-dataflow-run examples/intensity_lut/intensity_lut.json
```

[All examples](../README.md)
