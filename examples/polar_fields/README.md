# Polar field rotation

Open [polar_fields.json](polar_fields.json) with **File → Open…** and
click **Run once**. Two Sobel blocks measure the signed X and Y
gradients of a synthetic intensity ramp. Away from the image border
they are 16 and −24, giving a magnitude of about 28.84. The graph
converts the gradients to polar fields, adds 90 degrees, then converts
back. The rotated interior components are approximately 24 and 16.
The magnitude is preserved to floating-point precision.

**Cartesian to polar fields** and **Polar to Cartesian fields** are
under **Image processing → Image math**. Connect aligned, monochrome
fields with matching float32 or float64 depth. Both directions retain
the source pixel frame. Positive rotation runs from positive X toward
positive Y, which is clockwise when viewing an image with Y down.

The Cartesian block tags its angle field with the selected units.
The inverse block's **Auto** setting reads that tag and treats
untagged fields as radians. Choose **Radians** or **Degrees** to
override it. The inverse accepts a nonnegative magnitude field. Leave
**Magnitude** disconnected to output unit vectors.

Angles and reconstructed components are approximate. Small
round-trip differences are expected.
[OpenCV array operations reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run examples/polar_fields/polar_fields.json
```

[All examples](../README.md)
